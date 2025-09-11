from prob.o_ran_env import *
from algo.DAMC_LSTM_t.nn_CLSTM import *

import torch
import numpy as np
from matplotlib import pyplot as plt

sigma = 50 # admm penalty parameter

def dcon_x(x,r,Sidx):
    if Sidx == 0:
        return jac_h_H(x)*r
    elif Sidx == 1:
        return [r@jac_h_L(x)]
    else:
        return [r[:prob.B]@jac_g_rho(x) + r[prob.B:]@jac_g_eta(x)]
    
def DL_lambda(x,Sidx):
    if Sidx == 0:
        return con_h_H(x)
    elif Sidx == 1:
        return con_h_L(x)
    else:
        g_rho, g_eta = con_g_rho(x), con_g_eta(x)
        return np.array([g_rho, g_eta], dtype=object)
    
def local_grad_lambda(x,z,r,num_agent, prob):
    B=prob.d.B
    grad_lambda = []
    for i in range(num_agent):
        local_var = z.copy()
        local_var[i] = x[i]
        local_var = np.concatenate([local_var[:,:B].flatten(),local_var[:,B:].flatten()])
        grad_lambda.append(DL_lambda(local_var,i))
    return grad_lambda


def local_grad_x(x,z,r,gamma,num_agent,prob):
    B=prob.B
    grad_x = []
    for i in range(num_agent):
        local_var = z.copy()
        local_var[i] = x[i]
        local_var = np.concatenate([local_var[:,:B].flatten(),local_var[:,B:].flatten()])
        _, grad = objective_with_lambda(local_var)
        grad = np.concatenate(unpack_vars(grad), axis=1)[i]
        temp_dcon_x = np.concatenate(unpack_vars(dcon_x(local_var,r[i].numpy(),i)[0]), axis=1)
        grad += gamma[i] + sigma*(x[i]-z[i]) + temp_dcon_x[i]
        grad_x.append(grad[:,np.newaxis])
    return grad_x



def lambda_proj(r):
    alpha = 2
    _r = np.where(r < -1, -alpha * r - (alpha - 1), r)
    _r = np.where(r > 1, alpha * r - (alpha - 1), _r)
    _r = np.where((r >= -1) & (r <= 1), r ** alpha, _r)
    # using softplus instead of 
    #return np.log(1+np.exp(r)) 
    return _r

def derive_grad_lambda(z, r):
    # x, z: (num_problem, num_var)
    alpha = 2
    dr = np.where(r < -1, -alpha, 0)
    dr = np.where(r > 1, alpha, dr)
    dr = np.where((r >= -1) & (r <= 1), alpha*r**(alpha-1), dr)
    dr = dr.astype(np.float32)
    #dr = 1-1/(1+np.exp(r))
    partial_grad_lambda_p = np.concatenate([-z,z-6,np.sum(z, keepdims=True)-ub], axis=0)
    grad_lambda = dr * partial_grad_lambda_p 
    return grad_lambda
def derive_grad_lambda_with_x(x, r):
    # x, z: (num_problem, num_var)
    alpha = 2
    dr_set = []
    partial_grad_lambda_p = None
    for i in range(r.shape[0]):
        dr = np.where(r[i] < -1, -alpha, 0)
        dr = np.where(r[i] > 1, alpha, dr)
        dr = np.where((r[i] >= -1) & (r[i] <= 1), alpha*r[i]**(alpha-1), dr)
        dr = dr.astype(np.float32)
        dr_set.append(dr)
    



    x1 = x[0][0]
    x2 = x[1][:,0]
    alpha = 2
    dr = np.where(r < -1, -alpha, 0)
    dr = np.where(r > 1, alpha, dr)
    dr = np.where((r >= -1) & (r <= 1), alpha*r**(alpha-1), dr)
    dr = dr.astype(np.float32)
    #dr = 1-1/(1+np.exp(r))
    partial_grad_lambda_p = np.array([-x1[0],-x2[1],x1[0]-6,x2[1]-6,x2[0]+x2[1]-ub],dtype=np.float32)
    grad_lambda = dr * partial_grad_lambda_p 
    return grad_lambda




def my_train_true_gradient(paras, problems, models, optimizers):

    (x_optimizer, lambda_optimizer) = optimizer
    (x_models, lambda_models) = model
    (num_epoch, num_frame, num_iteration, num_var, num_agent, num_con, num_problem, arg_nn) = paras
    num_layer = 2
    zt = 10  # update frequency of z
    GT = []
    acc = []
    for epoch in range(num_epoch):

        init_x = torch.abs(torch.randn(num_agent, num_var))
        init_r = [torch.abs(torch.randn(len_lambda[i])) for i in range(num_agent)]  # ensure non-negative

        for x_model in x_models:
            for param in x_model.parameters():
                param.requires_grad = True
        for lambda_model in lambda_models:
            for param in lambda_model.parameters():
                param.requires_grad = True

        reserved_x = init_x
        reserved_r = init_r

        gamma = torch.zeros((num_agent, num_var), dtype=torch.float32)

        z = np.abs(np.random.randn( num_agent,  num_var))  # random noise for constraint
        #z = z / np.sum(z, axis= 1)
        z = torch.tensor(z, dtype=torch.float32)

        hidden_x = [(torch.randn(num_layer, 1, arg_nn.hidden_size), torch.randn(num_layer, 1, arg_nn.hidden_size)) for _ in range(num_agent)]
        hidden_lambda = [(torch.randn(num_layer, 1, arg_nn.hidden_size), torch.randn(num_layer, 1, arg_nn.hidden_size)) for i in range(num_agent)]

        for f in range(num_iteration):
            for j in range(num_frame):
                x = reserved_x
                r = reserved_r
                grad_x = local_grad_x(x.numpy(), z.numpy(), r, gamma.numpy(), num_agent, problems)
                _x = []
                for s in range(num_agent):
                    delta, hidden_temp = x_models[s](torch.tensor(grad_x[s], dtype=torch.float32), hidden_x[s])
                    hidden_x[s] = (hidden_temp[0].detach(), hidden_temp[1].detach())
                    temp_new_x = torch.relu(x[s] + delta[:,0,0])
                    _x.append(temp_new_x)
                    x[s] = temp_new_x.detach()
                grad_x = local_grad_x(x.numpy(), z.numpy(), r, gamma.numpy(), num_agent, problems)
                for s in range(num_agent):
                    _x[s].backward(torch.tensor(grad_x[s][:,0], dtype=torch.float32), retain_graph=True)

                if j % zt ==0 :
                    for s in range(num_agent):
                        gamma[s] = gamma[s] + sigma * (x[s] - z[s])
                    z = gamma+ sigma * x
                    z = z / (num_agent * sigma)
                    z = torch.relu(z)




if __name__ == "__main__":

    num_epoch = 2000
    num_frame = 20
    num_iteration = 10
    learning_rate_x = 0.01
    learning_rate_lambda = 0.01

    prob, info = build_demo_instance(N=6, K=40, slice_probs=[0.5,0.3,0.2], seed=1000)
    S, B = prob.S, prob.B

    def pack_vars(rho, eta): return np.concatenate([rho.flatten(), eta.flatten()])
    def unpack_vars(x):
        n=S*B; rho=x[:n].reshape(S,B); eta=x[n:].reshape(S,B); return rho,eta
    
    rho0 = np.where(prob.d.a_sb > 0, 1.0 / np.maximum(1, np.sum(prob.d.a_sb, axis=0, keepdims=True)), 0.0)
    eta0 = rho0.copy()
    x0 = pack_vars(rho0, eta0)

    # build Lagrangian and gradient function
    lambda_util = 1.0  # 只缩放效用(∑w log(ε+R))，成本项不缩放

    def objective_with_lambda(x):
        rho,eta = unpack_vars(x)
        f_base, grad_rho_base, grad_eta_base, extras = prob.objective_and_grad(rho, eta)
        # 将 base 梯度拆回： grad_base = grad_util - grad_cost
        cost_grad_eta = prob.d.beta[:,None]*prob.d.Pmax_W[None,:]
        cost_grad_rho = prob.d.delta[:,None]*prob.d.W
        # 目标：lambda*util - cost
        util = extras['util']
        f_lambda = (lambda_util*np.sum(prob.d.weights*util)
                    - np.sum(prob.d.beta[:,None]*eta*prob.d.Pmax_W[None,:])
                    - np.sum(prob.d.delta[:,None]*rho*prob.d.W))
        # 梯度：lambda*grad_util - grad_cost = lambda*(grad_base+cost_grad) - cost_grad
        grad_rho = lambda_util*(grad_rho_base + cost_grad_rho) - cost_grad_rho
        grad_eta = lambda_util*(grad_eta_base + cost_grad_eta) - cost_grad_eta
        grad = pack_vars(grad_rho, grad_eta)
        return -f_lambda, -grad  # SciPy 是最小化

    # bounds: 0 ≤ rho ≤ a, 0 ≤ eta ≤ a
    lb = np.zeros_like(x0)
    ub = np.concatenate([prob.d.a_sb.flatten(), prob.d.a_sb.flatten()])
    bounds = Bounds(lb, ub)

    # per-BS 资源约束
    def con_g_rho(x):
        rho,eta = unpack_vars(x); return prob.constraints_and_jacobians(rho,eta)['g_rho']
    def jac_g_rho(x):
        rho,eta = unpack_vars(x); J = prob.constraints_and_jacobians(rho,eta)['J_g_rho']
        J_full = np.zeros((B, 2*S*B))
        for b in range(B):
            for s in range(S):
                J_full[b, s*B+b] = J[s,b]
        return J_full
    def con_g_eta(x):
        rho,eta = unpack_vars(x); return prob.constraints_and_jacobians(rho,eta)['g_eta']
    def jac_g_eta(x):
        rho,eta = unpack_vars(x); J = prob.constraints_and_jacobians(rho,eta)['J_g_eta']
        J_full = np.zeros((B, 2*S*B))
        for b in range(B):
            for s in range(S):
                J_full[b, S*B + (s*B+b)] = J[s,b]
        return J_full
    nlc_g_rho = NonlinearConstraint(con_g_rho, -np.inf*np.ones(B), np.zeros(B), jac=jac_g_rho)
    nlc_g_eta = NonlinearConstraint(con_g_eta, -np.inf*np.ones(B), np.zeros(B), jac=jac_g_eta)

    # eMBB sum-rate
    def con_h_H(x):
        rho,eta = unpack_vars(x); return np.array([prob.constraints_and_jacobians(rho,eta)['h_H']])
    def jac_h_H(x):
        rho,eta = unpack_vars(x); cons=prob.constraints_and_jacobians(rho,eta)
        Jrho, Jeta = cons['grad_hH_rho'], cons['grad_hH_eta']
        J_full = np.zeros((1, 2*S*B))
        for s in range(S):
            for b in range(B):
                J_full[0, s*B+b] = Jrho[s,b]
                J_full[0, S*B + (s*B+b)] = Jeta[s,b]
        return J_full
    nlc_h_H = NonlinearConstraint(con_h_H, np.zeros(1), np.inf*np.ones(1), jac=jac_h_H)

    # URLLC per-BS
    def con_h_L(x):
        rho,eta = unpack_vars(x); return prob.constraints_and_jacobians(rho,eta)['h_L']
    def jac_h_L(x):
        rho,eta = unpack_vars(x); cons=prob.constraints_and_jacobians(rho,eta)
        Jrho, Jeta = cons['grad_hL_rho'], cons['grad_hL_eta']  # (B,S,B)
        J_full = np.zeros((B, 2*S*B))
        for b in range(B):
            for s in range(S):
                for c in range(B):
                    J_full[b, s*B + c] += Jrho[b,s,c]
                    J_full[b, S*B + (s*B+c)] += Jeta[b,s,c]
        return J_full
    nlc_h_L = NonlinearConstraint(con_h_L, np.zeros(B), np.inf*np.ones(B), jac=jac_h_L)

    # solve problem using Scipy
    res = minimize(fun=lambda x: objective_with_lambda(x)[0],
                x0=x0,
                jac=lambda x: objective_with_lambda(x)[1],
                method='trust-constr',
                bounds=bounds,
                constraints=[nlc_g_rho, nlc_g_eta, nlc_h_H, nlc_h_L],
                options=dict(verbose=3, maxiter=100, xtol=1e-4, gtol=1e-2)) # option to be determined
    
    rho_opt, eta_opt = unpack_vars(res.x)
    cons_final = prob.constraints_and_jacobians(rho_opt, eta_opt)

    

    class arg_nn:
        hidden_size = 32
        hidden_size_x = 20

    
    x_models = [x_LSTM(1, arg_nn) for _ in range(S)]
    len_lambda = [1, B, 2*B]
    lambda_models = [x_LSTM(1, arg_nn) for _ in range(S)]

    x_optimizers = [torch.optim.Adam(x_models[i].parameters(), lr=0.004) for i in range(S)]
    lambda_optimizers = [torch.optim.Adam(lambda_models[i].parameters(), lr=0.004) for i in range(S)]
    
    model = (x_models, lambda_models)
    optimizer = (x_optimizers, lambda_optimizers)

    paras = (num_epoch, num_frame, num_iteration, 2*B, S, len_lambda, 1, arg_nn)


    L_train_result, Loss_train_result, L_truth_result, obj_truth_result = my_train_true_gradient(paras, prob, model, optimizer)
