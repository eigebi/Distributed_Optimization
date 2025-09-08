from prob.o_ran_env import *
from algo.DAMC_LSTM_t.nn_CLSTM import *

import torch
import numpy as np
from matplotlib import pyplot as plt



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
    GT = []
    acc = []
    for epoch in range(num_epoch):

        init_x = torch.randn(1, num_agent, num_var)
        init_r = [torch.abs(torch.randn(1, len_lambda[i])) for i in range(num_agent)]  # ensure non-negative

        for x_model in x_models:
            for param in x_model.parameters():
                param.requires_grad = True
        for lambda_model in lambda_models:
            for param in lambda_model.parameters():
                param.requires_grad = True

        reserved_x = init_x
        reserved_r = init_r

        gamma = torch.zeros((1, num_agent, num_var), dtype=torch.float32)

        z = np.random.randn(1, num_agent, num_agent * num_var)  # random noise for constraint
        z = z / np.sum(z, axis= 1)
        z = torch.tensor(z, dtype=torch.float32)

        for frame in range(num_frame):
            for iter in range(num_iteration):
                k = frame * num_iteration + iter + 1
                r = reserved_r
                x = reserved_x

                if iter % 1 ==1:
                    grad_lambda = None
                    # process grad_lambda
                    for i in range(num_agent):
                        delta_lambda, hidden_lambda = lambda_models[i](grad_lambda, None)




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
                options=dict(verbose=3, maxiter=10000, xtol=1e-4, gtol=1e-2)) # option to be determined
    
    rho_opt, eta_opt = unpack_vars(res.x)
    cons_final = prob.constraints_and_jacobians(rho_opt, eta_opt)

    

    class arg_nn:
        hidden_size = 32
        hidden_size_x = 20

    
    x_models = [x_LSTM(2*B, arg_nn) for _ in range(S)]
    len_lambda = [1, B, 2]
    lambda_models = [lambda_LSTM(len_lambda[i], arg_nn) for i in range(S)]

    x_optimizers = [torch.optim.Adam(x_models[i].parameters(), lr=0.004) for i in range(S)]
    lambda_optimizers = [torch.optim.Adam(lambda_models[i].parameters(), lr=0.004) for i in range(S)]
    
    model = (x_models, lambda_models)
    optimizer = (x_optimizers, lambda_optimizers)

    paras = (num_epoch, num_frame, num_iteration, 2*B, S, len_lambda, 1, arg_nn)


    L_train_result, Loss_train_result, L_truth_result, obj_truth_result = my_train_true_gradient(paras, prob, model, optimizer)
