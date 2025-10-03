#from prob.o_ran_env import *
from o_ran_env import *
from algo.DAMC_LSTM_t.nn_CLSTM import *

import torch
import numpy as np
from matplotlib import pyplot as plt

from scipy.optimize import minimize
from scipy.optimize import LinearConstraint, Bounds, NonlinearConstraint
from scipy.linalg import block_diag

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


def opt_theta(prob, x):
    
    for s in range(3):
        for b in range(B):
            u_set = np.where((prob.s_u==s) & (prob.b_u==b))[0]
            idx = np.intersect1d(u_set, np.where(con_g(x)<0)[0])
            idx_ = np.intersect1d(u_set, np.where(con_g(x)>=0)[0])
            if idx.shape[0] == 0:
                continue
            prob.theta[idx] += 0.02 * np.abs(con_g(x)[idx]/con_g(x)[idx].mean())
            prob.theta[idx_] = np.maximum(0, prob.theta[idx_]-0.01*prob.theta[idx_])
            prob.phi[u_set] = prob.theta[u_set]



if __name__ == "__main__":

    num_epoch = 2000
    num_frame = 20
    num_iteration = 10
    learning_rate_x = 0.01
    learning_rate_lambda = 0.01



    B, K = 15, 300
    prob = FormulationV2(B=B, K=K, alpha_rho=1, alpha_eta=1, slice_probs=(0.1, 0.3, 0.6))
    #rho0 = 0.9 * prob.a_sb / np.maximum(1, prob.a_sb.sum(axis=0, keepdims=True))
    #eta0 = 0.9 * prob.a_sb / np.maximum(1, prob.a_sb.sum(axis=0, keepdims=True))
    #x0 = prob.merge(rho0, eta0)    
    x0 = prob.init_feasible()

    # build Lagrangian and gradient function
    lambda_util = 1.0  # 只缩放效用(∑w log(ε+R))，成本项不缩放


    # bounds: 0 ≤ rho ≤ a, 0 ≤ eta ≤ a
    lb = np.zeros_like(x0)
    #ub = np.concatenate([prob.a_sb.flatten(), prob.a_sb.flatten()])
    ub = np.ones_like(x0)
    bounds = Bounds(lb, ub)



  
    A = block_diag(np.ones([1,3]), np.kron(np.ones(3), np.eye(B)))
    lc_sum_res = LinearConstraint(A, np.zeros(B+1), np.ones(B+1))
    # per-UE min rate constraint
    def con_g(x):
        return prob.per_ue_rate_constraints(x)
    def con_jac(x):
        _, drho, deta = prob.per_ue_constraints_and_jacobians(x)
        return np.concatenate([drho[:,:,np.newaxis], deta], axis=2).reshape(prob.K, -1)
    nlc_g = NonlinearConstraint(con_g, np.zeros(K), np.inf*np.ones(K), jac=con_jac)
    

    # eMBB sum-rate
    # reserved for future use
    '''
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
    '''
    # solve problem using Scipy
    Rmin_map={0: 1e5, 1: 1e4, 2: 1e4}
    prob.Rmin_u = np.array([Rmin_map[int(prob.s_u[u])] for u in range(prob.K)], dtype=np.float64)
    for _ in range(20):
        res = minimize(fun=lambda x: prob.objective_and_grads(x)[0],
                    x0=x0,
                    jac=lambda x: prob.objective_and_grads(x)[1],
                    method='SLSQP',
                    bounds=bounds,
                    constraints=[lc_sum_res, nlc_g],
                    options=dict(maxiter=500, ftol=1e-8, disp=True)) 
                    #options=dict(verbose=3, maxiter=1000, xtol=1e-20, gtol=1e-8)) # option to be determined
        print(np.count_nonzero(con_g(x0)<=0),np.count_nonzero(con_g(res.x)<=0))

        x0 = res.x
        opt_theta(prob, x0)


    Rmin_map={0: 1e6, 1: 1e5, 2: 1e5}
    prob.Rmin_u = np.array([Rmin_map[int(prob.s_u[u])] for u in range(prob.K)], dtype=np.float64)
    for _ in range(10):
        res = minimize(fun=lambda x: prob.objective_and_grads(x)[0],
                    x0=x0,
                    jac=lambda x: prob.objective_and_grads(x)[1],
                    method='SLSQP',
                    bounds=bounds,
                    constraints=[lc_sum_res, nlc_g],
                    options=dict(maxiter=500, ftol=1e-8, disp=True)) 
                    #options=dict(verbose=3, maxiter=1000, xtol=1e-20, gtol=1e-3)) # option to be determined
        print(np.count_nonzero(con_g(x0)<=0),np.count_nonzero(con_g(res.x)<=0))
        x0 = res.x
        opt_theta(prob, x0)
    #rho_opt, eta_opt = prob.split_x(res.x)
    cons_final = con_g(res.x)
    print(np.count_nonzero(con_g(x0)<=0),np.count_nonzero(con_g(res.x)<=0))

    

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
