import torch
import numpy as np
from prob import AP_problem
from utils.nn_CLSTM import *

np.random.seed(10000)



def lambda_proj(r):
    alpha = 2
    _r = torch.where(r < -1, -alpha * r - (alpha - 1), r)
    _r = torch.where(r > 1, alpha * r - (alpha - 1), _r)
    _r = torch.where((r >= -1) & (r <= 1), r ** alpha, _r)
    return _r

def x_proj(x):
    pass

# here the gradient is for the whole batch
def derive_grad_lambda(problems, x, z, r, if_dz):
    # x, z: (num_problem, num_var)
    alpha = 2
    dr = np.where(r < -1, -alpha, 0)
    dr = np.where(r > 1, alpha, dr)
    dr = np.where((r >= -1) & (r <= 1), alpha*r**(alpha-1), dr)
    partial_grad_lambda_p = problems.gradient_lambda(z)
    grad_lambda = dr * partial_grad_lambda_p # to edit
    return grad_lambda

def derive_grad_x(problems, x, z, r, is_dz):
    #r_p = lambda_proj(r)
    r_p = r
    if is_dz:
        input = z
    else:
        input = x
    grad_x = [problems.gradient_x(input, r_p, i, is_dz) for i in range(num_agent)]
    return torch.tensor(grad_x).transpose(1, 0)  # (num_problem, num_agent, num_var)
    


# fits the distributed version of the algorithm
def my_train_true_gradient(flags, paras, problems, model, optimizer):
 
 
    (x_optimizer, lambda_optimizer) = optimizer
    (x_models, lambda_model) = model
    (num_epoch, num_frame, num_iteration, num_var, num_agent, num_con, num_problem, arg_nn) = paras
    if_dz = flags['dz']
    #if_dz = False
    # in traning, we do not use DAMC
    num_layer = 2
    # data
    obj_truth_result = []
    acc = []

    # collect the objective value of the true solution
    obj_truth_result = problems.opt_solution()
    #    obj_truth_result.append(prob.objective(prob.solve().x.reshape(1,-1)))


    # hyperparameters for DAMC
    M_g = 1
    sigma = 500
    L_f = [[problems.local_probs[p][i].L_fi for i in range(num_agent)] for p in range(num_problem)]
    L_f = np.array(L_f)
    L_f = np.max(L_f,axis=1)
    tau = 5
    rho = (num_agent*tau*L_f)**2+np.sqrt((num_agent*tau*L_f)**4+7*num_agent*num_agent*tau*tau*L_f*L_f*L_f)
    #rho = 0.005*rho
    
    theta = rho*rho/(15*M_g*M_g*(2*rho+7*L_f))



    
    #start training: generate multiple initial point and evolve iteratively
    for epoch in range(num_epoch):

        # initialization
        init_x = torch.randn(num_problem, num_agent ,np.sum(num_var))
        init_r = torch.abs(torch.randn(num_problem, len_lambda))

        hidden_x = [(torch.randn(num_layer, num_problem, arg_nn.hidden_size), torch.randn(num_layer, num_problem, arg_nn.hidden_size)) for _ in range(num_agent)]
        hidden_lambda = (torch.randn(num_layer, num_problem, arg_nn.hidden_size), torch.randn(num_layer, num_problem, arg_nn.hidden_size)) # (num_layer, batch_size, hidden_size)

        for x_model in x_models:
            for param in x_model.parameters():
                param.requires_grad = True
        for param in lambda_model.parameters():
            param.requires_grad = True

        reserved_x = init_x 
        reserved_r = init_r

        gamma = torch.zeros((num_problem, num_agent, np.sum(num_var)), dtype=torch.float32)

        z = 20*np.random.randn(num_problem, np.sum(num_var))
        z_graph = np.zeros((np.sum(num_var), num_agent), dtype=np.int32)
        z = np.clip(z, np.zeros_like(z), ub)
        for j in range(np.sum(num_var)):
            for n in range(num_agent):
                if j in problems.local_index[n]:
                    z_graph[j, n] = 1
        z = torch.tensor(z, dtype=torch.float32)

   
           
        # train the model
        for frame in range(num_frame):
            for iter in range(num_iteration):
                k = frame * num_iteration + iter + 1
                eta = 1/(rho*np.power(k,1/4))
                eta= 0.000000001
                eta_ = 1/(rho*np.power(np.maximum(k,0),1/4))
                #eta = 1
                beta = (num_agent*tau)**2*L_f/2 + theta+4/(theta*eta_*eta_)
                beta = 0.5*beta
                beta = 10000

                r = reserved_r
                x = reserved_x # to reshape
                # update lambda via meta-learning
                if iter % 1 == 1:
                    grad_lambda = derive_grad_lambda(problems, x, z, r, if_dz)
                    delta_lambda, hidden_lambda = lambda_model(grad_lambda, h_s=hidden_lambda)
                    hidden_lambda = (hidden_lambda[0].detach(), hidden_lambda[1].detach())

                    _r = r + delta_lambda
                    r = _r.detach()
                    grad_lambda = torch.tensor(derive_grad_lambda(problems, x, z, r, if_dz), dtype = torch.float32)
                    _r.backward(-grad_lambda, retain_graph=True)
                # update lambda via 1st order gradient
                # distribtued update of x
                #r = torch.maximum((torch.tensor(theta)*(torch.cat([-z,z-torch.tensor(problems.global_ub),z @ problems.A.T - problems.C], dim=1)) + r ) / (eta*theta + 1), 0)
                
                z_np = z.detach().numpy()
                x_np = np.concatenate([x[:,i,problems.local_probs[0][i].var_index].detach().numpy() for i in range(num_agent)], axis=1)
                r = torch.tensor(np.maximum((theta[:,np.newaxis]*np.concatenate([-x_np,x_np-problems.global_ub,np.sum(x.detach().numpy()[:,problems.con_assignment] * problems.A,axis = 2)-problems.C], axis=1) + r.detach().numpy() ) / (eta*theta[:,np.newaxis] + 1), 0),dtype=torch.float32)
                #r = torch.tensor(np.maximum((theta[:,np.newaxis]*np.concatenate([-z_np,z_np-problems.global_ub,z.detach().numpy()[:,problems.con_assignment]@ problems.A.T - problems.C], axis=1) + r.detach().numpy() ) / (eta*theta[:,np.newaxis] + 1), 0),dtype=torch.float32)

                grad_x = derive_grad_x(problems, x, z, r, if_dz)
                grad_x += gamma + torch.tensor(rho[:,np.newaxis],dtype=torch.float32) * (x - z)
                


                for i in range(num_agent):
                    delta_x,hidden_x[i] = x_models[i](grad_x[:,i,problems.local_index[i]], h_s=hidden_x[i])
                    hidden_x[i] = (hidden_x[i][0].detach(), hidden_x[i][1].detach())

                    _x = x[:,i,problems.local_index[i]] + delta_x.view(-1, len_x[i])
                    x[:,i,problems.local_index[i]] = _x.detach()
                    #x = torch.clip(x, torch.zeros(num_problem, num_agent, np.sum(num_var)), torch.tensor(ub,dtype=torch.float32))
                    x = torch.clip(x, -100*torch.ones(num_problem, num_agent, np.sum(num_var)), 100*torch.ones(num_problem, num_agent, np.sum(num_var)))
                    # a little bit complicated, gradient calculation could be out the loop

                    grad_x = derive_grad_x(problems, x, z, r, if_dz)
                    grad_x += gamma + torch.tensor(rho[:,np.newaxis], dtype=torch.float32) * (x - z)

                    _x.backward(torch.tensor(grad_x[:,i,problems.local_index[i]]), retain_graph=True)
                    # current L value of agent 0
                L1 = problems.local_probs[0][0].obj(x[:,0,problems.local_probs[0][0].var_index].detach().numpy())
                L2 = problems.local_probs[0][1].obj(x[:,0,problems.local_probs[0][1].var_index].detach().numpy())
                #print("L[0]: ", np.sum(L1),np.sum(L2))  

                # consensus update of gamma and z
                for i in range(num_agent):
                        gamma[:,i,problems.local_index[i]] += torch.tensor(rho[:,np.newaxis],dtype=torch.float32) * (x[:,i,problems.local_index[i]] - z[:,problems.local_index[i]]).detach()
                z = torch.tensor(obj_truth_result[0].x[np.newaxis,:],dtype=torch.float32)
                if iter % 5 == 5:
                    x_temp = torch.zeros((num_problem, num_agent, np.sum(num_var)), dtype=torch.float32)
                    for i in range(num_agent):
                        x_temp[:,i,problems.local_index[i]] = x[:,i,problems.local_index[i]].detach()
                    for k in range(np.sum(num_var)):
                        _id = z_graph[k,:]
                        z[:,k] = (torch.sum(gamma[:,_id==1,k].view(num_problem,-1), dim=1) + torch.tensor(rho,dtype=torch.float32) * torch.sum(x_temp[:,_id==1,k].view(num_problem,-1), dim=1)+torch.tensor(beta,dtype=torch.float32) * z[:,k]) / (torch.tensor(beta,dtype=torch.float32) + np.sum(_id) * torch.tensor(rho,dtype=torch.float32))
                    #z = torch.clip(z, torch.zeros(num_problem,np.sum(num_var)), torch.tensor(ub))
                    #beta = beta * 1.01
                    #print(gamma[0,:,0])



                reserved_r = r
                reserved_x = x
                

            # update the global consensus variables z


                        
            lambda_optimizer.step()
            lambda_optimizer.zero_grad()
            for i in range(num_agent):
                x_optimizer[i].step()
                x_optimizer[i].zero_grad()

            precision  = 0
            x_test = np.zeros_like(z.detach().numpy())
            for p_id in range(num_problem):
                for i in range(num_agent):
                    x_test[p_id, problems.local_index[i]] = x[p_id, i, problems.local_index[i]].detach().numpy()

            r_p = lambda_proj(r)
            for p_id in range(num_problem):
                precision += 1 - np.abs(problems.obj(x_test[p_id],p_id)-obj_truth_result[p_id].fun)/np.abs(obj_truth_result[p_id].fun)
            precision /= num_problem
            acc.append(precision)
            print("epoch: ", epoch ,"precision: ", precision)
            #print("delta: ",  [problems[i].objective(reserved_x[i].view(1,-1).detach().numpy()) for i in range(len(problems))], obj_truth_result)
            # r_p = lambda_proj(r)

            #print("L truth: ", prob(x.detach().numpy(), r_p.detach().numpy()), "obj truth: ", prob.objective(x.detach().numpy()))
            #latest_r = r
            #latest_x = x
            #r = reserved_r
            #x = reserved_x
                    
        #r = latest_r
        #x = latest_x
        # Save models and accuracy array
       
        # print result each iteration
        #print("lambda:",r_p.detach().numpy())
        #print("x:",x.detach().numpy())
        #res = prob.solve()
        #print("opt x: ", res.x)
        #print("constraint function: ", prob.check_con(_x.detach().numpy())[-1])
        #print("opt obj: ", prob.objective(res.x.reshape(1,-1)))
            
    #torch.save(x_model.state_dict(), 'x_model.pth')
    #torch.save(lambda_model.state_dict(), 'lambda_model.pth')
    #np.save('accuracy.npy', np.array(acc))

    # end of iterations
    #np.save('L_train.npy',np.array(L_train_result))
    #np.save('Loss_train.npy',np.array(Loss_train_result))
    #np.save('L_truth.npy',np.array(L_truth_result))
    #np.save('obj_train.npy',np.array(obj_truth_result))











if __name__ == "__main__":
    num_var = [2, 2]
    num_agent = 2
    num_con = 1
    #local_update = False
    #dx_from_z = False
    flags = {
        'dz': True,
        'DAMC': True,
    }
    # repeate the experiment 100 times (test) with randomly generated parameters
    acc = []
    inf = []
    grad_gap = []

    num_problem = 1
    problems = AP_problem(num_var, num_agent, num_con, num_problem)
    ub = np.array(problems.global_ub)

    len_x = problems.local_var_num
    len_lambda = 2*np.sum(num_var) + num_con

    num_epoch = 50
    num_iteration = 20
    num_frame = 20000//20
    class arg_nn:
        hidden_size = 32
        hidden_size_x = 20

    paras = (num_epoch, num_frame, num_iteration, num_var, num_agent, num_con, num_problem, arg_nn)
    # centralized lambda, distributed x
    x_models = [x_LSTM(len_x[i], arg_nn) for i in range(num_agent)]
    lambda_model = lambda_LSTM(len_lambda, arg_nn)

    # if validation
    #x_models[i].load_state_dict(torch.load(f"x_model_{i}.pth")) for i in range(num_agent)]
    #lambda_model.load_state_dict(torch.load('lambda_model.pth'))
    

    # optimizers initialization
    x_optimizers = [torch.optim.Adam(x_models[i].parameters(), lr=0.004) for i in range(num_agent)]
    lambda_optimizer = torch.optim.Adam(lambda_model.parameters(), lr=0.001)

    model = (x_models, lambda_model)
    optimizer = (x_optimizers, lambda_optimizer)

    my_train_true_gradient(flags, paras, problems, model, optimizer)   


















    for test in range(1):
        print("test: ", test)
        AP = AP_problem(num_var, num_agent, num_con)
        res = AP.opt_solution()
        #print(-res.fun,res.x)

        ub = AP.global_ub
        M_g = 1
        
        sigma = 500

        #S=8


        # hyperparameters for DAMC
        L_f = [AP.local_probs[i].L_fi for i in range(num_agent)]
        L_f = np.max(L_f)
        tau = 5
        rho = (num_agent*tau*L_f)**2+np.sqrt((num_agent*tau*L_f)**4+7*num_agent*num_agent*tau*tau*L_f*L_f*L_f)
        rho = 0.005*rho
        theta = rho*rho/(15*M_g*M_g*(2*rho+7*L_f))
        


        obj_iter = []
        obj_t = []

        stationary_gap_t = []
        infeasibility_t = []


        base_p = np.random.uniform(0.3, 0.5, num_agent)
        #base_p = np.ones(N)

        
        z = 20*np.random.randn(np.sum(num_var))
        z = np.clip(z, 0, ub)
        # z is the global variables, while on any local_var_index, they do not update in consensus way
        z_graph = np.zeros((np.sum(num_var), num_agent),dtype=np.int32)
        for i in range(np.sum(num_var)):
            for n in range(num_agent):
                if i in AP.local_probs[n].consensus_var_index or (not local_update and i in AP.local_probs[n].local_var_index):
                    z_graph[i, n] = 1
                    # build consensus graph of z
        dx = np.zeros((num_agent, np.sum(num_var)), dtype=np.float32)
        for i in range(num_agent):
            dx[i,AP.local_probs[i].var_index] = AP.gradient_x(i, z)[AP.local_probs[i].var_index]
        latest_x = np.tile(z, (num_agent, 1))
        gamma = np.zeros((num_agent,np.sum(num_var)), dtype=np.float32)
        sample = np.zeros(num_agent, dtype=np.int32)
        r = np.zeros(num_con, dtype=np.float32)
        delay = np.zeros(num_agent, dtype=np.int32)
        id_not_done = np.ones(num_agent, dtype=np.int32)
        k = 1.0 #time step for global update
        for t in range(10000):
            # calculate the current objective value
            if local_update:
                temp = np.zeros(np.sum(num_var))
                for i in range(num_agent):
                    temp[AP.local_probs[i].local_var_index] = latest_x[i, AP.local_probs[i].local_var_index]
                    temp[AP.local_probs[i].consensus_var_index] = z[AP.local_probs[i].consensus_var_index]
                obj_t.append(AP.obj(temp))  
            else:
                obj_t.append(AP.obj(z))

            # time evolves with delayed agents
            for i in range(num_agent):
                sample[i] = np.random.choice([0,1], p=[base_p[i], 1 - base_p[i]])
            id_not_done = id_not_done * sample
            delay[np.where(id_not_done==1)] += 1
            if np.any(id_not_done==1) and np.max(delay[np.where(id_not_done==1)])>=tau:
                continue

            # local update
            eta = 1/(rho*np.power(k,1/4))
            eta= 0.000000001
            eta_ = 1/(rho*np.power(np.maximum(k,0),1/4))
            #eta = 1
            beta = (num_agent*tau)**2*L_f/2 + theta#+4/(theta*eta_*eta_)
            beta = 0.0005*beta
            for i in np.where(id_not_done==0)[0]:
                delta = dx[i] + (r[:, np.newaxis] * AP.A)[AP.con_assignment == i].sum(axis=0)
                if dx_from_z:
                    latest_x[i, AP.local_probs[i].consensus_var_index] = z[AP.local_probs[i].consensus_var_index] - (delta[AP.local_probs[i].consensus_var_index] + gamma[i, AP.local_probs[i].consensus_var_index]) / rho
                    if local_update:
                        latest_x[i, AP.local_probs[i].local_var_index] = latest_x[i, AP.local_probs[i].local_var_index] - (delta[AP.local_probs[i].local_var_index] ) / rho
                    else:
                        latest_x[i, AP.local_probs[i].local_var_index] = z[AP.local_probs[i].local_var_index] - (delta[AP.local_probs[i].local_var_index] + gamma[i, AP.local_probs[i].local_var_index]) / rho
                else:
                    latest_x[i] = (-delta -gamma[i]+rho*z+beta*latest_x[i]) /( rho+beta)
                latest_x[i] = np.clip(latest_x[i], 0, ub)
                gamma[i]    = gamma[i] + rho * (latest_x[i] - z) 
                #gamma[i,AP.local_probs[i].consensus_var_index] = gamma[i,AP.local_probs[i].consensus_var_index] + rho * (latest_x[i,AP.local_probs[i].consensus_var_index] - z[AP.local_probs[i].consensus_var_index])
                # global update
            #eta = 1/(rho*np.power(k,1/1))
            #beta = 0.2*(num_agent*tau)**2*L_f/2 #+ theta+4/(theta*eta*eta)
            if dx_from_z:
                for i in range(np.sum(num_var)):
                    _id = z_graph[i]
                    z[i] = (np.sum(gamma[_id==1,i])+rho*np.sum(latest_x[_id==1,i])+beta*z[i])/(beta+np.sum(_id)*rho)
            else:
                for i in range(np.sum(num_var)):
                    _id = z_graph[i]
                    z[i] = (np.sum(gamma[_id==1,i])+rho*np.sum(latest_x[_id==1,i]))/(np.sum(_id)*rho)
            z = np.clip(z, 0, ub)
            if local_update:
                for i in range(num_agent):
                    z[AP.local_probs[i].local_var_index] = latest_x[i, AP.local_probs[i].local_var_index]
            r = np.maximum((theta*(AP.A @ z - AP.C) + r ) / (eta*theta + 1), 0)
            k += 1
            for i in np.where(id_not_done==0)[0]:
                if dx_from_z:
                    dx[i] = AP.gradient_x(i, z)
                else:
                    dx[i] = AP.gradient_x(i, latest_x[i])
                delay[i] = 0
            id_not_done = np.ones(num_agent, dtype=np.int32)
            obj = AP.obj(z)
            obj_iter.append(np.abs(res.fun-obj)/np.abs(res.fun))
            gap = 0
            for i in range(num_agent):
                _id = AP.local_probs[i].local_var_index
                delta = dx[i] + (r[:, np.newaxis] * AP.A)[AP.con_assignment == i].sum(axis=0)
                temp = latest_x[i, _id] - np.clip(latest_x[i]-delta/beta/1000, 0, ub)[_id]
                gap += np.sum(1000*beta*np.abs(temp))
            gap /= np.sum(num_var)
            infeasibility = np.sum(np.maximum(AP.A @ z - AP.C,0))/num_con
            stationary_gap_t.append(gap)
            infeasibility_t.append(AP.A @ z - AP.C)
            #print(np.abs(res.fun-obj)/np.abs(res.fun),gap, infeasibility)
        print("test: ", test, "acc: ", np.abs(res.fun-obj)/np.abs(res.fun), "gap: ", gap, "infeasibility: ", infeasibility)
        acc.append(np.array(obj_iter))
        inf.append(np.array(infeasibility_t))
        grad_gap.append(np.array(stationary_gap_t))
    np.save("acc_dx_std_non.npy", np.array(acc,dtype=object))
    np.save("inf_dx_std_non.npy", np.array(inf,dtype=object))
    np.save("grad_gap_dx_std_non.npy", np.array(grad_gap,dtype=object))