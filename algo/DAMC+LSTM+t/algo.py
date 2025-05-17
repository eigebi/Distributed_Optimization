import numpy as np
from prob import AP_problem

np.random.seed(10000)

if __name__ == "__main__":
    num_var = [25, 25]
    num_agent = 2
    num_con = 5
    local_update = False
    dx_from_z = False
    # repeate the experiment 100 times with randomly generated parameters
    acc = []
    inf = []
    grad_gap = []
    for test in range(100):
        print("test: ", test)
        AP = AP_problem(num_var, num_agent, num_con)
        res = AP.opt_solution()
        #print(-res.fun,res.x)

        ub = AP.global_ub
        M_g = 1
        
        sigma = 500

        #S=8

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