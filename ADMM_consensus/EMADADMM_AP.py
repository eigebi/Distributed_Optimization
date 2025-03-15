import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.linalg import block_diag

np.random.seed(10000)
class AP:
    def __init__(self, num_agent):
        self.num_agent = num_agent
        location = np.random.randint(0, 100, (num_agent, 2))
        # each sub network connets to at most 2 other sub networks
        distance = np.zeros((num_agent, num_agent))
        for i in range(num_agent):
            for j in range(i+1, num_agent):
                distance[i, j] = np.linalg.norm(location[i] - location[j])
                distance[j, i] = distance[i, j]
        neighbor = np.zeros((num_agent, num_agent), dtype=np.int32)
        for i in range(num_agent):
            index = np.argsort(distance[i])
            neighbor[i, index[1]] = 1
            neighbor[i, index[2]] = 1
        self.neighbor = neighbor
        self.location = location
        # plot the network  topology
        import matplotlib.pyplot as plt
        plt.scatter(location[:, 0], location[:, 1])
        for i in range(num_agent):
            for j in range(num_agent):
                if neighbor[i, j] == 1:
                    plt.plot([location[i, 0], location[j, 0]], [location[i, 1], location[j, 1]])
        plt.show()
        
        # assign edges to nodes
        
class local_AP:
    def __init__(self,num_var):
        alpha = np.random.uniform(-2, -1, num_var)
        beta = np.random.uniform(1, 9, num_var)
        self.ub = 2*beta    # symetric
        self.obj = lambda x: -1/(1+np.exp(alpha*(x-beta)))
        self.jac = lambda x: +(alpha*np.exp(-alpha*(x-beta)))/(1+np.exp(-alpha*(x-beta)))**2
        self.local_var_index = None
        self.consensus_var_index = None
        self.var_index = None
        self.L_fi = np.max(np.abs(alpha/4))
        #but remember this is a utilization function, needs to be maximized, or minimize its negative
class AP_problem:
    def __init__(self, num_var, num_agent, num_con):
        self.num_var = num_var # list
        self.num_agent = num_agent # 2
        self.num_con = num_con
        A = []
        for i in range(self.num_agent):
            A.append(np.random.randint(0, 2, (self.num_con, self.num_var[i])))
        self.A = np.concatenate(A, axis=1)
        self.C = np.random.randint(10,30,self.num_con)
        temp = np.add.reduceat(self.A, [0,self.num_var[0]], axis=1)
        self.con_assignment = np.argsort(temp)[:,-1]
        self.local_probs = [local_AP(self.num_var[i]) for i in range(num_agent)]
        self.global_ub = np.concatenate([self.local_probs[j].ub for j in range(self.num_agent)])
        self.derive_local_id()
        
    def derive_local_id(self):
        for i in range(self.num_agent):
            index_i = np.zeros(sum(self.num_var),dtype=np.int32)
            index_i[np.sum(self.num_var[0:i],dtype=np.int32):np.sum(self.num_var[0:i+1],dtype=np.int32)] = 1
            local_index_i = np.where((np.sum(self.A[self.con_assignment!=i],axis=0)==0) & (index_i==1))[0]
            self.local_probs[i].local_var_index = local_index_i
            consensus_index_i = np.where(\
                ((np.sum(self.A[self.con_assignment==i],axis=0)!=0) & (index_i==0)) |\
                ((np.sum(self.A[self.con_assignment!=i],axis=0)!=0) & (index_i==1)))[0]    
            self.local_probs[i].consensus_var_index = consensus_index_i
            self.local_probs[i].var_index = np.where(index_i==1)[0]
    def gradient_x(self, agent_id, z):
        x_full = np.zeros(sum(self.num_var))
        x_full[self.local_probs[agent_id].var_index] = self.local_probs[agent_id].jac(z[self.local_probs[agent_id].var_index])
        return x_full
    def obj(self, z):
        obj = []
        for i in range(self.num_agent):
            obj.append(self.local_probs[i].obj(z[self.local_probs[i].var_index]))
        return np.sum(obj)   
    def opt_solution(self):
        
        con1 = LinearConstraint(self.A, 0, self.C)
        con2 = LinearConstraint(np.eye(sum(self.num_var)), 0, self.global_ub)
        x0 = np.random.randn(sum(self.num_var))
        res = minimize(self.obj, x0,constraints=[con1, con2])
        return res
    
if __name__ == "__main__":
    num_var = [25, 25]
    num_agent = 2
    num_con = 5
    local_update = False
    dx_from_z = False
    AP = AP_problem(num_var, num_agent, num_con)
    res = AP.opt_solution()
    print(-res.fun,res.x)

    ub = AP.global_ub
    M_g = 1
    
    sigma = 500

    #S=8

    L_f = [AP.local_probs[i].L_fi for i in range(num_agent)]
    L_f = np.max(L_f)
    tau = 5
    rho = (num_agent*tau*L_f)**2+np.sqrt((num_agent*tau*L_f)**4+7*num_agent*num_agent*tau*tau*L_f*L_f*L_f)
    rho = 0.8*rho
    theta = rho*rho/(15*M_g*M_g*(2*rho+7*L_f))
    


    obj_iter = []
    obj_t = []

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
    for t in range(20000):
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
        eta = 1/(rho*np.power(k,1/1))
        beta = 0.9*(num_agent*tau)**2*L_f/2 + theta#+4/(theta*eta*eta)
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
        obj_iter.append(obj)
        print(obj)
    np.save('obj_t_local'+str(local_update)+'dz'+str(dx_from_z)+'g_t'+str(res.fun)+".npy", obj_t)
