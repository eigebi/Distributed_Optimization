import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.linalg import block_diag

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
        
        #alpha = np.random.uniform(2, 6, num_var)
        #self.obj = lambda x:-np.log(x+1)/np.log(alpha+1)
        #self.jac = lambda x:-1/(x+1)/np.log(alpha+1)

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
        self.C = np.random.randint(10,20,self.num_con)
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
        x0 = np.abs(np.random.randn(sum(self.num_var)))
        res = minimize(self.obj, x0,constraints=[con1, con2])
        return res
    