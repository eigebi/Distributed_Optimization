import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.linalg import block_diag

'''
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
'''


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

        # only operated locally
        self.assigned_var_index = None 
        # all variables that this agent can access, including local and consensus variables
        self.consensus_var_index = None
        # variables before constraints assignment
        self.var_index = None
        self.L_fi = np.max(np.abs(alpha/4))
        #but remember this is a utilization function, needs to be maximized, or minimize its negative


class AP_problem:
    def __init__(self, num_var, num_agent, num_con, num_problems):
        self.num_var = num_var # list
        self.num_agent = num_agent # 2
        self.num_con = num_con
        self.num_problems = num_problems

        A = []
        for i in range(self.num_agent):
            A.append(np.random.randint(0, 2, (self.num_con, self.num_var[i])))
        self.A = np.concatenate(A, axis=1)
        self.C = np.random.randint(10,20,[self.num_problems, self.num_con])
        temp = np.add.reduceat(self.A, [0,self.num_var[0]], axis=1)
        self.con_assignment = np.argsort(temp)[:,-1]    
        self.local_probs = [[local_AP(self.num_var[i]) for i in range(num_agent)] for _ in range(num_problems)]
        self.global_ub = [np.concatenate([self.local_probs[n][j].ub for j in range(self.num_agent)]) for n in range(num_problems)]
        self.gradient_gx = np.zeros((self.num_agent, 2*np.sum(self.num_var)+self.num_con, np.sum(self.num_var)),dtype=np.float32)
        self.derive_local_id()
        self.local_index = [self.local_probs[0][i].consensus_var_index for i in range(self.num_agent)]
        self.local_var_num = [self.local_probs[0][i].consensus_var_index.shape[0] for i in range(self.num_agent)]
        self.assigned_index = [self.local_probs[0][i].assigned_var_index for i in range(self.num_agent)]
        
        
    def derive_local_id(self):
        # regardless of the number of problems
        for i in range(self.num_agent):
            index_i = np.zeros(sum(self.num_var),dtype=np.int32)
            index_i[np.sum(self.num_var[0:i],dtype=np.int32):np.sum(self.num_var[0:i+1],dtype=np.int32)] = 1
            assigned_index_i = np.where((np.sum(self.A[self.con_assignment==i],axis=0)!=0) & (index_i==0))[0]
            for n in range(self.num_problems):
                self.local_probs[n][i].assigned_var_index = assigned_index_i 
            #consensus_index_i = np.where(\
            #    ((np.sum(self.A[self.con_assignment==i],axis=0)!=0) & (index_i==0)) |\
            #    ((np.sum(self.A[self.con_assignment!=i],axis=0)!=0) & (index_i==1)))[0]
            consensus_index_i = np.where(\
                ((np.sum(self.A[self.con_assignment==i],axis=0)!=0) & (index_i==0)) |\
                ((index_i==1)))[0]   
            for n in range(self.num_problems):
                self.local_probs[n][i].consensus_var_index = consensus_index_i
                self.local_probs[n][i].var_index = np.where(index_i==1)[0]

            # assign local variables to each agent
            temp1 = np.diag(index_i)
            temp2 = np.where(self.con_assignment==i,1,0)[:,np.newaxis] * self.A
            self.gradient_gx[i] = np.concatenate([-temp1, temp1, temp2], axis=0)
       
        


            
    # partial derivative w.r.t. x, or \nabla f + lambda \nabla g
    def gradient_x(self, x, r, agent_id, is_dz=False):
        r = r.detach().numpy()
        if is_dz:
            z = x.detach().numpy()
        else: #from dx
            z = x[:,agent_id,:].detach().numpy()
        dx_full = np.zeros_like(z, dtype=np.float32)
        for p_id in range(self.num_problems):
            local_index = self.local_probs[p_id][agent_id].var_index 
            dx_full[p_id,local_index] = self.local_probs[p_id][agent_id].jac(z[p_id,local_index])
        dx_full += r @ self.gradient_gx[agent_id]
        return dx_full
    
    # partial derivative w.r.t. lambda, or g_x
    def gradient_lambda(self, x):
        x = x.detach().numpy()
        return np.concatenate([-x, x - self.global_ub, x @ self.A.T - self.C], axis=1)
        

    def obj(self, x, p_id):
        obj = np.zeros(self.num_agent, dtype=np.float32)
        #x = x.detach().numpy()
        for i in range(self.num_agent):
            obj[i] = np.sum(self.local_probs[p_id][i].obj(x[self.local_probs[p_id][i].var_index]))
        return np.sum(obj)  
    
    def jac(self, x, p_id):
        jac = np.zeros((self.num_agent, sum(self.num_var)), dtype=np.float32)
        #x = x.detach().numpy()
        for i in range(self.num_agent):
            jac[i, self.local_probs[p_id][i].var_index] = self.local_probs[p_id][i].jac(x[self.local_probs[p_id][i].var_index])
        return np.sum(jac, axis=0)
    
    def L(self, x, r, z, gamma, p_id):
        # x is dx, r is lambda, z is dz
        # x, r, z are all numpy arrays
        # gamma is a scalar
        L = self.obj(x, p_id)
        pass
     
    def opt_solution(self):
        result = []
        for p_id in range(self.num_problems):
            con1 = LinearConstraint(self.A, 0, self.C[p_id])
            con2 = LinearConstraint(np.eye(sum(self.num_var)), 0, self.global_ub[p_id])
            x0 = np.abs(np.random.randn(sum(self.num_var)))
            res = minimize(lambda x : self.obj(x ,p_id), x0,constraints=[con1, con2], jac=lambda x : self.jac(x, p_id))
            result.append(res)
        return result
    