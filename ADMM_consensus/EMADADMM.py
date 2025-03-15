import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.linalg import block_diag

np.random.seed(10000)







class local_problem:
    def __init__(self,num_var,prob_arg):
        self.prob_arg = prob_arg
        diag = np.random.randint(5,10,num_var)
        mu = np.random.uniform(0.0001,10,num_var)
        self.L_fi = np.max(np.abs(mu/diag))
        self.L_fi = 10
        
        temp = np.diag(mu)
        self.obj = lambda x: (x/diag-1)@ temp@(x/diag-1) - x@(1/diag)
        self.jac = lambda x: 2/diag * temp@(x/diag-1)- 1/diag
        #self.obj = lambda x: np.sum(-1/(1+np.exp(mu*(x-diag))))
        #self.jac = lambda x: +(mu*np.exp(-mu*(x-diag)))/(1+np.exp(-mu*(x-diag)))**2
        self.ub = np.random.randint(10,20,num_var)
        self.var_index = None


class problem_generator:
    def __init__(self, num_var, num_agent, num_con, arg):
        self.num_var = num_var
        self.num_agent = num_agent
        self.num_con = num_con
        self. arg = arg
        self.A = np.random.randint(0, 2, (self.num_con, self.num_var))
        self.C = np.random.randint(5,10,self.num_con)
        
        self.num_var_local = self.num_var//self.num_agent
        self.num_var_local_ = self.num_var_local + self.num_var%self.num_agent
        temp = np.add.reduceat(self.A, np.arange(0, self.num_var, self.num_var_local),axis=1)
        if self.num_var%self.num_agent!=0:
            temp[:,-2] += temp[:,-1]
            temp = temp[:,:-1]
        self.con_assignment = np.argsort(temp)[:,-1]
        self.local_probs = [local_problem(self.num_var_local,arg) for _ in range(num_agent-1)]
        self.local_probs.append(local_problem(self.num_var_local_, arg))
        self.derive_local_id()
        

    def derive_local_id(self):
        for i in range(self.num_agent):
            index_i = np.zeros(self.num_var,dtype=np.int32)
            if i < self.num_agent - 1:
                index_i[i*self.num_var_local:(i+1)*self.num_var_local] = 1
            else:
                index_i[i*self.num_var_local:] = 1
            index_i = np.where((index_i + np.sum(self.A[self.con_assignment==i],axis=0))>0)[0]
            self.local_probs[i].var_index = index_i

    def gradient_x(self, agent_id, x_i):
        
        if agent_id < self.num_agent - 1:
            indices = np.arange(agent_id * self.num_var_local, (agent_id + 1) * self.num_var_local)
        else:
            indices = np.arange(agent_id * self.num_var_local, self.num_var)
        x_full = np.zeros(self.num_var)
        x_full[indices] = self.local_probs[agent_id].jac(x_i[indices])
        return x_full
    def gradient_r(self, z):
        return self.A@z - self.C
    
    def obj(self, z):
        obj = []
        for i in range(self.num_agent):
            if i < self.num_agent - 1:
                indices = np.arange(i * self.num_var_local, (i + 1) * self.num_var_local)
            else:
                indices = np.arange(i * self.num_var_local, self.num_var)

            obj.append(self.local_probs[i].obj(z[indices]))
        return np.sum(obj) 
    
    def opt_solution(self):
        for i in range(self.num_agent):
            global_ub = np.concatenate([self.local_probs[j].ub for j in range(self.num_agent)])
        con1 = LinearConstraint(self.A, 0, self.C)
        con2 = LinearConstraint(np.eye(self.num_var), 0, global_ub)
        x0 = np.random.randn(self.num_var)
        res = minimize(self.obj, x0,constraints=[con1, con2])
        return res
    

    
if __name__ == '__main__':
    #network = AP(10)

    




    class prob_arg:
        sigma_1 = 1
        mu_1 = 0
        sigma_2 = 1
        mu_2 = 0
        ub = 10
        total_resource = 26
        overall_ub = 5

    N=1
    num_var = 5
    num_con =5
    problems = problem_generator(num_var, N, num_con, prob_arg)
    #problems.gradient_x(0,np.random.randn(1000//16),np.random.randn(20))
    ub = np.concatenate([problems.local_probs[i].ub for i in range(N)])
    opt = problems.opt_solution()
    g_t = opt.fun

    M_g = 1
    
    sigma = 500
    S=8
    L_f = [problems.local_probs[i].L_fi for i in range(N)]
    L_f = np.max(L_f)
    tau = 5
    #rho = (N*tau*L_f)**2+np.sqrt((N*tau*L_f)**4+7*N*N*tau*tau*L_f*L_f*L_f)
    rho = 5
    theta = rho*rho/(15*M_g*M_g*(2*rho+7*L_f))
    


    obj_iter = []
    obj_t = []

    base_p = np.random.uniform(0.9, 1, N)
    #base_p = np.ones(N)
    z = 20*np.random.randn(num_var)
    z = np.clip(z, 0, ub)
    z_graph = np.zeros((num_var, N),dtype=np.int32)
    for i in range(num_var):
        for n in range(N):
            if i in problems.local_probs[n].var_index:
                z_graph[i, n] = 1
    dx = np.zeros((N,num_var), dtype=np.float32)
    for i in range(N):
        dx[i] = problems.gradient_x(i, z)
    latest_x = np.tile(z, (N, 1))
    gamma = np.zeros((N,num_var), dtype=np.float32)
    sample = np.zeros(N, dtype=np.int32)
    r = np.zeros(num_con, dtype=np.float32)
    delay = np.zeros(N, dtype=np.int32)
    id_not_done = np.ones(N, dtype=np.int32)
    k = 1.0 #time step for global update
    for t in range(20000):
        obj_t.append(problems.obj(z))
        for i in range(N):
            sample[i] = np.random.choice([0,1], p=[base_p[i], 1 - base_p[i]])
        id_not_done = id_not_done * sample
        delay[np.where(id_not_done==1)] += 1
        if np.any(id_not_done==1) and np.max(delay[np.where(id_not_done==1)])>=tau:
            continue
        # local update
        for i in np.where(id_not_done==0)[0]:
            delta = dx[i] + (r[:,np.newaxis]*problems.A)[problems.con_assignment == i].sum(axis=0)
            latest_x[i] = z - (delta + gamma[i]) / rho
            latest_x[i] = np.clip(latest_x[i], 0, ub)
            gamma[i] = gamma[i] + rho * (latest_x[i] - z)
        # global update
        eta = 1/(rho*np.power(k,1/4))
        beta = 0.2*(N*tau)**2*L_f/2 + theta#+4/(theta*eta*eta)

        for i in range(num_var):
            _id = z_graph[i]
            z[i] = (np.sum(gamma[_id==1,i])+rho*np.sum(latest_x[_id==1,i])+beta*z[i])/(beta+np.sum(_id)*rho)
        z = np.clip(z, 0, ub)
        r = np.maximum((theta*(problems.A @ z - problems.C) + r ) / (eta*theta + 1), 0)
        k += 1
        for i in np.where(id_not_done==0)[0]:
            dx[i] = problems.gradient_x(i, z)
            delay[i] = 0
        id_not_done = np.ones(N, dtype=np.int32)
        obj = problems.obj(z)
        obj_iter.append(obj)
        print(obj,r,k)
        #np.save('obj_iter_tau'+str(tau)+'g_t'+str(g_t)+".npy", obj_iter)
        #np.save('obj_t_tau'+str(tau)+'g_t'+str(g_t)+".npy", obj_t)
        #np.save('obj_iter_N'+str(N)+'g_t'+str(g_t)+".npy", obj_iter)
        #np.save('obj_t_N'+str(N)+'g_t'+str(g_t)+".npy", obj_t)

    pass

                
