import numpy as np
from env_mec import SystemConfig, CentralizedMECEnv

class DMCSolver:
    def __init__(self, env):
        self.env = env
        self.N = env.cfg.num_edges
        self.task_per_edge = env.cfg.tasks_per_edge
        self.K = self.N * self.task_per_edge  # Total number of tasks
        # 变量存储 (Structure: [b, p, rho])
        # 我们用 list 存储每个 Agent 的本地变量，模拟分布式内存
        #self.agents_x = [] # 这个只是辅助变量，用于consensus收敛的; 现在想确保xi,z 存的都是分位数
        self.agents_x = np.ones((self.N,self.task_per_edge)) # 每个 Agent 的变量向量化存储
        self.agents_x = [np.ones(self.task_per_edge*2)*0.001 for _ in range(self.N-1)]
        self.agents_x.append(np.ones(self.task_per_edge*(self.N+1))*0.001) # 最后一个Agent
        self.gamma = np.zeros((self.N,self.task_per_edge)) # 每个 Agent 的consensus dual向量化存储
        self.gamma = [np.zeros(self.task_per_edge*2) for _ in range(self.N-1)]
        self.gamma.append(np.zeros(self.task_per_edge*(self.N+1))) # 最后一个Agent  
        '''for i in range(self.N): # 初始化每个 Agent 的变量
            K_i = np.sum(env.b_u == i)
            # Init: b=small, p=small, rho=equal
            b = np.ones(K_i) * (env.cfg.bandwidth_Hz / env.K)
            p = np.ones(K_i) * (env.cfg.Pmax_W / 10) # 10 somewhaht number of UE per BS
            rho = np.ones(self.S) / self.S
            self.agents_x.append({'b': b, 'p': p, 'rho': rho})
        '''
        #self.util_norm_factor = env.util_norm_factor  # utility normalization factor
        # constraints normalization factors
        #self.cons_norm_factors = env.cons_norm_factors  # UE rate, BS Power, Slice Bandwidth

        # Global Variables (Scheduler)
        #self.z_rho = np.ones(self.S) / self.S
        #self.r_global = 0.0 # Global Slice Sum Multiplier
        self.z = np.ones((self.N, self.task_per_edge*2))*0.001  # 全局变量向量化存储
        self.lamb = [np.zeros(self.task_per_edge) for _ in range(self.N-1)]  # task 约束 + 总资源约束
        self.lamb.append(np.zeros(self.task_per_edge+1))  # 加一个 总资源约束  # 最后一个Agent
 
        # Params, not complete yet
        self.rho_penalty = 2000
        self.theta = 0.1
        self.beta = 1000
        
    def unpack_x(self, x):
        all_f_partial = x[:self.task_per_edge]
        all_t_partial = x[self.task_per_edge:self.task_per_edge*2]

        return all_f_partial, all_t_partial
    def partial_to_full(self, x,i):
        local_f, local_t = self.unpack_x(x)
        local_f = local_f*(self.env.cfg.f_max_bound-self.env.cfg.f_min_bound) + self.env.cfg.f_min_bound
        local_t = local_t*(self.env.T_max[i*self.task_per_edge:(i+1)*self.task_per_edge]-0.01) + 0.01
        return local_f, local_t
    
        
    def solve(self, max_iter=500, lr=1e-5):
        history = {'util': [], 'viol': [], 'consensus': []}
        z = self.z.copy()  # 全局变量向量化存储
        gamma = self.gamma.copy()  # consensus multiplier 向量化存储
        x = self.agents_x.copy()  # 本地变量向量化存储
        lamb = self.lamb.copy()
        task_per_edge = self.task_per_edge
        beta = self.beta
        theta = self.theta
        eta = 1/self.rho_penalty # need to tune 
        N = self.N
        
        for t in range(1000):
            # consensus update for z
            for i in range(self.N-1):
                next_z_f_i = (beta*z[i,:self.task_per_edge] + self.rho_penalty* (x[i][:self.task_per_edge]+x[-1][i*self.task_per_edge:(i+1)*self.task_per_edge]) + (gamma[i][:self.task_per_edge]+gamma[-1][i*self.task_per_edge:(i+1)*self.task_per_edge]))/(2 * self.rho_penalty+beta)
                next_z_t_i = (beta*z[i,self.task_per_edge:] + self.rho_penalty* x[i][self.task_per_edge:] + gamma[i][self.task_per_edge:])/(self.rho_penalty+beta)
                z[i,:self.task_per_edge] = np.clip(next_z_f_i, 0, 1.0)
                z[i,self.task_per_edge:] = np.clip(next_z_t_i, 0, 1.0)
            # last agent
            next_z_N = (beta*z[N-1]+self.rho_penalty* x[N-1][(N-1)*self.task_per_edge:] + gamma[N-1][(N-1)*self.task_per_edge:])/(self.rho_penalty+beta)
            next_z_N = np.clip(next_z_N, 0, 1.0)
            z[N-1] = next_z_N
            



            for i in range(self.N):
                # 分量还原成真实值
   
                local_f, local_t = self.partial_to_full(z[i],i)
                grad_f, grad_t = self.env.get_local_utility_gradients(local_f, local_t,i)
                # --- Step 1: Local Update (Each Agent) ---
                dgdf, dgdt, g = self.env.get_local_constraint_gradients(local_f, local_t,i)
                #grad_f_i = grad_f[i*task_per_edge:(i+1)*task_per_edge]
                #grad_t_i = grad_t[i*task_per_edge:(i+1)*task_per_edge]
                #g_i = g[i*task_per_edge:(i+1)*task_per_edge]
                #dgdf_i = dgdf[i*task_per_edge:(i+1)*task_per_edge]
                #dgdt_i = dgdt[i*task_per_edge:(i+1)*task_per_edge]
                dgdf  = dgdf
                dgdt  = dgdt


                dx1 = (grad_f + lamb[i][:task_per_edge]@dgdf)*(self.env.cfg.f_max_bound-self.env.cfg.f_min_bound)
                dx2 = (grad_t + lamb[i][:task_per_edge]@dgdt)*(self.env.T_max[i*self.task_per_edge:(i+1)*self.task_per_edge]-0.01)
                if i != N-1:
                    dx = np.concatenate([dx1, dx2])
                    latest_x = z[i]-(dx + gamma[i]) / self.rho_penalty
                    gamma[i] += self.rho_penalty * (latest_x - z[i])
                else:
                    dx = np.zeros_like(x[i])
                    dx[:N*self.task_per_edge] += lamb[i][-1]*np.ones(N*self.task_per_edge)*(self.env.cfg.f_max_bound-self.env.cfg.f_min_bound)
                    dx[(N-1)*self.task_per_edge: N*self.task_per_edge] += dx1
                    dx[N*self.task_per_edge:] += dx2
                    _z = np.concatenate([z[:,:self.task_per_edge].reshape(-1), z[i][self.task_per_edge:]])
                    latest_x = _z-(dx + gamma[i]) / self.rho_penalty
                    
                    gamma[i] += self.rho_penalty * (latest_x - _z)
                    g_ = np.sum(z[:,:self.task_per_edge])*(self.env.cfg.f_max_bound-self.env.cfg.f_min_bound)+0.1*N*self.task_per_edge - self.env.cfg.F_cloud_max
                    g = np.concatenate([g, [g_*1]])
                    
                latest_lambda = np.maximum((lamb[i] + (theta * g))/(1+theta*eta) ,0)

                x[i] = latest_x
                lamb[i] = latest_lambda       

                
       
            beta = 1/eta+400
            eta=1/(t+1)**(1/4)/self.rho_penalty
            print(z[0])
        
            #if t % 50 == 0:
            #    print(f"Iter {t}: Util={u:.2f}, QoS Viol={total_qos_viol:.4f}, ConsErr={cons_err:.4f}")
                
        return history












if __name__ == "__main__":
    cfg = SystemConfig()
    
    env = CentralizedMECEnv(cfg)

    algo = DMCSolver(env)
    algo.solve(max_iter=50000, lr=1e-5)
    M_g = 10
    sigma = 100
    L_f = 100
    N = 1 # number of problems/env 暂时先考虑单个环境
    tau = 1
    rho = (N*tau*L_f)**2+np.sqrt((N*tau*L_f)**4+7*N*N*tau*tau*L_f*L_f*L_f)
    theta = rho*rho/(15*M_g*M_g*(2*rho+7*L_f))


    ub = None

    z = np.random.randn(env.K*2)
    z = np.clip(z,0,ub)
    z_graph = None #应该是在env里有类似的能转换

    dx = np.zeros((env.B,env.K*2), dtype=np.float32)
    for i in range(env.B):
        dx[i] = env.get_net_utility_gradients(z[:env.K], z[env.K:], i)