import numpy as np
from env_mec import SystemConfig, CentralizedMECEnv
import matplotlib.pyplot as plt

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
        self.z = np.ones((self.N, self.task_per_edge*2))*0.1  # 全局变量向量化存储
        self.lamb = [np.zeros(self.task_per_edge) for _ in range(self.N-1)]  # task 约束 + 总资源约束
        self.lamb.append(np.zeros(self.task_per_edge+1))  # 加一个 总资源约束  # 最后一个Agent
 
        # Params, not complete yet
        self.rho_penalty = 2000
        self.theta = 0
        self.beta = 0
        
    def unpack_x(self, x):
        all_f_partial = x[:self.task_per_edge]
        all_t_partial = x[self.task_per_edge:self.task_per_edge*2]

        return all_f_partial, all_t_partial
    def partial_to_full(self, x,i):
        local_f, local_t = self.unpack_x(x)
        local_f = local_f*(self.env.cfg.f_max_bound-self.env.cfg.f_min_bound) + self.env.cfg.f_min_bound
        local_t = local_t*(self.env.T_max[i*self.task_per_edge:(i+1)*self.task_per_edge]-0.01) + 0.01
        return local_f, local_t
    
        
    def solve(self, max_iter=500, rho=100, theta=0.01, dad=False):
        history = {'util': [], 'viol': [], 'consensus': []}
        z = self.z.copy()  # 全局变量向量化存储
        gamma = self.gamma.copy()  # consensus multiplier 向量化存储
        x = self.agents_x.copy()  # 本地变量向量化存储
        lamb = self.lamb.copy()
        task_per_edge = self.task_per_edge
        beta = self.beta
        #theta = self.theta
        eta = 0 # need to tune 
        N = self.N
        self.rho_penalty = rho

        g_history = []
        g_local = []
        j_loss = []
        
        for t in range(max_iter):
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
            j_temp = 0
            

            g_global = []
            v_temp = []
            obj = []

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
                dgdf  = dgdf*10
                dgdt  = dgdt*10
                g = g*10

                g_local.append([g.copy()])

                dx1 = (grad_f + lamb[i][:task_per_edge]@np.diag(dgdf))*(self.env.cfg.f_max_bound-self.env.cfg.f_min_bound)
                dx2 = (grad_t + lamb[i][:task_per_edge]@np.diag(dgdt))*(self.env.T_max[i*self.task_per_edge:(i+1)*self.task_per_edge]-0.01)
                if i != N-1:
                    dx = np.concatenate([dx1, dx2])
                    latest_x = z[i]-(dx + gamma[i]) / self.rho_penalty
                    j_temp += np.sum((dx+ gamma[i]+rho*(x[i]-z[i]))**2)

                    gamma[i] += self.rho_penalty * (latest_x - z[i])
                else:
                    dx = np.zeros_like(x[i])
                    dx[:N*self.task_per_edge] += lamb[i][-1]*np.ones(N*self.task_per_edge)*(self.env.cfg.f_max_bound-self.env.cfg.f_min_bound)
                    dx[(N-1)*self.task_per_edge: N*self.task_per_edge] += dx1
                    dx[N*self.task_per_edge:] += dx2
                    _z = np.concatenate([z[:,:self.task_per_edge].reshape(-1), z[i][self.task_per_edge:]])
                    latest_x = _z-(dx + gamma[i]) / self.rho_penalty
                    j_temp += np.sum((dx+ gamma[i])**2)

                    
                    gamma[i] += self.rho_penalty * (latest_x - _z)
                    g_ = np.sum(z[:,:self.task_per_edge])*(self.env.cfg.f_max_bound-self.env.cfg.f_min_bound)+0.1*N*self.task_per_edge - self.env.cfg.F_cloud_max
                    g = np.concatenate([g, [g_*1]])
                    g_global.append(g_.copy())
                    
                latest_lambda = np.maximum((lamb[i] + (theta * g))/(1+theta*eta) ,0)
                j_temp += ((lamb[i] > 0).astype(float)) @ (g**2)

                x[i] = latest_x
                lamb[i] = latest_lambda      

            j_loss.append(j_temp/self.N)
            g_history.append(np.mean(np.maximum(np.concatenate(g_local,0)/10,0)**2))
                
            if dad:
                eta=1/(0+1)**(1/4)/rho*10
                beta = 1000 
                eta = 0
            else:
                eta=1/(t+1)**(1/4)/rho
                beta = 1 + 1/eta
                #beta=1000
            
            print(z[0])
        
            #if t % 50 == 0:
            #    print(f"Iter {t}: Util={u:.2f}, QoS Viol={total_qos_viol:.4f}, ConsErr={cons_err:.4f}")
                
        return g_history












if __name__ == "__main__":
    cfg = SystemConfig()
    
    env = CentralizedMECEnv(cfg)

    algo = DMCSolver(env)
    
    iter_num = 1000
    g_1 = algo.solve(max_iter=iter_num, rho = 10, theta = 0.05, dad = False) 
    g_2 = algo.solve(max_iter=iter_num, rho = 10, theta =0.02, dad = False)
    g_3 = algo.solve(max_iter=iter_num, rho = 10, theta =0.01, dad = False)
    g_4 = algo.solve(max_iter=iter_num, rho = 10, theta =0.001, dad = False)
    #g_dad = algo.solve(max_iter=1000, rho = 5000, theta = 0.1, dad = True)
    plt.plot(g_1, 
         label='DMC $\\rho=10$', 
         color='#1f77b4',         # 专业的深蓝色
         linestyle='-',           # 实线
         linewidth=2)            # 【关键】每隔10个点画一个标记，防止太密
    plt.plot(g_2, 
         label='DMC $\\rho=100$', 
         color='#ff7f0e',         # 专业的橙色
         linestyle='-',           # 实线
         linewidth=2)            # 【关键】每隔10个点画一个标记，防止太密
    plt.plot(g_3, 
         label='DMC $\\rho=1000$',          
         color='#2ca02c',         # 专业的绿色
         linestyle='-',           # 实线
         linewidth=2)            # 【关键】每隔10个点画一个标记，防止太密
    plt.plot(g_4, 
         label='DMC $\\rho=10000$',          
         color='#d62728',         # 专业的红色
         linestyle='-',           # 实线
         linewidth=2)            # 【关键】每隔10个点画一个标记

    # 2. 绘制 Gradient Ascent Descent (红色，空心方块 's')
    #plt.yscale('log')

    # --- 装饰设置 ---

    # 设置网格为虚线
    plt.grid(True, linestyle='--', alpha=0.7)

    # 设置图例 (Legend)
    plt.legend(loc='best',           # 自动寻找最佳位置
            frameon=True,         # 显示图例边框
            fancybox=True,        # 圆角边框
            framealpha=0.9,       # 边框透明度
            edgecolor='gray',     # 边框颜色
            fontsize=15)

    # 坐标轴标签
    #plt.xlabel('Iteration number', fontsize=12, fontweight='bold')
    #plt.ylabel('Infeasibility', fontsize=12, fontweight='bold')
    #plt.legend(['Proposed DMC', 'Gradient Ascent Descent'])
    plt.yscale('log')
    plt.xlabel('Iteration number', fontsize=15)
    #plt.xlim(0, 100)
    plt.ylabel('Infeasibility', fontsize=15)
    plt.show()