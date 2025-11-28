import numpy as np
from env_test import WirelessEnvNumpy, StandardTopology, EnvCfg
import matplotlib.pyplot as plt

class DMCSolver:
    def __init__(self, env):
        self.env = env
        self.N = env.B
        self.S = env.S
        self.UEs_per_BS = [np.count_nonzero(env.b_u == i) for i in range(self.N)]
        self.slice_partial = np.array([0.4,0.2,0.4])  # 假设初始slice bandwidth分配比例已知
        # 变量存储 (Structure: [b, p, rho])
        # 我们用 list 存储每个 Agent 的本地变量，模拟分布式内存
        #self.agents_x = [] # 这个只是辅助变量，用于consensus收敛的; 现在想确保xi,z 存的都是分位数
        self.agents_x = np.ones((self.N, env.K*2))*0.15 # 每个 Agent 的变量向量化存储
        self.gamma = np.zeros((self.N, env.K*2)) # 每个 Agent 的consensus dual向量化存储
        '''for i in range(self.N): # 初始化每个 Agent 的变量
            K_i = np.sum(env.b_u == i)
            # Init: b=small, p=small, rho=equal
            b = np.ones(K_i) * (env.cfg.bandwidth_Hz / env.K)
            p = np.ones(K_i) * (env.cfg.Pmax_W / 10) # 10 somewhaht number of UE per BS
            rho = np.ones(self.S) / self.S
            self.agents_x.append({'b': b, 'p': p, 'rho': rho})
        '''
        self.util_norm_factor = env.util_norm_factor  # utility normalization factor
        # constraints normalization factors
        self.cons_norm_factors = env.cons_norm_factors  # UE rate, BS Power, Slice Bandwidth

        # Global Variables (Scheduler)
        #self.z_rho = np.ones(self.S) / self.S
        #self.r_global = 0.0 # Global Slice Sum Multiplier
        self.z = np.ones(env.K*2)*0.026  # 全局变量向量化存储
        self.lamb = [np.zeros((np.count_nonzero(env.b_u==0)+1+3))]  # 0号BS constraint 个数为UE数量+本地power约束+本地slice带宽约束+全局slice约束
        for i in range(1, self.N):
            self.lamb.append(np.zeros((np.count_nonzero(env.b_u==i)+1+3)))  # 其他BS constraint 个数为UE数量+本地power约束+本地slice带宽约束
 
        # Params, not complete yet
        self.rho_penalty = 1000
        self.theta = 0.0005
        self.beta = 1000
        
    def unpack_x(self, x):
        all_b_partial = x[self.env.K*0:self.env.K]
        all_p_partial = x[self.env.K:self.env.K*2]
        return all_b_partial, all_p_partial
    def partial_to_full(self, x):
        all_b_partial, all_p_partial = self.unpack_x(x) # slice bandwidth in Hz
        all_b_full = np.zeros(self.env.K)
        all_p_full = np.zeros(self.env.K)
        # 获取每个用户的带宽值,这里与slice分的无关
        all_b_full = all_b_partial*self.slice_partial[self.env.s_u] * self.env.cfg.bandwidth_Hz
        # per UE power (compatile with various Pmax)
        all_p_full = all_p_partial * self.env.Pmax[self.env.b_u]
        return all_b_full, all_p_full
    def full_to_partial_grad(self, gb_full, gp_full, slice_full):
        # 把关于真实resource的梯度转换为关于分量时的梯度， 即链式法则，只是放在这里不一定用
        all_b_partial_grad = gb_full * self.env.cfg.bandwidth_Hz #不应该把slice分为的变量带进去，应该是作为约束存在的
        all_p_partial_grad = gp_full * self.env.Pmax[self.env.b_u]
        pass
        return all_b_partial_grad, all_p_partial_grad

    def solve(self, max_iter=500, rho=5000, theta=0.01, dad=False):
        history = {'util': [], 'viol': [], 'consensus': []}
        z = self.z.copy()  # 全局变量向量化存储
        gamma = self.gamma.copy()  # consensus multiplier 向量化存储
        x = self.agents_x.copy()  # 本地变量向量化存储
        lamb = self.lamb.copy()
        UEs_per_BS = self.UEs_per_BS
        beta = self.beta
        #theta = theta
        #rho = rho
        eta = 0 # need to tune 
        g_history = []
        g_local = []
        g_power = []
        g_slice = []

        
        for t in range(max_iter):
            # consensus update for z
            
            next_z = (beta*z+np.sum(rho* x + gamma, 0))/(self.N * rho+beta)
            next_z = np.clip(next_z, 0, 1.0)
            if t==0:
                z = self.z.copy()
            else:
                z = next_z
            
            g_global = []
            v_temp = []
            obj = []

            # --- Step 1: Local Update (Each Agent) ---
            
            for i in range(self.N):
                # 分量还原成真实值
                local_b, local_p = self.partial_to_full(z)
                #local_slice_B = np.ones(self.S)/self.S * self.env.cfg.bandwidth_Hz  # 这里的slice bandwidth是全局变量，不是本地变量
                local_slice_B = self.slice_partial * self.env.cfg.bandwidth_Hz
                # 计算本地目标函数关于本地变量的梯度
                grad_b, grad_p, util_i, _ = self.env.get_net_utility_gradients(local_b, local_p, i)
                # scale
                grad_b = 1/self.util_norm_factor * grad_b
                grad_p = 1/self.util_norm_factor * grad_p
                # 整合梯度，这里还没变成关于分量的梯度
                #dudz = np.concatenate([grad_b, grad_p, np.zeros(self.S)])  # slice部分没在utility里体现
                
                
                # 这里需要计算本地的每个约束关于变量的梯度 (这里就要开始scale了)
                dgdb, dgdp, dgdsb, g = self.env.get_constraints_gradients(local_b, local_p, local_slice_B, i)
                factors = np.repeat(np.array(self.cons_norm_factors), repeats=[UEs_per_BS[i], 1, 3]).reshape(-1,1)
                dgdb = 1/factors * dgdb
                dgdp = 1/factors * dgdp
                #dgdsb = 1/factors * dgdsb
                #g = g[:UEs_per_BS[i]+1]       
                v_temp.append([g])
                
                dx1 = (-grad_b + lamb[i][:UEs_per_BS[i]+1+3]@dgdb)*self.slice_partial[self.env.s_u]*self.env.cfg.bandwidth_Hz
                dx2 = (-grad_p + lamb[i][:UEs_per_BS[i]+1+3]@dgdp)*self.env.Pmax[self.env.b_u]
                #dx3 = lamb[i][:UEs_per_BS[i]+1+3]@dgdsb*self.env.cfg.bandwidth_Hz
                #if i==0:
                #    dx3 += lamb[i][-1]*np.ones(self.S)/self.cons_norm_factors[2]*self.env.cfg.bandwidth_Hz  # 这里的链式法则需要再确认
                dx = np.concatenate([dx1, dx2])
                #   梯度下降更新本地变量
                
                latest_x = z - (dx+ gamma[i]) / rho
                g[:UEs_per_BS[i]] /= self.cons_norm_factors[0]
                g[UEs_per_BS[i]] /= self.cons_norm_factors[1]
                g[UEs_per_BS[i]+1:UEs_per_BS[i]+4] /= self.cons_norm_factors[2]
                # 线性的constraints最好scale一下 不然收敛很慢
                

               
                
                #if i!=0:
                latest_lambda = np.maximum(lamb[i] + (theta * g)/(1+theta*eta) ,0) # 投影到非负正交集
                #else:
                #temp = (np.sum(local_slice_B) - self.env.cfg.bandwidth_Hz) / self.cons_norm_factors[2]*10
                #g_temp = np.concatenate([g, np.array([temp])])
                #latest_lambda = np.maximum(lamb[i] + (theta * g_temp)/(1+theta*eta) ,0) # 投影到非负正交集
                
                gamma[i] += rho * (latest_x - next_z)
                #x[i] = np.clip(latest_x, 0, 1.0)
                x[i] = latest_x
                lamb[i] = latest_lambda
            g_local.append(np.mean(np.maximum(np.concatenate(v_temp),0)[:,:UEs_per_BS[i]]**2))
            g_power.append(np.mean(np.maximum(np.concatenate(v_temp),0)[:,UEs_per_BS[i]:UEs_per_BS[i]+1]**2))
            g_slice.append(np.mean(np.maximum(np.concatenate(v_temp),0)[:,UEs_per_BS[i]+1:]**2))
            g_history.append(np.mean(np.maximum(np.concatenate(v_temp),0)**2))    

            if dad:
                eta=1/(0+1)**(1/4)/rho*10
                beta = 1000 
                eta = 0
            else:
                eta=1/(t+1)**(1/4)/rho
                beta = 1000 + 1000/eta
            
            
            print(z)
        
            #if t % 50 == 0:
            #    print(f"Iter {t}: Util={u:.2f}, QoS Viol={total_qos_viol:.4f}, ConsErr={cons_err:.4f}")
 
        return g_history
        












if __name__ == "__main__":
    cfg = EnvCfg()
    topo = StandardTopology(cfg)
    bs_xy = topo.generate_hex_bs(num_rings=2)
    data = topo.generate_ues_robust(bs_xy, K_per_bs=40, num_slices=3)
    
    env = WirelessEnvNumpy(len(bs_xy), len(data[1]), 3, data, cfg)
    print(f"Topology: {env.B} BS, {env.K} UEs")
    algo = DMCSolver(env)
    iter_num = 1000
    g_1 = algo.solve(max_iter=iter_num, rho = 10, theta = 0.05, dad = False) 
    g_2 = algo.solve(max_iter=iter_num, rho = 100, theta =0.05, dad = False)
    g_3 = algo.solve(max_iter=iter_num, rho = 1000, theta =0.05, dad = False)
    g_4 = algo.solve(max_iter=iter_num, rho = 10000, theta =0.05, dad = False)
    #g_dad = algo.solve(max_iter=1000, rho = 5000, theta = 0.1, dad = True)
    plt.plot(g_1, 
         label='DMC $\\rho=1000$', 
         color='#1f77b4',         # 专业的深蓝色
         linestyle='-',           # 实线
         linewidth=2)            # 【关键】每隔10个点画一个标记，防止太密
    plt.plot(g_2, 
         label='DMC $\\rho=100$', 
         color='#ff7f0e',         # 专业的橙色
         linestyle='-',           # 实线
         linewidth=2)            # 【关键】每隔10个点画一个标记，防止太密
    plt.plot(g_3, 
         label='DMC $\\rho=5000$',          
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
   