import numpy as np
from env_test import WirelessEnvNumpy, StandardTopology, EnvCfg

class DMCSolver:
    def __init__(self, env):
        self.env = env
        self.N = env.B
        self.S = env.S
        self.UEs_per_BS = [np.count_nonzero(env.b_u == i) for i in range(self.N)]
        # 变量存储 (Structure: [b, p, rho])
        # 我们用 list 存储每个 Agent 的本地变量，模拟分布式内存
        #self.agents_x = [] # 这个只是辅助变量，用于consensus收敛的; 现在想确保xi,z 存的都是分位数
        self.agents_x = np.ones((self.N, env.K*2 + self.S))*0.01 # 每个 Agent 的变量向量化存储
        self.gamma = np.zeros((self.N, env.K*2 + self.S)) # 每个 Agent 的consensus dual向量化存储
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
        self.z = np.ones(env.K*2 + self.S)*0.001  # 全局变量向量化存储
        self.lamb = [np.zeros((np.count_nonzero(env.b_u==0)+1+3+1))]  # 0号BS constraint 个数为UE数量+本地power约束+本地slice带宽约束+全局slice约束
        for i in range(1, self.N):
            self.lamb.append(np.zeros((np.count_nonzero(env.b_u==i)+1+3)))  # 其他BS constraint 个数为UE数量+本地power约束+本地slice带宽约束
 
        # Params, not complete yet
        self.rho_penalty = 500.0
        self.theta = 0.1
        self.beta = 1000
        
    def unpack_x(self, x):
        all_b_partial = x[self.env.K*0:self.env.K]
        all_p_partial = x[self.env.K:self.env.K*2]
        slice_rho_partial = x[self.env.K*2:self.env.K*2+self.S]
        return all_b_partial, all_p_partial, slice_rho_partial
    def partial_to_full(self, x):
        all_b_partial, all_p_partial, slice_partial = self.unpack_x(x)
        slice_b_full = slice_partial * self.env.cfg.bandwidth_Hz # 每个slice的带宽
        all_b_full = np.zeros(self.env.K)
        all_p_full = np.zeros(self.env.K)
        # 获取每个用户的带宽值,这里与slice分的无关
        all_b_full = all_b_partial * self.env.cfg.bandwidth_Hz
        # per UE power (compatile with various Pmax)
        all_p_full = all_p_partial * self.env.Pmax[self.env.b_u]
        return all_b_full, all_p_full, slice_b_full
    def full_to_partial_grad(self, gb_full, gp_full, slice_full):
        # 把关于真实resource的梯度转换为关于分量时的梯度， 即链式法则，只是放在这里不一定用
        all_b_partial_grad = gb_full * self.env.cfg.bandwidth_Hz #不应该把slice分为的变量带进去，应该是作为约束存在的
        all_p_partial_grad = gp_full * self.env.Pmax[self.env.b_u]
        pass
        return all_b_partial_grad, all_p_partial_grad

        
    def solve(self, max_iter=500, lr=1e-5):
        history = {'util': [], 'viol': [], 'consensus': []}
        z = self.z.copy()  # 全局变量向量化存储
        gamma = self.gamma.copy()  # consensus multiplier 向量化存储
        x = self.agents_x.copy()  # 本地变量向量化存储
        lamb = self.lamb.copy()
        UEs_per_BS = self.UEs_per_BS
        beta = self.beta
        theta = self.theta
        eta = 1/self.rho_penalty # need to tune 

        
        for t in range(max_iter):
            # consensus update for z
            next_z = np.sum(beta*z+self.rho_penalty* x + gamma, 0)/(self.N * self.rho_penalty+beta)
            next_z = np.clip(next_z, 0, 1.0)
            z = next_z

            # --- Step 1: Local Update (Each Agent) ---
            
            for i in range(self.N):
                # 分量还原成真实值
                local_b, local_p, local_slice_B = self.partial_to_full(z)
                # 计算本地目标函数关于本地变量的梯度
                grad_b, grad_p,_ , _ = self.env.get_net_utility_gradients(local_b, local_p, i)
                # scale
                grad_b /= self.util_norm_factor
                grad_p /= self.util_norm_factor
                # 整合梯度，这里还没变成关于分量的梯度
                dudz = np.concatenate([grad_b, grad_p, np.zeros(self.S)])  # slice部分没在utility里体现
                
                
                # 这里需要计算本地的每个约束关于变量的梯度 (这里就要开始scale了)
                dgdb, dgdp, dgdsb, g = self.env.get_constraints_gradients(local_b, local_p, local_slice_B, i)
                dgdb /= self.cons_norm_factors[0]
                dgdp /= self.cons_norm_factors[1]
                dgdsb /= self.cons_norm_factors[2]
                dx1 = (-grad_b + lamb[i][:UEs_per_BS[i]+4]@dgdb)*self.env.cfg.bandwidth_Hz
                dx2 = (-grad_p + lamb[i][:UEs_per_BS[i]+4]@dgdp)*self.env.Pmax[self.env.b_u]
                dx3 = lamb[i][:UEs_per_BS[i]+4]@dgdsb*np.ones(self.S)*self.env.cfg.bandwidth_Hz
                if i==0:
                    dx3 += lamb[i][-1]*np.ones(self.S)*self.env.cfg.bandwidth_Hz  # 这里的链式法则需要再确认
                dx = np.concatenate([dx1, dx2, dx3])
                #   梯度下降更新本地变量
                
                latest_x = z - (dx+ gamma[i]) / self.rho_penalty
                g[:UEs_per_BS[i]] /= 10e6
                g[UEs_per_BS[i]] /= self.env.Pmax[self.env.b_u][0]  # 这里的链式法则需要再确认
                g[UEs_per_BS[i]+1:UEs_per_BS[i]+4] /= self.env.cfg.bandwidth_Hz
                # 线性的constraints最好scale一下 不然收敛很慢
                if i!=0:
                    latest_lambda = np.maximum(lamb[i] + (theta * g)/(1+theta*eta) ,0) # 投影到非负正交集
                else:
                    temp = np.concatenate([g, [np.sum(local_slice_B)/self.env.cfg.bandwidth_Hz - 1.0*1000]])
                    latest_lambda = np.maximum(lamb[i] + (theta * temp)/(1+theta*eta) ,0) # 投影到非负正交集

                gamma[i] += self.rho_penalty * (latest_x - next_z)
                x[i] = latest_x
                lamb[i] = latest_lambda

            #beta = 100/eta
            eta=1/(t+1)**4
            print(z)
        
            #if t % 50 == 0:
            #    print(f"Iter {t}: Util={u:.2f}, QoS Viol={total_qos_viol:.4f}, ConsErr={cons_err:.4f}")
                
        return history












if __name__ == "__main__":
    cfg = EnvCfg()
    topo = StandardTopology(cfg)
    bs_xy = topo.generate_hex_bs(num_rings=1)
    data = topo.generate_ues_robust(bs_xy, K_per_bs=5, num_slices=3)
    
    env = WirelessEnvNumpy(len(bs_xy), len(data[1]), 3, data, cfg)
    print(f"Topology: {env.B} BS, {env.K} UEs")
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