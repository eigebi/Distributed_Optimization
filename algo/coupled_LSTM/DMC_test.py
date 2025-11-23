import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ==========================================
# Part 1: 物理环境与拓扑 (Physics & Topology)
# ==========================================
@dataclass
class EnvCfg:
    # 5G Urban Macro 参数
    fc_GHz: float = 3.5
    bandwidth_Hz: float = 100e6         # 100 MHz
    noise_figure_dB: float = 9.0
    pathloss_exponent: float = 3.6
    shadowing_sigma_dB: float = 8.0
    min_distance_m: float = 35.0
    Pmax_W: float = 80.0                # 80 W
    snr_drop_threshold_db: float = -6.0
    seed: int = 2025
    
    # 优化目标参数 (Net Utility)
    alpha_b: float = 1e-7               # 带宽单价
    alpha_p: float = 0.1                # 功率单价

class StandardTopology:
    def __init__(self, cfg, isd=500.0):
        self.cfg = cfg; self.isd = isd
        self.rng = np.random.default_rng(cfg.seed)

    def generate_hex_bs(self, num_rings=2):
        """生成 19-BS 坐标 (Center + 2 Rings)"""
        locs = [[0.0, 0.0]]
        if num_rings >= 1:
            for i in range(6):
                a = np.radians(30 + 60*i)
                locs.append([self.isd*np.cos(a), self.isd*np.sin(a)])
        if num_rings >= 2:
            for i in range(6):
                a = np.radians(30 + 60*i); locs.append([2*self.isd*np.cos(a), 2*self.isd*np.sin(a)])
            for i in range(6):
                a = np.radians(60 + 60*i); r = np.sqrt(3)*self.isd
                locs.append([r*np.cos(a), r*np.sin(a)])
        return np.array(locs) + 3000.0

    def generate_ues_robust(self, bs_xy, K_per_bs=5, num_slices=3):
        """生成用户并执行准入控制"""
        B = len(bs_xy); ue_list = []; b_u_list = []
        # 计算热噪声功率 (Linear)
        k_B = 1.38e-23; T0 = 290; NF = 10**(self.cfg.noise_figure_dB/10)
        noise_p = k_B * T0 * self.cfg.bandwidth_Hz * NF
        
        for b in range(B):
            cnt = 0
            while cnt < K_per_bs:
                r = self.isd * 0.6 * np.sqrt(self.rng.random())
                th = self.rng.random() * 2 * np.pi
                cand = bs_xy[b] + [r*np.cos(th), r*np.sin(th)]
                d = np.linalg.norm(bs_xy - cand, axis=1)
                if np.min(d) < self.cfg.min_distance_m: continue
                
                # Check SNR
                d0=1.0; fc=self.cfg.fc_GHz*1e3
                pl0 = 32.45 + 20*np.log10(fc) + 20*np.log10(d0/1e3)
                pl = pl0 + 10*self.cfg.pathloss_exponent*np.log10(np.maximum(d[b],1))
                h = 10**(-pl/10)
                if 10*np.log10(self.cfg.Pmax_W*h/noise_p) < self.cfg.snr_drop_threshold_db: continue
                
                ue_list.append(cand); b_u_list.append(b); cnt += 1
        
        K = len(ue_list)
        # 切片分配: eMBB(0) 多一点
        s_u = self.rng.choice(num_slices, size=K, p=[0.5, 0.3, 0.2])
        
        # 计算全局信道矩阵 G [K, B]
        ue_arr = np.array(ue_list); G = np.zeros((K, B))
        d0=1.0; fc=self.cfg.fc_GHz*1e3
        pl0 = 32.45 + 20*np.log10(fc) + 20*np.log10(d0/1e3)
        
        for k in range(K):
            d = np.linalg.norm(bs_xy - ue_arr[k], axis=1)
            pl = pl0 + 10*self.cfg.pathloss_exponent*np.log10(np.maximum(d,1))
            # Shadowing 简化略过，保持梯度平滑
            G[k,:] = 10**(-pl/10)
            
        return bs_xy, ue_arr, np.array(b_u_list), s_u, G

class WirelessEnv:
    def __init__(self, B, K, S, topo_data, cfg: EnvCfg):
        self.cfg = cfg; self.B=B; self.K=K; self.S=S
        self.bs_xy, self.ue_xy, self.b_u, self.s_u, self.G = topo_data
        
        k_B = 1.38e-23; T0 = 290; NF = 10**(cfg.noise_figure_dB/10)
        self.N0_psd = k_B * T0 * NF
        self.Pmax = np.full(B, cfg.Pmax_W)
        self.eps = 1e-9
        
        # QoS & Weights
        self.w_u = np.ones(K)
        self.Rmin_u = np.full(K, 0.1e6) # 默认 0.1 Mbps
        self.Rmin_u[self.s_u == 0] = 5.0e6 # eMBB
        self.w_u[self.s_u == 0] = 5.0      # eMBB 权重高

    def compute_metrics(self, b_vec, p_vec):
        """前向计算: Rate, SINR, Interference"""
        # 1. 聚合功率 P[s, b]
        P_sb = np.zeros((self.S, self.B))
        np.add.at(P_sb, (self.s_u, self.b_u), p_vec)
        
        # 2. 信号
        sig = p_vec * self.G[np.arange(self.K), self.b_u]
        
        # 3. 干扰 (全网同切片 - 本基站)
        P_tx_rel = P_sb[self.s_u, :] 
        rx_tot = np.einsum('kb,kb->k', P_tx_rel, self.G)
        own = P_sb[self.s_u, self.b_u] * self.G[np.arange(self.K), self.b_u]
        interf = rx_tot - own
        
        # 4. Rate
        noise = self.N0_psd * b_vec
        sinr = sig / (interf + noise + 1e-15)
        rate = b_vec * np.log2(1 + sinr)
        return rate, sinr, interf

    def get_local_gradient_separable(self, bs_idx, b_i, p_i, rho_i, I_in_measured):
        """
        计算【自私】梯度 + 本地约束梯度。
        Agent 认为 I_in 是常数 (不计算对邻居的干扰导数)。
        """
        # 1. 索引
        ue_mask = (self.b_u == bs_idx)
        my_s = self.s_u[ue_mask]
        g_ii = self.G[ue_mask, bs_idx]
        my_w = self.w_u[ue_mask]
        my_rmin = self.Rmin_u[ue_mask]
        
        # 2. 前向 (Local View)
        sig = p_i * g_ii
        noise = self.N0_psd * b_i
        denom = I_in_measured + noise + 1e-15
        sinr = sig / denom
        rate = b_i * np.log2(1 + sinr)
        
        # 3. Objective Gradient (Min -Utility + Cost)
        dU_dR = my_w / (self.eps + rate)
        term_sinr = b_i / (np.log(2) * (1 + sinr))
        
        # Grad b
        dSINR_db = - sinr / denom * self.N0_psd
        gb_util = dU_dR * (np.log2(1+sinr) + term_sinr * dSINR_db)
        # Gradient of Loss = - (dU/db - alpha_b)
        # 注意：Cost 只有 BS 0 承担？还是所有人都承担？
        # 按照您的描述：BS 0 承担 Spectrum Usage (rho Cost)，其他人只承担 Power。
        # Bandwidth b 本身在本地是免费的，受 rho 约束。
        grad_b = -gb_util 
        
        # Grad p (Selfish: No Interference Penalty)
        dR_dp = term_sinr * (g_ii / denom)
        gp_util = dU_dR * dR_dp
        grad_p = -gp_util + self.cfg.alpha_p # Power Cost
        
        # Grad rho
        grad_rho = np.zeros_like(rho_i)
        if bs_idx == 0:
            # Manager 承担全网带宽成本
            grad_rho += self.cfg.alpha_b 
            
        # 4. Local Constraint Gradients (Penalty Method)
        # C1: Sum b <= rho * W
        for s in range(self.S):
            mask_s = (my_s == s)
            if not np.any(mask_s): continue
            
            sum_b = np.sum(b_i[mask_s])
            limit = rho_i[s] * self.cfg.bandwidth_Hz
            viol = np.maximum(0, sum_b - limit)
            
            if viol > 0:
                # Penalty L = weight * (sum b - rho W)^2
                # dL/db = 2 * w * viol
                # dL/drho = 2 * w * viol * (-W)
                w_cap = 1e3 / (1e6)**2 # Scaling
                grad_b[mask_s] += 2 * w_cap * viol
                grad_rho[s] += 2 * w_cap * viol * (-self.cfg.bandwidth_Hz)
                
        # C2: QoS (Rate >= Rmin)
        qos_viol = np.maximum(0, my_rmin - rate)
        if np.any(qos_viol > 0):
            w_qos = 1e3
            # Approximate direction: increase b and p
            grad_b -= w_qos * qos_viol 
            grad_p -= w_qos * qos_viol
            
        return grad_b, grad_p, grad_rho, qos_viol

# ==========================================
# Part 2: DMC 算法求解器 (Strictly Distributed)
# ==========================================
class DMCSolver:
    def __init__(self, env):
        self.env = env
        self.N = env.B
        self.S = env.S
        
        # 变量存储 (Structure: [b, p, rho])
        # 我们用 list 存储每个 Agent 的本地变量，模拟分布式内存
        self.agents_x = [] 
        for i in range(self.N):
            K_i = np.sum(env.b_u == i)
            # Init: b=small, p=small, rho=equal
            b = np.ones(K_i) * (env.cfg.bandwidth_Hz / env.K)
            p = np.ones(K_i) * (env.cfg.Pmax_W / 10)
            rho = np.ones(self.S) / self.S
            self.agents_x.append({'b': b, 'p': p, 'rho': rho})
            
        # Global Variables (Scheduler)
        self.z_rho = np.ones(self.S) / self.S
        self.r_global = 0.0 # Global Slice Sum Multiplier
        
        # Multipliers
        self.gamma_rho = np.zeros((self.N, self.S)) # Consensus Dual
        
        # Params
        self.rho_penalty = 5.0
        self.theta = 0.1 # Global constraint step
        
    def solve(self, max_iter=500, lr=1e-5):
        history = {'util': [], 'viol': [], 'consensus': []}
        
        for t in range(max_iter):
            # --- Step 0: Environment Measurement ---
            # 模拟物理环境运行一次，获取每个人受到的干扰 I_in
            # 这需要先收集所有人的 p
            all_p = np.zeros(self.env.K)
            for i in range(self.N):
                idx = np.where(self.env.b_u == i)[0]
                all_p[idx] = self.agents_x[i]['p']
            
            # 实际上我们需要 b 来计算 N0*b
            all_b = np.zeros(self.env.K)
            for i in range(self.N):
                idx = np.where(self.env.b_u == i)[0]
                all_b[idx] = self.agents_x[i]['b']
                
            # 计算全网 Metrics，提取 I_in (Interference + Noise)
            # 注意：compute_metrics 返回的 interf 是纯干扰
            _, _, I_interf_all = self.env.compute_metrics(all_b, all_p)
            
            # --- Step 1: Local Update (Parallel) ---
            # x_i = x_i - lr * ( Grad_Local + Consensus_Penalty )
            
            total_qos_viol = 0
            
            for i in range(self.N):
                # 1. 获取本地数据
                idx = np.where(self.env.b_u == i)[0]
                I_in_i = I_interf_all[idx] # 观测到的干扰
                
                # 2. 计算"自私"梯度 (Separable Objective)
                gb, gp, gr, viol = self.env.get_local_gradient_separable(
                    i, 
                    self.agents_x[i]['b'], 
                    self.agents_x[i]['p'], 
                    self.agents_x[i]['rho'], 
                    I_in_i
                )
                total_qos_viol += np.sum(viol)
                
                # 3. 加上 Consensus Gradient (对 rho)
                # L_con = gamma * (rho - z) + rho_pen/2 * ||rho - z||^2
                # Grad = gamma + rho_pen * (rho - z)
                g_con = self.gamma_rho[i] + self.rho_penalty * (self.agents_x[i]['rho'] - self.z_rho)
                gr += g_con
                
                # 4. 梯度下降更新
                # 注意：带宽和功率是物理量，量级大，需要不同的 LR 或者 Scaling
                # 这里简单处理：Scale lr
                self.agents_x[i]['b'] -= lr * 1e6 * gb
                self.agents_x[i]['p'] -= lr * gp
                self.agents_x[i]['rho'] -= lr * 1e-1 * gr
                
                # 5. 投影 (Projection)
                self.agents_x[i]['b'] = np.maximum(self.agents_x[i]['b'], 1e3)
                self.agents_x[i]['p'] = np.clip(self.agents_x[i]['p'], 1e-6, self.env.Pmax[i])
                self.agents_x[i]['rho'] = np.clip(self.agents_x[i]['rho'], 0.01, 1.0)
                
                # 6. 更新本地 Gamma
                self.gamma_rho[i] += self.rho_penalty * (self.agents_x[i]['rho'] - self.z_rho)

            # --- Step 2: Global Update (Scheduler) ---
            # 1. Update z (Consensus Variable)
            # z = Mean(rho_i + gamma_i/rho_pen)
            # 简化版 ADMM Consensus: z = Mean(rho)
            rho_matrix = np.array([a['rho'] for a in self.agents_x]) # (N, S)
            z_new = np.mean(rho_matrix, axis=0)
            
            # 2. Update r (Global Constraint Multiplier)
            # Constraint: sum(z) <= 1  (Managed by BS 0 via gradients? No, handled here)
            # 您说 "BS 0 管理 slice"，意味着 BS 0 的梯度里有这个惩罚？
            # 在上面 get_local_gradient 里，BS 0 有 alpha_b 成本。
            # 如果是硬约束 sum(rho) <= 1，通常由 Scheduler 投影 z
            
            # 这里我们简单做一个投影：如果 z 之和 > 1，归一化
            if np.sum(z_new) > 1.0:
                z_new = z_new / np.sum(z_new)
            
            # 计算 Consensus Error
            cons_err = np.linalg.norm(rho_matrix - z_new)
            self.z_rho = z_new
            
            # 记录
            # 为了显示 Utility，我们再算一次
            rate, _, _ = self.env.compute_metrics(all_b, all_p)
            u = np.sum(self.env.w_u * np.log(1e-9 + rate))
            history['util'].append(u)
            history['viol'].append(total_qos_viol)
            history['consensus'].append(cons_err)
            
            if t % 50 == 0:
                print(f"Iter {t}: Util={u:.2f}, QoS Viol={total_qos_viol:.4f}, ConsErr={cons_err:.4f}")
                
        return history

# ==========================================
# Part 3: 执行
# ==========================================
if __name__ == "__main__":
    # 1. Setup
    cfg = EnvCfg(bandwidth_Hz=40e6, Pmax_W=40.0) # 宽松一点方便收敛
    topo = StandardTopology(cfg)
    bs_xy = topo.generate_hex_bs(num_rings=1) # 7 BS
    data = topo.generate_ues_robust(bs_xy, K_per_bs=3, num_slices=3)
    env = WirelessEnv(len(bs_xy), len(data[1]), 3, data, cfg)
    
    print(f"Simulating {env.B} BS, {env.K} UEs...")
    
    # 2. Solve
    solver = DMCSolver(env)
    hist = solver.solve(max_iter=500, lr=1e-4)
    
    # 3. Plot
    plt.figure(figsize=(12, 4))
    plt.subplot(131); plt.plot(hist['util']); plt.title('Utility')
    plt.subplot(132); plt.plot(hist['viol']); plt.title('QoS Violation')
    plt.subplot(133); plt.plot(hist['consensus']); plt.title('Consensus Error')
    plt.tight_layout(); plt.show()