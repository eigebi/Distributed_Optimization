import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time

# ==========================================
# 1. 基础配置 (Configuration)
# ==========================================
@dataclass
class EnvCfg:
    # 3GPP UMa 风格参数
    fc_GHz: float = 3.5
    bandwidth_Hz: float = 20e6          # 系统总带宽 W_total (20 MHz)
    noise_figure_dB: float = 7.0
    pathloss_exponent: float = 3.5
    shadowing_sigma_dB: float = 8.0
    min_distance_m: float = 35.0        # 最小距离保护
    Pmax_W: float = 40.0                # 基站最大功率 (40 W)
    
    # 仿真设置
    snr_drop_threshold_db: float = -5.0 # 准入控制：SNR低于此值不接入
    seed: int = 2025

def thermal_noise_psd_w_per_hz(noise_figure_dB: float, T_kelvin: float = 290.0):
    k_B = 1.380649e-23
    N0 = k_B * T_kelvin
    NF = 10.0 ** (noise_figure_dB / 10.0)
    return N0 * NF

def calc_pathloss_linear(d_m: np.ndarray, cfg: EnvCfg, rng: np.random.Generator):
    """计算线性信道增益 (Large-scale Fading)"""
    d0 = 1.0
    fc_MHz = cfg.fc_GHz * 1e3
    PL0 = 32.45 + 20.0 * np.log10(fc_MHz) + 20.0 * np.log10(d0/1e3)
    
    pl_db = PL0 + 10.0 * cfg.pathloss_exponent * np.log10(np.maximum(d_m, d0))
    
    if cfg.shadowing_sigma_dB > 0:
        shadow = rng.normal(0.0, cfg.shadowing_sigma_dB, size=d_m.shape)
        pl_db += shadow
        
    return 10.0 ** (-pl_db / 10.0)

# ==========================================
# 2. 拓扑生成器 (Standard Topology)
# ==========================================
class StandardTopology:
    def __init__(self, cfg: EnvCfg, isd=500.0):
        self.cfg = cfg
        self.isd = isd
        self.rng = np.random.default_rng(cfg.seed)

    def generate_hex_bs(self, num_rings=2):
        """生成 19-BS 六边形布局 (num_rings=2)"""
        locs = [[0.0, 0.0]]
        # Ring 1
        if num_rings >= 1:
            for i in range(6):
                angle = np.radians(30 + 60 * i)
                locs.append([self.isd * np.cos(angle), self.isd * np.sin(angle)])
        # Ring 2
        if num_rings >= 2:
            for i in range(6):
                # Vertex
                angle = np.radians(30 + 60 * i)
                locs.append([2 * self.isd * np.cos(angle), 2 * self.isd * np.sin(angle)])
            for i in range(6):
                # Midpoint
                angle = np.radians(60 + 60 * i)
                r = np.sqrt(3) * self.isd
                locs.append([r * np.cos(angle), r * np.sin(angle)])
        
        return np.array(locs) + 3000.0 # Offset

    def generate_ues_robust(self, bs_xy, K_per_bs=5, num_slices=3):
        """生成用户并分配切片，剔除弱覆盖用户"""
        B = len(bs_xy)
        ue_list = []
        b_u_list = []
        
        noise_psd = thermal_noise_psd_w_per_hz(self.cfg.noise_figure_dB)
        full_noise = noise_psd * self.cfg.bandwidth_Hz
        cell_radius = self.isd * 0.6

        for b in range(B):
            cnt = 0
            center = bs_xy[b]
            while cnt < K_per_bs:
                r = cell_radius * np.sqrt(self.rng.random())
                theta = self.rng.random() * 2 * np.pi
                cand = center + np.array([r * np.cos(theta), r * np.sin(theta)])
                
                dists = np.linalg.norm(bs_xy - cand, axis=1)
                if np.min(dists) < self.cfg.min_distance_m: continue
                
                h_serve = calc_pathloss_linear(np.array([dists[b]]), self.cfg, self.rng)[0]
                snr_db = 10 * np.log10((self.cfg.Pmax_W * h_serve) / full_noise)
                
                if snr_db < self.cfg.snr_drop_threshold_db: continue
                
                ue_list.append(cand)
                b_u_list.append(b)
                cnt += 1
        
        K = len(ue_list)
        s_u = self.rng.choice(num_slices, size=K) # Random Slice Association
        
        ue_arr = np.array(ue_list)
        G = np.zeros((K, B))
        for k in range(K):
            dists = np.linalg.norm(bs_xy - ue_arr[k], axis=1)
            G[k, :] = calc_pathloss_linear(dists, self.cfg, self.rng)
            
        return (bs_xy, ue_arr, np.array(b_u_list), s_u, G)

# ==========================================
# 3. 无线环境 (计算核心)
# ==========================================
class WirelessEnvNumpy:
    def __init__(self, B, K, S, topo_data, cfg: EnvCfg):
        self.cfg = cfg
        self.B, self.K, self.S = B, K, S
        self.bs_xy, self.ue_xy, self.b_u, self.s_u, self.G = topo_data
        self.N0 = thermal_noise_psd_w_per_hz(cfg.noise_figure_dB)
        self.Pmax = np.full(B, cfg.Pmax_W)
        
        # 权重与QoS
        self.w_u = np.ones(K)
        # 差异化 QoS: Slice 0 (eMBB) 高要求, Slice 1/2 低要求
        self.Rmin_u = np.full(K, 0.1e6) 
        self.Rmin_u[self.s_u == 0] = 0.5e6 
        
        self.eps = 1e-9

    def compute_metrics(self, b_vec, p_vec):
        """计算速率与干扰 (Forward)"""
        # 1. 聚合功率: P_total[s, b]
        P_total_sb = np.zeros((self.S, self.B))
        np.add.at(P_total_sb, (self.s_u, self.b_u), p_vec)
        
        # 2. 信号
        signal = p_vec * self.G[np.arange(self.K), self.b_u]
        
        # 3. 干扰 (同切片干扰)
        # 全网该切片发射总能量
        P_tx_relevant = P_total_sb[self.s_u, :] 
        total_rx = np.einsum('kb,kb->k', P_tx_relevant, self.G)
        # 减去本站
        own_bs_power = P_total_sb[self.s_u, self.b_u] * self.G[np.arange(self.K), self.b_u]
        interference = total_rx - own_bs_power
        
        # 4. Rate
        noise = self.N0 * b_vec
        sinr = signal / (interference + noise + 1e-15)
        rate = b_vec * np.log2(1.0 + sinr)
        
        return rate, sinr, interference

    def get_user_level_gradients(self, b_vec, p_vec):
        """
        计算对 b_k, p_k 的精确梯度 (Backward)
        Returns: grad_b, grad_p, metrics
        """
        rate, sinr, interf = self.compute_metrics(b_vec, p_vec)
        
        utility = np.sum(self.w_u * np.log(self.eps + rate))
        
        # Common terms
        dU_dR = self.w_u / (self.eps + rate)
        ln2 = np.log(2.0)
        term_sinr = b_vec / (ln2 * (1.0 + sinr))
        denom = interf + self.N0 * b_vec + 1e-15
        
        # --- Gradient wrt Bandwidth (b) ---
        dSINR_db = - sinr / denom * self.N0
        grad_b = dU_dR * (np.log2(1.0 + sinr) + term_sinr * dSINR_db)
        
        # --- Gradient wrt Power (p) ---
        grad_p = np.zeros(self.K)
        
        # Part A: Local Gain
        dR_dp_self = term_sinr * (self.G[np.arange(self.K), self.b_u] / denom)
        grad_p += dU_dR * dR_dp_self
        
        # Part B: Shared Interference Penalty (Broadcast)
        victim_sens = dU_dR * term_sinr * (-sinr / denom) # dU/dI
        
        Price_sb = np.zeros((self.S, self.B))
        for s in range(self.S):
            mask = (self.s_u == s)
            if not np.any(mask): continue
            
            sens_s = victim_sens[mask]
            G_s = self.G[mask, :]
            
            # Total impact on victims
            total_impact = sens_s @ G_s 
            
            # Remove self-interference (b_u == b)
            own_impact = np.zeros(self.B)
            b_u_s = self.b_u[mask]
            contrib = sens_s * G_s[np.arange(len(sens_s)), b_u_s]
            np.add.at(own_impact, b_u_s, contrib)
            
            Price_sb[s, :] = total_impact - own_impact
            
        # Broadcast penalty to users
        grad_p += Price_sb[self.s_u, self.b_u]
        
        return grad_b, grad_p, {'utility': utility, 'rate': rate}

# ==========================================
# 4. 集中式求解器 (Scipy Baseline)
# ==========================================
def solve_centralized_scipy(env):
    """
    优化变量: 
      - b_vec: (K,) 用户带宽
      - p_vec: (K,) 用户功率
      - rho:   (S,) 全局切片比例 (Explicitly Maintained!)
    
    约束:
      1. sum(rho) <= 1
      2. sum(p_k in BS) <= Pmax
      3. sum(b_k in BS,Slice) <= rho[s] * W_total  <-- 核心切片约束
      4. Rate_k >= Rmin
    """
    print(f"\n=== Centralized Optimization (19 BS, {env.K} UEs) ===")
    
    K, B, S = env.K, env.B, env.S
    W_total = env.cfg.bandwidth_Hz
    
    # Initial Guess
    x0_b = np.ones(K) * (W_total / K)
    x0_p = np.ones(K) * (env.cfg.Pmax_W / 10.0)
    x0_rho = np.ones(S) / S
    x0 = np.concatenate([x0_b, x0_p, x0_rho])
    
    # Split helper
    def unpack(x):
        return x[:K], x[K:2*K], x[2*K:]
    
    # Objective: Maximize Utility (Minimize -Utility)
    def objective(x):
        b, p, _ = unpack(x)
        # 使用解析梯度加速
        gb, gp, metrics = env.get_user_level_gradients(b, p)
        
        f = -metrics['utility']
        # Jacobian: [-gb, -gp, 0] (rho 不直接影响 utility，只在约束里)
        g = np.concatenate([-gb, -gp, np.zeros(S)]) 
        return f, g
    
    # Constraints
    cons = []
    
    # C1: Global Rho Sum <= 1
    cons.append({
        'type': 'ineq', 'fun': lambda x: 1.0 - np.sum(unpack(x)[2]),
        'jac': lambda x: np.concatenate([np.zeros(2*K), -np.ones(S)])
    })
    
    # C2: BS Power Sum <= Pmax
    for b in range(B):
        idx = np.where(env.b_u == b)[0]
        if len(idx) == 0: continue
        def power_con(x, idx=idx, limit=env.Pmax[b]):
            return limit - np.sum(unpack(x)[1][idx])
        def power_jac(x, idx=idx):
            g = np.zeros(2*K + S)
            g[K + idx] = -1.0 # p part
            return g
        cons.append({'type': 'ineq', 'fun': power_con, 'jac': power_jac})
        
    # C3: Local Slice Bandwidth <= Global Rho * W
    # sum_{k in (b,s)} b_k - rho_s * W <= 0  => rho_s * W - sum(...) >= 0
    for b in range(B):
        for s in range(S):
            idx = np.where((env.b_u == b) & (env.s_u == s))[0]
            # 如果某基站某切片没用户，约束自动满足 (rho*W >= 0)
            
            def slice_con(x, idx=idx, s=s):
                b_vec, _, rho = unpack(x)
                used = np.sum(b_vec[idx]) if len(idx) > 0 else 0.0
                return rho[s] * W_total - used
            
            def slice_jac(x, idx=idx, s=s):
                g = np.zeros(2*K + S)
                if len(idx) > 0:
                    g[idx] = -1.0 # b part
                g[2*K + s] = W_total # rho part
                return g
            
            cons.append({'type': 'ineq', 'fun': slice_con, 'jac': slice_jac})
            
    # C4: QoS Rate >= Rmin
    # 非线性约束，不提供 Jacobian (让 scipy 估算) 以防手动推导出错
    def qos_con(x):
        b, p, _ = unpack(x)
        rate, _, _ = env.compute_metrics(b, p)
        return rate - env.Rmin_u
    cons.append({'type': 'ineq', 'fun': qos_con})
    
    # Bounds
    b_bnds = [(1e3, W_total) for _ in range(K)]
    p_bnds = [(1e-6, env.cfg.Pmax_W) for _ in range(K)]
    r_bnds = [(0.01, 1.0) for _ in range(S)]
    bounds = b_bnds + p_bnds + r_bnds
    
    # Optimize
    start_time = time.time()
    print("Starting SLSQP...")
    res = minimize(objective, x0, method='SLSQP', jac=True, 
                   bounds=bounds, constraints=cons, 
                   options={'maxiter': 5000, 'ftol': 1e-3, 'disp': True})
    
    dur = time.time() - start_time
    
    b_opt, p_opt, rho_opt = unpack(res.x)
    _, _, metrics = env.get_user_level_gradients(b_opt, p_opt)
    
    return res, metrics, rho_opt, dur

# ==========================================
# 5. Main Demo
# ==========================================
if __name__ == "__main__":
    # 1. 环境搭建
    cfg = EnvCfg()
    topo_gen = StandardTopology(cfg)
    
    # 生成 19 个基站，每个基站 5 个用户 = 95 UEs
    bs_xy = topo_gen.generate_hex_bs(num_rings=1)
    topo_data = topo_gen.generate_ues_robust(bs_xy, K_per_bs=4, num_slices=3)
    
    env = WirelessEnvNumpy(len(bs_xy), len(topo_data[1]), 3, topo_data, cfg)
    print(f"Map: {env.B} BS, {env.K} UEs. Max Power: {cfg.Pmax_W}W")
    
    # 2. 运行集中式求解
    res, metrics, rho_opt, duration = solve_centralized_scipy(env)
    
    # 3. 输出结果
    print("\n" + "="*40)
    print(f"Optimization Success: {res.success}")
    print(f"Time Elapsed: {duration:.2f} s")
    print(f"Final Utility: {metrics['utility']:.4f}")
    print(f"Avg Rate: {metrics['rate']/1e6:.2f} Mbps")
    print(f"QoS Violations: {np.sum(metrics['qos_viol'] > 0)} users")
    print("-" * 40)
    print(f"Optimized Global Slice Ratios (rho): {np.round(rho_opt, 3)}")
    print(f"Sum Rho: {np.sum(rho_opt):.4f}")
    print("="*40)
    
    # 简单的可视化检查
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(metrics['rate']/1e6, bins=20, alpha=0.7)
    plt.axvline(0.5, color='r', linestyle='--', label='Rmin (0.5M)') # 假设 Rmin 0.5
    plt.xlabel('Rate (Mbps)'); plt.title('User Rate Distribution')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.bar(['Slice 0', 'Slice 1', 'Slice 2'], rho_opt)
    plt.title('Global Slice Bandwidth Allocation')
    plt.tight_layout()
    plt.show()