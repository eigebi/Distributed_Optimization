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
    fc_GHz: float = 3.5
    bandwidth_Hz: float = 100e6          # 100 MHz
    noise_figure_dB: float = 7.0
    pathloss_exponent: float = 3.5
    shadowing_sigma_dB: float = 8.0
    min_distance_m: float = 35.0        # 最小距离
    Pmax_W: float = 80.0                # 40 W (46 dBm)
    snr_drop_threshold_db: float = -5.0 # 准入控制
    seed: int = 2025

def thermal_noise_psd_w_per_hz(noise_figure_dB: float, T_kelvin: float = 290.0):
    k_B = 1.380649e-23
    N0 = k_B * T_kelvin
    NF = 10.0 ** (noise_figure_dB / 10.0)
    return N0 * NF

def calc_pathloss_linear(d_m: np.ndarray, cfg: EnvCfg, rng: np.random.Generator):
    d0 = 1.0
    fc_MHz = cfg.fc_GHz * 1e3
    PL0 = 32.45 + 20.0 * np.log10(fc_MHz) + 20.0 * np.log10(d0/1e3)
    pl_db = PL0 + 10.0 * cfg.pathloss_exponent * np.log10(np.maximum(d_m, d0))
    if cfg.shadowing_sigma_dB > 0:
        pl_db += rng.normal(0.0, cfg.shadowing_sigma_dB, size=d_m.shape)
    return 10.0 ** (-pl_db / 10.0)

# ==========================================
# 2. 拓扑生成器
# ==========================================
class StandardTopology:
    def __init__(self, cfg: EnvCfg, isd=500.0):
        self.cfg = cfg
        self.isd = isd
        self.rng = np.random.default_rng(cfg.seed)

    def generate_hex_bs(self, num_rings=2):
        """生成 19-BS 坐标"""
        locs = [[0.0, 0.0]]
        if num_rings >= 1:
            for i in range(6):
                angle = np.radians(30 + 60 * i)
                locs.append([self.isd * np.cos(angle), self.isd * np.sin(angle)])
        if num_rings >= 2:
            for i in range(6):
                angle = np.radians(30 + 60 * i)
                locs.append([2 * self.isd * np.cos(angle), 2 * self.isd * np.sin(angle)])
            for i in range(6):
                angle = np.radians(60 + 60 * i)
                r = np.sqrt(3) * self.isd
                locs.append([r * np.cos(angle), r * np.sin(angle)])
        return np.array(locs) + 3000.0

    def generate_ues_robust(self, bs_xy, K_per_bs=5, num_slices=3):
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
        s_u = self.rng.choice(num_slices, size=K, p=[0.1, 0.3, 0.6])
        ue_arr = np.array(ue_list)
        G = np.zeros((K, B))
        for k in range(K):
            dists = np.linalg.norm(bs_xy - ue_arr[k], axis=1)
            G[k, :] = calc_pathloss_linear(dists, self.cfg, self.rng)
            
        return (bs_xy, ue_arr, np.array(b_u_list), s_u, G)

# ==========================================
# 3. 无线环境 (User-Level) - 已修正 Metrics 返回
# ==========================================
class WirelessEnvNumpy:
    def __init__(self, B, K, S, topo_data, cfg: EnvCfg):
        self.cfg = cfg
        self.B, self.K, self.S = B, K, S
        self.bs_xy, self.ue_xy, self.b_u, self.s_u, self.G = topo_data
        self.N0 = thermal_noise_psd_w_per_hz(cfg.noise_figure_dB)
        self.Pmax = np.full(B, cfg.Pmax_W)
        self.w_u = np.ones(K)
        
        # 差异化 QoS
        self.Rmin_u = np.full(K, 0.05e6)
        self.Rmin_u[self.s_u == 0] = 1.5e6 
        self.eps = 1e-9

    def compute_metrics(self, b_vec, p_vec):
        P_total_sb = np.zeros((self.S, self.B))
        np.add.at(P_total_sb, (self.s_u, self.b_u), p_vec)
        
        signal = p_vec * self.G[np.arange(self.K), self.b_u]
        P_tx_relevant = P_total_sb[self.s_u, :] 
        total_rx = np.einsum('kb,kb->k', P_tx_relevant, self.G)
        own_bs_power = P_total_sb[self.s_u, self.b_u] * self.G[np.arange(self.K), self.b_u]
        interference = total_rx - own_bs_power
        
        noise = self.N0 * b_vec
        sinr = signal / (interference + noise + 1e-15)
        rate = b_vec * np.log2(1.0 + sinr)
        return rate, sinr, interference

    def get_user_level_gradients(self, b_vec, p_vec):
        rate, sinr, interf = self.compute_metrics(b_vec, p_vec)
        utility = np.sum(self.w_u * np.log(self.eps + rate))
        
        dU_dR = self.w_u / (self.eps + rate)
        ln2 = np.log(2.0)
        term_sinr = b_vec / (ln2 * (1.0 + sinr))
        denom = interf + self.N0 * b_vec + 1e-15
        
        # Gradients
        dSINR_db = - sinr / denom * self.N0
        grad_b = dU_dR * (np.log2(1.0 + sinr) + term_sinr * dSINR_db)
        
        grad_p = np.zeros(self.K)
        dR_dp_self = term_sinr * (self.G[np.arange(self.K), self.b_u] / denom)
        grad_p += dU_dR * dR_dp_self
        
        victim_sens = dU_dR * term_sinr * (-sinr / denom)
        Price_sb = np.zeros((self.S, self.B))
        
        for s in range(self.S):
            mask = (self.s_u == s)
            if not np.any(mask): continue
            sens_s = victim_sens[mask]
            G_s = self.G[mask, :]
            total_impact = sens_s @ G_s 
            own_impact = np.zeros(self.B)
            b_u_s = self.b_u[mask]
            contrib = sens_s * G_s[np.arange(len(sens_s)), b_u_s]
            np.add.at(own_impact, b_u_s, contrib)
            Price_sb[s, :] = total_impact - own_impact
            
        grad_p += Price_sb[self.s_u, self.b_u]
        
        qos_viol = np.maximum(0, self.Rmin_u - rate)

        qos_con_func = rate - self.Rmin_u

        
        # --- FIX: 返回完整的向量数据 ---
        metrics = {
            'utility': utility,
            'avg_rate': np.mean(rate),
            'qos_viol_sum': np.sum(qos_viol),
            'rate': rate,          # [K] 向量，用于画直方图
            'qos_viol': qos_viol   # [K] 向量，用于统计违约数
        }
        
        return grad_b-1e-8, grad_p-0.01, metrics
    
    def distributed_gradients(self, b_vec, p_vec, i = 0):
        if i is not None:
            b_temp = np.zeros_like(b_vec)
            b_temp[self.b_u == i] = b_vec[self.b_u == i]
        rate, sinr, interf = self.compute_metrics(b_vec, p_vec)
        utility = np.sum(self.w_u * np.log(self.eps + rate))
        
        dU_dR = self.w_u / (self.eps + rate)
        ln2 = np.log(2.0)
        term_sinr = b_vec / (ln2 * (1.0 + sinr))
        denom = interf + self.N0 * b_vec + 1e-15
        
        # Gradients
        dSINR_db = - sinr / denom * self.N0
        grad_b = dU_dR * (np.log2(1.0 + sinr) + term_sinr * dSINR_db)
        
        grad_p = np.zeros(self.K)
        dR_dp_self = term_sinr * (self.G[np.arange(self.K), self.b_u] / denom)
        grad_p += dU_dR * dR_dp_self
        
        victim_sens = dU_dR * term_sinr * (-sinr / denom)
        Price_sb = np.zeros((self.S, self.B))
        
        for s in range(self.S):
            mask = (self.s_u == s)
            if not np.any(mask): continue
            sens_s = victim_sens[mask]
            G_s = self.G[mask, :]
            total_impact = sens_s @ G_s 
            own_impact = np.zeros(self.B)
            b_u_s = self.b_u[mask]
            contrib = sens_s * G_s[np.arange(len(sens_s)), b_u_s]
            np.add.at(own_impact, b_u_s, contrib)
            Price_sb[s, :] = total_impact - own_impact
            
        grad_p += Price_sb[self.s_u, self.b_u]
        
        qos_viol = np.maximum(0, self.Rmin_u - rate)
        
        # --- FIX: 返回完整的向量数据 ---
        metrics = {
            'utility': utility,
            'avg_rate': np.mean(rate),
            'qos_viol_sum': np.sum(qos_viol),
            'rate': rate,          # [K] 向量，用于画直方图
            'qos_viol': qos_viol   # [K] 向量，用于统计违约数
        }
        
        return grad_b, grad_p, metrics

# ==========================================
# 4. 集中式求解器 (Scipy)
# ==========================================
def solve_centralized_scipy(env):
    K, B, S = env.K, env.B, env.S
    W_total = env.cfg.bandwidth_Hz
    
    # Init Guess
    x0_b = np.ones(K) * (W_total / K)
    x0_p = np.ones(K) * (env.cfg.Pmax_W / 10.0)
    x0_rho = np.ones(S) / S
    x0 = np.concatenate([x0_b, x0_p, x0_rho])
    
    def unpack(x): return x[:K], x[K:2*K], x[2*K:]
    
    def objective(x):
        b, p, _ = unpack(x)
        gb, gp, m = env.get_user_level_gradients(b, p)
        f = -m['utility']
        g = np.concatenate([-gb, -gp, np.zeros(S)]) 
        return f, g
    
    cons = []
    # sum(rho) <= 1
    cons.append({'type': 'ineq', 'fun': lambda x: 1.0 - np.sum(unpack(x)[2]), 
                 'jac': lambda x: np.concatenate([np.zeros(2*K), -np.ones(S)])})
    
    # sum(p) <= Pmax per BS
    for b in range(B):
        idx = np.where(env.b_u == b)[0]
        if len(idx) == 0: continue
        def p_con(x, idx=idx, plim=env.Pmax[b]): return plim - np.sum(unpack(x)[1][idx])
        def p_jac(x, idx=idx):
            g = np.zeros(2*K + S); g[K + idx] = -1.0; return g
        cons.append({'type': 'ineq', 'fun': p_con, 'jac': p_jac})
    
    # sum(b) <= rho[s]*W per BS/Slice
    for b in range(B):
        for s in range(S):
            idx = np.where((env.b_u == b) & (env.s_u == s))[0]
            def s_con(x, idx=idx, s=s):
                b_vec, _, rho = unpack(x)
                return rho[s] * W_total - (np.sum(b_vec[idx]) if len(idx)>0 else 0.0)
            def s_jac(x, idx=idx, s=s):
                g = np.zeros(2*K + S); 
                if len(idx)>0: g[idx] = -1.0
                g[2*K + s] = W_total; return g
            cons.append({'type': 'ineq', 'fun': s_con, 'jac': s_jac})
            
    # Rate >= Rmin
    def qos_con(x):
        b, p, _ = unpack(x)
        rate, _, _ = env.compute_metrics(b, p)
        return rate - env.Rmin_u
    cons.append({'type': 'ineq', 'fun': qos_con})
    
    bnds = [(1e3, W_total)]*K + [(1e-6, env.cfg.Pmax_W)]*K + [(0.01, 1.0)]*S
    
    print("Starting SLSQP optimization (this may take 10-20s)...")
    start = time.time()
    res = minimize(objective, x0, method='SLSQP', jac=True, bounds=bnds, constraints=cons, 
                   options={'maxiter': 500, 'ftol': 1e-3, 'disp': True})
    dur = time.time() - start
    
    b_opt, p_opt, rho_opt = unpack(res.x)
    _, _, metrics = env.get_user_level_gradients(b_opt, p_opt)
    return res, metrics, rho_opt, dur

# ==========================================
# 5. Main Demo (修正版)
# ==========================================
if __name__ == "__main__":
    cfg = EnvCfg()
    topo_gen = StandardTopology(cfg)
    bs_xy = topo_gen.generate_hex_bs(num_rings=1)
    topo_data = topo_gen.generate_ues_robust(bs_xy, K_per_bs=8, num_slices=3)
    
    env = WirelessEnvNumpy(len(bs_xy), len(topo_data[1]), 3, topo_data, cfg)
    print(f"\nTopology: {env.B} BS, {env.K} UEs generated.")
    
    # 运行求解
    res, metrics, rho_opt, duration = solve_centralized_scipy(env)
    
    # --- 修正后的输出逻辑 ---
    print("\n" + "="*40)
    print(f"Optimization Success: {res.success}")
    print(f"Message: {res.message}")
    print(f"Time: {duration:.2f}s")
    print("-" * 40)
    print(f"Final Utility: {metrics['utility']:.4f}")
    print(f"Avg Rate: {metrics['avg_rate']/1e6:.2f} Mbps")
    
    # 这里的 qos_viol 现在是向量，我们可以正确计算了
    viol_count = np.sum(metrics['qos_viol'] > 1e-6) # 允许微小误差
    print(f"QoS Violations: {viol_count} / {env.K} users")
    print(f"Global Slice Ratios (rho): {np.round(rho_opt, 3)}")
    print("="*40)
    
    # --- 可视化 ---
    plt.figure(figsize=(12, 5))
    
    # 1. 速率 CDF
    plt.subplot(1, 2, 1)
    rates_mbps = metrics['rate'] / 1e6
    sorted_rates = np.sort(rates_mbps)
    p = 1. * np.arange(len(sorted_rates)) / (len(sorted_rates) - 1)
    plt.plot(sorted_rates, p, linewidth=2)
    plt.axvline(env.Rmin_u[env.s_u == 1][0]/1e6, color='r', linestyle='--', label='Rmin (Low)')
    plt.axvline(env.Rmin_u[env.s_u == 0][0]/1e6, color='orange', linestyle='--', label='Rmin (High)')
    plt.title("User Rate CDF")
    plt.xlabel("Rate (Mbps)")
    plt.ylabel("CDF")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 2. 带宽切片柱状图
    plt.subplot(1, 2, 2)
    plt.bar(['Slice 0', 'Slice 1', 'Slice 2'], rho_opt, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title("Optimized Global Slice Quotas")
    plt.ylabel("Fraction of Total Bandwidth")
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()