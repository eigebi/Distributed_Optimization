import numpy as np
import time
from dataclasses import dataclass
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ==========================================
# 1. 基础配置 (Configuration)
# ==========================================
@dataclass
class EnvCfg:
    fc_GHz: float = 3.5
    bandwidth_Hz: float = 100e6         # 100 MHz
    noise_figure_dB: float = 9.0
    pathloss_exponent: float = 3.6
    shadowing_sigma_dB: float = 8.0
    min_distance_m: float = 35.0
    Pmax_W: float = 80.0                # 80 W
    snr_drop_threshold_db: float = -6.0 
    seed: int = 42
    
    # === 优化目标参数 (Net Utility) ===
    # 目标: Max sum(w*log(R)) - alpha_b * sum(b) - alpha_p * sum(p)
    alpha_b: float = 5e-9              # 带宽单价 (针对 Hz)
    alpha_p: float = 0.01               # 功率单价 (针对 Watt)

def thermal_noise_psd(cfg):
    k_B = 1.380649e-23; T0 = 290.0
    return k_B * T0 * (10**(cfg.noise_figure_dB/10))

def calc_pathloss(d_m, cfg, rng):
    d0 = 1.0; fc = cfg.fc_GHz * 1e3
    PL0 = 32.45 + 20*np.log10(fc) + 20*np.log10(d0/1e3)
    pl = PL0 + 10*cfg.pathloss_exponent*np.log10(np.maximum(d_m, d0))
    if cfg.shadowing_sigma_dB > 0: pl += rng.normal(0, cfg.shadowing_sigma_dB, size=d_m.shape)
    return 10**(-pl/10.0)

# ==========================================
# 2. 拓扑生成 (不变)
# ==========================================
class StandardTopology:
    def __init__(self, cfg, isd=500.0):
        self.cfg = cfg; self.isd = isd
        self.rng = np.random.default_rng(cfg.seed)

    def generate_hex_bs(self, num_rings=2):
        locs = [[0.0, 0.0]]
        if num_rings >= 1:
            for i in range(6):
                a = np.radians(30 + 60*i)
                locs.append([self.isd*np.cos(a), self.isd*np.sin(a)])
        if num_rings >= 2:
            for i in range(6):
                a = np.radians(30 + 60*i)
                locs.append([2*self.isd*np.cos(a), 2*self.isd*np.sin(a)])
            for i in range(6):
                a = np.radians(60 + 60*i)
                locs.append([np.sqrt(3)*self.isd*np.cos(a), np.sqrt(3)*self.isd*np.sin(a)])
        return np.array(locs) + 3000.0

    def generate_ues_robust(self, bs_xy, K_per_bs=5, num_slices=3):
        B = len(bs_xy); ue_list = []; b_u_list = []
        noise_psd = thermal_noise_psd(self.cfg)
        full_noise = noise_psd * self.cfg.bandwidth_Hz
        
        for b in range(B):
            cnt = 0
            while cnt < K_per_bs:
                r = self.isd * 0.6 * np.sqrt(self.rng.random())
                th = self.rng.random() * 2 * np.pi
                cand = bs_xy[b] + [r*np.cos(th), r*np.sin(th)]
                
                dists = np.linalg.norm(bs_xy - cand, axis=1)
                if np.min(dists) < self.cfg.min_distance_m: continue
                
                h = calc_pathloss(np.array([dists[b]]), self.cfg, self.rng)[0]
                if 10*np.log10(self.cfg.Pmax_W * h / full_noise) < self.cfg.snr_drop_threshold_db: continue
                
                ue_list.append(cand); b_u_list.append(b); cnt += 1
        
        K = len(ue_list)
        # 切片分布: eMBB 40%, URLLC 30%, mMTC 30%
        s_u = self.rng.choice(num_slices, size=K, p=[0.15, 0.35, 0.5])
        G = np.zeros((K, B))
        ue_arr = np.array(ue_list)
        for k in range(K):
            d = np.linalg.norm(bs_xy - ue_arr[k], axis=1)
            G[k,:] = calc_pathloss(d, self.cfg, self.rng)
            
        return bs_xy, ue_arr, np.array(b_u_list), s_u, G

# ==========================================
# 3. 无线环境核心 (含 Net Utility Gradient)
# ==========================================
class WirelessEnvNumpy:
    def __init__(self, B, K, S, topo_data, cfg: EnvCfg):
        self.cfg = cfg
        self.B, self.K, self.S = B, K, S
        self.bs_xy, self.ue_xy, self.b_u, self.s_u, self.G = topo_data
        self.N0 = thermal_noise_psd(cfg)
        self.Pmax = np.full(B, cfg.Pmax_W)
        self.w_u = np.ones(K)
        
        # QoS: eMBB=5Mbps, URLLC=1Mbps
        self.Rmin_u = np.full(K, 0.1e6)
        self.Rmin_u[self.s_u == 0] = 0.5e6
        self.Rmin_u[self.s_u == 1] = 0.1e6
        
        # Weight: eMBB 高权重
        self.w_u[self.s_u == 0] = 1.0
        
        self.eps = 1e-9
        self.util_norm_factor = 1e2
        self.cons_norm_factors = [1e7, 1e6, 1e8] 

    def compute_metrics(self, b_vec, p_vec):
        """纯前向计算"""
        P_total_sb = np.zeros((self.S, self.B))
        np.add.at(P_total_sb, (self.s_u, self.b_u), p_vec)
        
        signal = p_vec * self.G[np.arange(self.K), self.b_u]
        
        # Interference (Co-slice ICI)
        P_tx_rel = P_total_sb[self.s_u, :]
        rx_total = np.einsum('kb,kb->k', P_tx_rel, self.G)
        own_bs = P_total_sb[self.s_u, self.b_u] * self.G[np.arange(self.K), self.b_u]
        interf = rx_total - own_bs
        
        sinr = signal / (interf + self.N0 * b_vec + 1e-15)
        rate = b_vec * np.log2(1.0 + sinr)
        return rate, sinr, interf

    def get_net_utility_gradients(self, b_vec, p_vec, i = None):
        """
        计算 Net Utility 的梯度:
        Grad = Grad(Utility) - Cost_Factor
        """
        if i is not None:
            b_temp = np.zeros_like(b_vec)
            b_temp[self.b_u==i] = b_vec[self.b_u==i]
            b_vec = b_temp
        rate, sinr, interf = self.compute_metrics(b_vec, p_vec)
        
        # 1. 计算数值
        utility = np.sum(self.w_u * np.log(self.eps + rate))
        cost = self.cfg.alpha_b * np.sum(b_vec) + self.cfg.alpha_p * np.sum(p_vec)
        net_utility = utility - cost
        
        # 2. 梯度推导
        dU_dR = self.w_u / (self.eps + rate)
        ln2 = np.log(2.0)
        term_sinr = b_vec / (ln2 * (1.0 + sinr))
        denom = interf + self.N0 * b_vec + 1e-15
        
        # --- Grad b ---
        dSINR_db = -sinr / denom * self.N0
        grad_b_util = dU_dR * (np.log2(1.0+sinr) + term_sinr*dSINR_db)
        # 关键修改：减去资源成本梯度
        grad_b = grad_b_util - self.cfg.alpha_b
        if i is not None:
            grad_b_temp = np.zeros_like(b_vec)
            grad_b_temp[self.b_u==i] = grad_b[self.b_u==i]
            grad_b = grad_b_temp
        
        # --- Grad p ---
        grad_p_util = np.zeros(self.K)
        # Self
        dR_dp_self = term_sinr * (self.G[np.arange(self.K), self.b_u] / denom)
        grad_p_util += dU_dR * dR_dp_self
        # Cross (Interference)
        vic_sens = dU_dR * term_sinr * (-sinr / denom)
        Price = np.zeros((self.S, self.B))
        for s in range(self.S):
            mask = (self.s_u == s)
            if not np.any(mask): continue
            sens = vic_sens[mask]; Gs = self.G[mask,:]
            tot = sens @ Gs
            own = np.zeros(self.B)
            contrib = sens * Gs[np.arange(len(sens)), self.b_u[mask]]
            np.add.at(own, self.b_u[mask], contrib)
            Price[s,:] = tot - own
        grad_p_util += Price[self.s_u, self.b_u]
        
        # 关键修改：减去资源成本梯度
        if i is not None:
            grad_p = grad_p_util
            grad_p[self.b_u == i] -= self.cfg.alpha_p
        else:
            grad_p = grad_p_util - self.cfg.alpha_p
        
        return grad_b, grad_p, net_utility, rate
    

    def get_constraints_gradients(self, b_vec, p_vec, i = None):
        """
        计算 每个agent上 的梯度:
        Grad = [Rmin - UE1_rate, Rmin - UE2_rate, ..., BS_power - Pmax, slice1_used - slice1_cap, ...]
        """
        if i is not None:
            b_temp = np.zeros_like(b_vec)
            b_temp[self.b_u==i] = b_vec[self.b_u==i]
            b_vec = b_temp
        rate, sinr, interf = self.compute_metrics(b_vec, p_vec)
        
        # 1. 计算数值
        utility = np.sum(self.w_u * np.log(self.eps + rate))
        cost = self.cfg.alpha_b * np.sum(b_vec) + self.cfg.alpha_p * np.sum(p_vec)
        net_utility = utility - cost
        
        # 2. 梯度推导
        dU_dR = self.w_u / (self.eps + rate)
        ln2 = np.log(2.0)
        term_sinr = b_vec / (ln2 * (1.0 + sinr))
        denom = interf + self.N0 * b_vec + 1e-15
        
        # --- Grad b ---
        dSINR_db = -sinr / denom * self.N0
        grad_b_util = dU_dR * (np.log2(1.0+sinr) + term_sinr*dSINR_db)
        # 关键修改：减去资源成本梯度
        grad_b = grad_b_util - self.cfg.alpha_b
        if i is not None:
            grad_b_temp = np.zeros_like(b_vec)
            grad_b_temp[self.b_u==i] = grad_b[self.b_u==i]
            grad_b = grad_b_temp
        
        # --- Grad p ---
        grad_p_util = np.zeros(self.K)
        # Self
        dR_dp_self = term_sinr * (self.G[np.arange(self.K), self.b_u] / denom)
        grad_p_util += dU_dR * dR_dp_self
        # Cross (Interference)
        vic_sens = dU_dR * term_sinr * (-sinr / denom)
        Price = np.zeros((self.S, self.B))
        for s in range(self.S):
            mask = (self.s_u == s)
            if not np.any(mask): continue
            sens = vic_sens[mask]; Gs = self.G[mask,:]
            tot = sens @ Gs
            own = np.zeros(self.B)
            contrib = sens * Gs[np.arange(len(sens)), self.b_u[mask]]
            np.add.at(own, self.b_u[mask], contrib)
            Price[s,:] = tot - own
        grad_p_util += Price[self.s_u, self.b_u]
        
        # 关键修改：减去资源成本梯度
        if i is not None:
            grad_p = grad_p_util
            grad_p[self.b_u == i] -= self.cfg.alpha_p
        else:
            grad_p = grad_p_util - self.cfg.alpha_p
        
        return grad_b, grad_p, net_utility, rate
   
    def get_constraints_gradients(self, b_vec, p_vec, rho_vec, bs_idx):
        """
        计算基站 bs_idx 的本地约束关于【全网所有变量】的 Jacobian 矩阵。
        
        Args:
            b_vec: (K_total,) 全网所有用户带宽
            p_vec: (K_total,) 全网所有用户功率
            rho_vec: (S,) 本地切片配额 (注意：rho通常只对本地约束求导)
            bs_idx: 当前基站索引
            
        Returns:
            jac_b: (Num_Cons, K_total)
            jac_p: (Num_Cons, K_total)  <-- 这里将包含跨基站的干扰梯度
            jac_rho: (Num_Cons, S)
            viols: (Num_Cons,)
        """
        # 1. 全局信息准备
        # ---------------------------
        K_total = self.K
        # 获取属于当前基站的用户索引 (Local UEs)
        local_mask = (self.b_u == bs_idx)
        local_indices = np.where(local_mask)[0] # Global indices of local UEs
        K_local = len(local_indices)
        
        # 2. 前向计算 (全网状态)
        # ---------------------------
        rate, sinr, interf = self.compute_metrics(b_vec, p_vec)
        
        # 提取本地相关的状态
        local_rate = rate[local_mask]
        local_sinr = sinr[local_mask]
        local_b = b_vec[local_mask]
        local_denom = interf[local_mask] + self.N0 * local_b + 1e-15
        
        # 3. 初始化 Jacobian (行数=本地约束数, 列数=全网变量数)
        # ------------------------------------------------------
        # Rows: QoS(K_local) + Power(1) + Slice_Cap(S)
        num_cons = K_local + 1 + self.S
        
        jac_b = np.zeros((num_cons, K_total))
        jac_p = np.zeros((num_cons, K_total))
        jac_rho = np.zeros((num_cons, self.S))
        viols = np.zeros(num_cons)
        
        ln2 = np.log(2.0)
        
        # =========================================================
        # Constraint Group 1: QoS (Rmin - R_k <= 0)
        # 涉及跨基站干扰，jac_p 是稠密的
        # =========================================================
        # 预计算公共求导项 dR/dSINR
        # term_sinr[k] = b_k / (ln2 * (1 + SINR_k))
        term_sinr = local_b / (ln2 * (1.0 + local_sinr)) # (K_local,)
        
        for i, u_global_idx in enumerate(local_indices):
            # i: 约束矩阵的行索引 (0 ~ K_local-1)
            # u_global_idx: 该约束对应的用户在全网的 ID
            
            # Violation
            viols[i] = self.Rmin_u[u_global_idx] - local_rate[i]
            
            # --- 对 b 的梯度 (Row i, Col u_global_idx) ---
            # 带宽不耦合，只影响自己
            dSINR_db = -local_sinr[i] / local_denom[i] * self.N0
            dR_db = np.log2(1.0 + local_sinr[i]) + term_sinr[i] * dSINR_db
            
            # g(x) = Rmin - R -> grad = -dR/db
            jac_b[i, u_global_idx] = -dR_db
            
            # --- 对 p 的梯度 (Row i, All Cols) ---
            # 这是一个大循环，或者向量化计算
            # dR_i / dp_j 存在，如果 j 干扰了 i
            
            # 1. 对自己的功率 (Self, j = u_global_idx)
            g_serve = self.G[u_global_idx, bs_idx]
            dR_dp_self = term_sinr[i] * (g_serve / local_denom[i])
            jac_p[i, u_global_idx] = -dR_dp_self
            
            # 2. 对干扰源的功率 (Cross, j != u_global_idx)
            # 只有同切片的用户才产生干扰
            my_slice = self.s_u[u_global_idx]
            
            # 找出全网所有属于该切片的用户 (干扰源候选人)
            interferer_mask = (self.s_u == my_slice) & (self.b_u != bs_idx)
            interferer_indices = np.where(interferer_mask)[0]
            
            if len(interferer_indices) > 0:
                # dR_i / dp_j = dR/dSINR * dSINR/dI * dI/dp_j
                # dSINR/dI = - SINR / Denom
                # dI/dp_j = G_{j_bs -> i_user} = G[u_global_idx, b_u[j]]
                
                chain_factor = term_sinr[i] * (-local_sinr[i] / local_denom[i]) # Scalar for user i
                
                # 获取这些干扰源所属的基站 ID
                interferer_bs = self.b_u[interferer_indices]
                
                # G[u, b] 表示基站 b 到用户 u 的增益
                # 这里我们需要：干扰源基站 -> 受害用户 i
                g_interf = self.G[u_global_idx, interferer_bs] # Vector
                
                # 填入 Jacobian 行
                # g(x) = -R -> grad = - (chain_factor * G)
                jac_p[i, interferer_indices] = -(chain_factor * g_interf)

        # =========================================================
        # Constraint Group 2: Local Power (sum(p_local) - Pmax <= 0)
        # 纯本地约束
        # =========================================================
        row_idx = K_local
        viols[row_idx] = np.sum(p_vec[local_mask]) - self.Pmax[bs_idx]
        
        # Grad: 对本地用户的 p 为 1，其他为 0
        jac_p[row_idx, local_indices] = 1.0
        
        # =========================================================
        # Constraint Group 3: Local Slice Capacity
        # sum(b_local_s) - rho_s * W <= 0
        # =========================================================
        start_row = K_local + 1
        # per slice B 约束处理。这里先假设传进来的是真实值，而非归一化值
        for s in range(self.S):
            row = start_row + s
            
            # 本地属于该 slice 的用户
            slice_mask_local = (self.s_u[local_indices] == s)
            slice_indices_global = local_indices[slice_mask_local]
            
            # Violation
            sum_b = np.sum(b_vec[slice_indices_global])
            viols[row] = sum_b - rho_vec[s]
            
            # Grad b: 对该切片的本地用户为 1
            jac_b[row, slice_indices_global] = 1.0
            
            # Grad rho: -W
            jac_rho[row, s] = -1.0

        return jac_b, jac_p, jac_rho, viols
# ==========================================
# 4. 严格硬约束求解器 (Two-Phase Hard Constraint Solver)
# ==========================================
def solve_centralized_hard_constrained(env):
    print(f"\n=== Centralized Solver (Net Utility Max, Hard Constraints) ===")
    K, B, S = env.K, env.B, env.S
    
    # --- Scaling Factors ---
    SCALE_B = 1e6  # 1 MHz -> 1.0
    SCALE_P = 1.0  # 1 W -> 1.0
    
    def unpack(x):
        return x[:K]*SCALE_B, x[K:2*K]*SCALE_P, x[2*K:]

    # --- Constraints Definition ---
    cons = []
    
    # C1: Sum(rho) <= 1
    cons.append({'type': 'ineq', 'fun': lambda x: 1.0 - np.sum(unpack(x)[2]),
                 'jac': lambda x: np.concatenate([np.zeros(2*K), -np.ones(S)])})
    
    # C2: BS Power <= Pmax
    for b in range(B):
        idx = np.where(env.b_u == b)[0]
        if len(idx)==0: continue
        def p_con(x, idx=idx, lim=env.Pmax[b]):
            return lim - np.sum(unpack(x)[1][idx])
        def p_jac(x, idx=idx):
            g = np.zeros(2*K+S); g[K+idx] = -SCALE_P; return g
        cons.append({'type': 'ineq', 'fun': p_con, 'jac': p_jac})
        
    # C3: Local Slice Capacity (sum b <= rho * W)
    for b in range(B):
        for s in range(S):
            idx = np.where((env.b_u==b) & (env.s_u==s))[0]
            def s_con(x, idx=idx, s=s):
                b_vec, _, rho = unpack(x)
                used = np.sum(b_vec[idx]) if len(idx)>0 else 0.0
                return rho[s] * env.cfg.bandwidth_Hz - used
            def s_jac(x, idx=idx, s=s):
                g = np.zeros(2*K+S)
                if len(idx)>0: g[idx] = -SCALE_B
                g[2*K+s] = env.cfg.bandwidth_Hz; return g
            cons.append({'type': 'ineq', 'fun': s_con, 'jac': s_jac})
            
    # C4: QoS Rate >= Rmin (Hard Non-convex)
    def qos_con(x):
        b, p, _ = unpack(x)
        rate, _, _ = env.compute_metrics(b, p)
        return rate - env.Rmin_u
    cons.append({'type': 'ineq', 'fun': qos_con})

    # Bounds
    b_bnds = [(1e-2, env.cfg.bandwidth_Hz/SCALE_B)]*K
    p_bnds = [(1e-6, env.cfg.Pmax_W/SCALE_P)]*K
    r_bnds = [(0.01, 1.0)]*S
    bounds = b_bnds + p_bnds + r_bnds

    # --- PHASE 1: Feasibility Search ---
    print("Phase 1: Finding feasible initialization...")
    x0 = np.concatenate([
        np.ones(K) * (env.cfg.bandwidth_Hz/K/SCALE_B),
        np.ones(K) * (env.cfg.Pmax_W/10.0/SCALE_P),
        np.ones(S) / S
    ])
    
    # Objective: Minimize max Violation
    def feas_obj(x):
        b, p, _ = unpack(x)
        rate, _, _ = env.compute_metrics(b, p)
        viol = np.maximum(0, env.Rmin_u - rate)
        return np.sum(viol**2)

    res_p1 = minimize(feas_obj, x0, method='SLSQP', bounds=bounds, constraints=cons,
                      options={'maxiter': 50, 'ftol': 1e-3, 'disp': False})
    
    print(f"Phase 1 Residual: {res_p1.fun:.4e}")
    x_start = res_p1.x # Always continue from best found point
    
    # --- PHASE 2: Net Utility Maximization ---
    print("Phase 2: Maximizing Net Utility...")
    
    def objective(x):
        b, p, _ = unpack(x)
        # 这里调用修正后的 get_net_utility_gradients
        gb, gp, net_u, _ = env.get_net_utility_gradients(b, p)
        
        # Minimize Negative Net Utility
        f = -net_u 
        
        # Gradients (Scaled & Negated)
        g_b_norm = -gb * SCALE_B
        g_p_norm = -gp * SCALE_P
        g_rho_norm = np.zeros(S)
        
        return f, np.concatenate([g_b_norm, g_p_norm, g_rho_norm])

    start = time.time()
    res = minimize(objective, x_start, method='SLSQP', jac=True, 
                   bounds=bounds, constraints=cons,
                   options={'maxiter': 1000, 'ftol': 1e-4, 'disp': True})
    dur = time.time() - start
    
    b_opt, p_opt, rho_opt = unpack(res.x)
    _, _, net_u_final, rate_opt = env.get_net_utility_gradients(b_opt, p_opt)
    
    return {
        'res': res, 'time': dur, 
        'net_utility': net_u_final, 
        'rate': rate_opt, 
        'rho': rho_opt, 'b': b_opt, 'p': p_opt,
        'qos_ok': np.all(rate_opt >= env.Rmin_u - 1e-3)
    }

# ==========================================
# 5. Main Visualization
# ==========================================
if __name__ == "__main__":
    cfg = EnvCfg()
    topo = StandardTopology(cfg)
    bs_xy = topo.generate_hex_bs(num_rings=1)
    data = topo.generate_ues_robust(bs_xy, K_per_bs=5, num_slices=3)
    
    env = WirelessEnvNumpy(len(bs_xy), len(data[1]), 3, data, cfg)
    print(f"Topology: {env.B} BS, {env.K} UEs")
    
    # 求解
    res = solve_centralized_hard_constrained(env)
    
    print("\n" + "="*40)
    print(f"Success: {res['res'].success}")
    print(f"All QoS Satisfied: {res['qos_ok']}")
    print(f"Final Net Utility: {res['net_utility']:.4f}")
    print(f"Slice Ratios: {np.round(res['rho'], 3)}")
    print("="*40)
    
    # --- 绘图 ---
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Rate CDF
    plt.subplot(1, 3, 1)
    r_mbps = np.sort(res['rate'])/1e6
    plt.plot(r_mbps, np.linspace(0,1,len(r_mbps)), lw=2)
    plt.axvline(5.0, c='r', ls='--', label='eMBB (5M)')
    plt.axvline(1.0, c='g', ls='--', label='URLLC (1M)')
    plt.xlabel('Rate (Mbps)'); plt.ylabel('CDF'); plt.grid(alpha=0.3)
    plt.title(f"Rate CDF")
    plt.legend()
    
    # Plot 2: Slice Allocation
    plt.subplot(1, 3, 2)
    plt.bar(['eMBB', 'URLLC', 'mMTC'], res['rho'], color=['tab:blue','tab:orange','tab:green'])
    plt.title('Global Bandwidth Quotas')
    plt.ylabel('Ratio')
    
    # Plot 3: Per-BS Power Usage (新增要求)
    plt.subplot(1, 3, 3)
    p_bs = np.zeros(env.B)
    np.add.at(p_bs, env.b_u, res['p'])
    plt.bar(range(env.B), p_bs, color='purple', alpha=0.7)
    plt.axhline(cfg.Pmax_W, color='r', linestyle='--', label='Max Power')
    plt.title('Per-BS Total Power Consumption')
    plt.xlabel('Base Station Index')
    plt.ylabel('Power (W)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()