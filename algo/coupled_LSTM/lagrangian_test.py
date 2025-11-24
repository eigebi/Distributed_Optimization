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
# 2. 拓扑生成
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
# 3. 无线环境核心
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
        self.Rmin_u[self.s_u == 0] = 5.0e6
        self.Rmin_u[self.s_u == 1] = 1.0e6
        
        # Weight: eMBB 高权重
        self.w_u[self.s_u == 0] = 2.0
        
        self.eps = 1e-9

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
            # 仅处理特定 BS 的逻辑省略，此处 Lagrange Solver 使用全局模式 (i=None)
            pass
            
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
        # 减去资源成本梯度
        grad_b = grad_b_util - self.cfg.alpha_b
        
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
        
        # 减去资源成本梯度
        grad_p = grad_p_util - self.cfg.alpha_p
        
        return grad_b, grad_p, net_utility, rate

# ==========================================
# 4. Lagrangian Solver (Primal-Dual)
# ==========================================
class LagrangianSolver:
    def __init__(self, env, lr_primal=1e-3, lr_dual=1e-2, max_iter=2000):
        self.env = env
        # Primal learning rates
        self.lr_b = lr_primal * 2e6  # Bandwidth 需要更大的缩放，因为数值大
        self.lr_p = lr_primal
        self.lr_rho = lr_primal * 0.1
        
        # Dual learning rate
        self.lr_dual = lr_dual
        self.max_iter = max_iter
        
        # Dual Variables (Lagrange Multipliers)初始化
        # 1. QoS Duals: lambda_u (associated with R_u >= Rmin)
        self.lam = np.zeros(env.K)
        # 2. Power Duals: mu_b (associated with sum(p) <= Pmax)
        self.mu = np.zeros(env.B)
        # 3. Slice Capacity Duals: nu_bs (associated with sum(b) <= rho * W)
        self.nu = np.zeros((env.B, env.S))

    def solve(self):
        print(f"\n=== Lagrangian Primal-Dual Solver ===")
        
        # 1. Initialization
        K, B, S = self.env.K, self.env.B, self.env.S
        
        # Primal Variables Initialization
        b_vec = np.full(K, self.env.cfg.bandwidth_Hz / K)
        p_vec = np.full(K, self.env.cfg.Pmax_W / 10.0)
        rho_vec = np.ones(S) / S
        
        # Tracking history
        hist_util = []
        hist_violation = []
        
        best_obj = -np.inf
        best_vars = (b_vec.copy(), p_vec.copy(), rho_vec.copy())

        # Backup original weights
        w_orig = self.env.w_u.copy()

        try:
            for it in range(self.max_iter):
                # ==========================
                # Step 1: Forward Pass
                # ==========================
                rate, _, _ = self.env.compute_metrics(b_vec, p_vec)
                
                # ==========================
                # Step 2: Dual Update (Gradient Descent on Duals)
                # lambda <- [lambda + lr * (Violation)]+
                # ==========================
                
                # 2.1 QoS Violation: Rmin - R <= 0
                viol_qos = self.env.Rmin_u - rate
                self.lam = np.maximum(0, self.lam + self.lr_dual * viol_qos)
                
                # 2.2 Power Violation: sum(p) - Pmax <= 0
                p_sum_bs = np.zeros(B)
                np.add.at(p_sum_bs, self.env.b_u, p_vec)
                viol_p = p_sum_bs - self.env.Pmax
                self.mu = np.maximum(0, self.mu + self.lr_dual * viol_p)
                
                # 2.3 Slice Violation: sum(b) - rho*W <= 0
                b_sum_bs_s = np.zeros((B, S))
                np.add.at(b_sum_bs_s, (self.env.b_u, self.env.s_u), b_vec)
                viol_s = b_sum_bs_s - rho_vec[None, :] * self.env.cfg.bandwidth_Hz
                # 注意：这里带宽数值很大，对偶更新时适当缩放以保持稳定
                self.nu = np.maximum(0, self.nu + self.lr_dual * 1e-6 * viol_s)

                # ==========================
                # Step 3: Primal Update (Gradient Ascent on Lagrangian)
                # ==========================
                
                # --- 3.1 Effective Weights Trick ---
                # 拉格朗日函数中关于速率的部分是: sum(w_u * log(R_u)) + sum(lambda_u * R_u)
                # 对 R 求导: w_u/R_u + lambda_u
                # 这等价于: (w_u + lambda_u * R_u) / R_u
                # 所以我们将权重暂时替换为 w_eff = w + lambda * R，即可复用原有的梯度函数
                w_eff = w_orig + self.lam * (rate + 1e-9)
                self.env.w_u = w_eff 
                
                # 获取梯度 (此时已包含 alpha_b 和 alpha_p 的成本)
                grad_b, grad_p, _, _ = self.env.get_net_utility_gradients(b_vec, p_vec)
                
                # 恢复原始权重
                self.env.w_u = w_orig
                
                # --- 3.2 Add Constraint Costs to Gradients ---
                
                # Adjust Grad B: subtract nu (Slice cost dual)
                # Constraint term is -nu * (sum b - ...). dL/db includes -nu
                grad_b_lag = grad_b - self.nu[self.env.b_u, self.env.s_u]
                
                # Adjust Grad P: subtract mu (Power cost dual)
                grad_p_lag = grad_p - self.mu[self.env.b_u]
                
                # Adjust Grad Rho: 
                # Constraint term in L: + sum_{b,s} nu_{b,s} * rho_s * W
                # dL/drho = sum_b (nu_{b,s} * W)
                grad_rho_lag = np.sum(self.nu, axis=0) * self.env.cfg.bandwidth_Hz

                # --- 3.3 Apply Gradients with Decay ---
                decay = 1.0 / (1.0 + 0.0005 * it)
                
                b_vec += self.lr_b * decay * grad_b_lag
                p_vec += self.lr_p * decay * grad_p_lag
                rho_vec += self.lr_rho * decay * grad_rho_lag
                
                # --- 3.4 Projection ---
                b_vec = np.maximum(100.0, b_vec) # Min 100Hz
                p_vec = np.clip(p_vec, 1e-6, self.env.Pmax[self.env.b_u]) # 简单裁剪
                
                # Rho Simplex Projection
                rho_vec = np.maximum(0.01, rho_vec)
                rho_vec = rho_vec / np.sum(rho_vec)
                
                # ==========================
                # Step 4: Logging & Best Model
                # ==========================
                # 计算违反程度 (L2 norm)
                total_viol = np.sum(np.maximum(0, viol_qos)**2) + \
                             np.sum(np.maximum(0, viol_p)**2) + \
                             np.sum(np.maximum(0, viol_s)**2)
                
                # 真实的原始目标函数值 (Net Utility)
                obj = np.sum(w_orig * np.log(1e-9 + rate)) - \
                      self.env.cfg.alpha_b * np.sum(b_vec) - \
                      self.env.cfg.alpha_p * np.sum(p_vec)
                
                hist_util.append(obj)
                hist_violation.append(total_viol)
                
                # 保存最佳的可行解 (容忍度 1e-3)
                if total_viol < 1e-2 and obj > best_obj:
                    best_obj = obj
                    best_vars = (b_vec.copy(), p_vec.copy(), rho_vec.copy())
                
                if it % 200 == 0:
                    print(f"Iter {it:4d} | NetUtil: {obj:8.2f} | Viol: {total_viol:.4e} | "
                          f"Avg Lambda: {np.mean(self.lam):.2f}")

        except KeyboardInterrupt:
            print("Stopped by user.")

        # ==========================
        # Return Results
        # ==========================
        if best_obj == -np.inf:
            print("Warning: No strictly feasible solution found, returning last step.")
            b_fin, p_fin, rho_fin = b_vec, p_vec, rho_vec
        else:
            print(f"Restoring best feasible solution with Util: {best_obj:.4f}")
            b_fin, p_fin, rho_fin = best_vars
            
        rate_fin, _, _ = self.env.compute_metrics(b_fin, p_fin)
        net_u_fin = np.sum(w_orig * np.log(1e-9 + rate_fin)) - \
                    self.env.cfg.alpha_b * np.sum(b_fin) - \
                    self.env.cfg.alpha_p * np.sum(p_fin)

        return {
            'net_utility': net_u_fin,
            'rate': rate_fin,
            'b': b_fin, 'p': p_fin, 'rho': rho_fin,
            'hist_util': hist_util,
            'hist_viol': hist_violation,
            'qos_ok': np.all(rate_fin >= self.env.Rmin_u - 1e-3)
        }

# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == "__main__":
    # 1. Setup Environment
    cfg = EnvCfg()
    topo = StandardTopology(cfg)
    bs_xy = topo.generate_hex_bs(num_rings=0)
    # K_per_bs=5, 3 slices
    data = topo.generate_ues_robust(bs_xy, K_per_bs=3, num_slices=3)
    env = WirelessEnvNumpy(len(bs_xy), len(data[1]), 3, data, cfg)
    
    print(f"\nTopology: {env.B} BS, {env.K} UEs")
    print(f"Rmin requirements: eMBB=5Mbps, URLLC=1Mbps")
    
    # 2. Run Lagrangian Solver
    # lr_primal 控制 b,p,rho 更新步长; lr_dual 控制对偶变量更新步长 (惩罚力度)
    solver = LagrangianSolver(env, lr_primal=2e-4, lr_dual=5e-2, max_iter=2500)
    res = solver.solve()
    
    # 3. Print Results
    print("\n" + "="*40)
    print(f"[Lagrangian] All QoS Satisfied: {res['qos_ok']}")
    print(f"[Lagrangian] Final Net Utility: {res['net_utility']:.4f}")
    print(f"[Lagrangian] Slice Ratios: {np.round(res['rho'], 3)}")
    print("="*40)

    # 4. Visualization
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Convergence History (Utility)
    plt.subplot(2, 2, 1)
    plt.plot(res['hist_util'], label='Net Utility')
    plt.title('Optimization Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('Net Utility')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Convergence History (Violation)
    plt.subplot(2, 2, 2)
    plt.plot(res['hist_viol'], color='orange', label='Total Violation')
    plt.yscale('log')
    plt.title('Constraint Violation (Log Scale)')
    plt.xlabel('Iteration')
    plt.ylabel('L2 Norm of Violations')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Rate CDF
    plt.subplot(2, 2, 3)
    r_mbps = np.sort(res['rate'])/1e6
    plt.plot(r_mbps, np.linspace(0,1,len(r_mbps)), lw=2, label='User Rates')
    plt.axvline(5.0, c='r', ls='--', label='eMBB Target (5M)')
    plt.axvline(1.0, c='g', ls='--', label='URLLC Target (1M)')
    plt.xlabel('Rate (Mbps)'); plt.ylabel('CDF')
    plt.legend()
    plt.title('User Rate Distribution')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Slice Allocation & BS Power
    plt.subplot(2, 2, 4)
    # Double axis chart
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Bar 1: Slice Config
    ax1.bar(['eMBB', 'URLLC', 'mMTC'], res['rho'], color='tab:blue', alpha=0.6, label='Slice Ratio')
    ax1.set_ylabel('Global Slice Ratio', color='tab:blue')
    
    # Bar 2: BS Power (Overlay as scatter or line to save space, or use average)
    p_bs = np.zeros(env.B)
    np.add.at(p_bs, env.b_u, res['p'])
    avg_p = np.mean(p_bs)
    ax2.axhline(avg_p, color='tab:red', linestyle='-', linewidth=2, label=f'Avg BS Power ({avg_p:.1f}W)')
    ax2.axhline(cfg.Pmax_W, color='red', linestyle='--', label='Max Power')
    ax2.set_ylabel('Power (W)', color='tab:red')
    
    plt.title('Resource Allocation Summary')
    
    plt.tight_layout()
    plt.show()