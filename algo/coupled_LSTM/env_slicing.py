import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List

# ==========================================
# 1. 基础配置 (Configuration)
# ==========================================
@dataclass
class EnvCfg:
    # 3GPP UMa 风格参数
    fc_GHz: float = 3.5
    bandwidth_Hz: float = 20e6          # 系统总带宽 W_total
    noise_figure_dB: float = 7.0
    pathloss_exponent: float = 3.5
    shadowing_sigma_dB: float = 8.0
    min_distance_m: float = 35.0        # 最小距离保护
    Pmax_W: float = 40.0                # 基站最大功率 (46 dBm)
    
    # 仿真设置
    snr_drop_threshold_db: float = -6.0 # 准入控制阈值
    seed: int = 2025

# 辅助物理计算函数
def thermal_noise_psd_w_per_hz(noise_figure_dB: float, T_kelvin: float = 290.0):
    k_B = 1.380649e-23
    N0 = k_B * T_kelvin
    NF = 10.0 ** (noise_figure_dB / 10.0)
    return N0 * NF

def calc_pathloss_linear(d_m: np.ndarray, cfg: EnvCfg, rng: np.random.Generator):
    """Log-distance pathloss + Log-normal shadowing -> Linear Gain"""
    d0 = 1.0
    # FSPL at 1m
    fc_MHz = cfg.fc_GHz * 1e3
    PL0 = 32.45 + 20.0 * np.log10(fc_MHz) + 20.0 * np.log10(d0/1e3)
    
    # PL(d) = PL0 + 10n log10(d) + X_sigma
    pl_db = PL0 + 10.0 * cfg.pathloss_exponent * np.log10(np.maximum(d_m, d0))
    
    # Shadowing
    if cfg.shadowing_sigma_dB > 0:
        shadow = rng.normal(0.0, cfg.shadowing_sigma_dB, size=d_m.shape)
        pl_db += shadow
        
    # Return Channel Gain (not loss) -> 10^(-PL/10)
    return 10.0 ** (-pl_db / 10.0)

# ==========================================
# 2. 拓扑生成器 (支持 19-BS 六边形布局)
# ==========================================
class StandardTopology:
    def __init__(self, cfg: EnvCfg, isd=500.0):
        self.cfg = cfg
        self.isd = isd
        self.rng = np.random.default_rng(cfg.seed)

    def generate_hex_bs(self, num_rings=2):
        """
        生成标准六边形蜂窝坐标。
        num_rings=1 -> 7 BS
        num_rings=2 -> 19 BS (中心 + 第一圈6 + 第二圈12)
        """
        # 中心基站
        locs = [[0.0, 0.0]]
        
        if num_rings >= 1:
            # 第一圈 (6个)
            for i in range(6):
                angle = np.radians(30 + 60 * i)
                x = self.isd * np.cos(angle)
                y = self.isd * np.sin(angle)
                locs.append([x, y])
        
        if num_rings >= 2:
            # 第二圈 (12个)
            # 顶点方向 (距离 2*ISD)
            for i in range(6):
                angle = np.radians(30 + 60 * i)
                x = 2 * self.isd * np.cos(angle)
                y = 2 * self.isd * np.sin(angle)
                locs.append([x, y])
            # 边中点方向 (距离 sqrt(3)*ISD)
            for i in range(6):
                angle = np.radians(60 + 60 * i)
                r = np.sqrt(3) * self.isd
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                locs.append([x, y])

        # 平移至正象限 (方便绘图，不影响距离)
        return np.array(locs) + 3000.0

    def generate_ues_robust(self, bs_xy, K_per_bs=5, num_slices=3):
        """
        在基站覆盖范围内生成用户，并执行准入控制 (SNR Threshold)。
        """
        B = len(bs_xy)
        ue_list = []
        b_u_list = []
        
        noise_psd = thermal_noise_psd_w_per_hz(self.cfg.noise_figure_dB)
        # 假设全带宽满功率下的 SNR 用于准入判断
        full_noise = noise_psd * self.cfg.bandwidth_Hz
        
        cell_radius = self.isd * 0.6 # 略微重叠以制造边缘用户

        for b in range(B):
            cnt = 0
            center = bs_xy[b]
            while cnt < K_per_bs:
                # 极坐标均匀分布
                r = cell_radius * np.sqrt(self.rng.random())
                theta = self.rng.random() * 2 * np.pi
                cand = center + np.array([r * np.cos(theta), r * np.sin(theta)])
                
                # 1. 最小距离检查
                dists = np.linalg.norm(bs_xy - cand, axis=1)
                if np.min(dists) < self.cfg.min_distance_m:
                    continue
                
                # 2. SNR 准入检查 (避免不可行解)
                h_serve = calc_pathloss_linear(np.array([dists[b]]), self.cfg, self.rng)[0]
                rx_p = self.cfg.Pmax_W * h_serve
                snr_db = 10 * np.log10(rx_p / full_noise)
                
                if snr_db < self.cfg.snr_drop_threshold_db:
                    continue # Drop
                
                ue_list.append(cand)
                b_u_list.append(b)
                cnt += 1
        
        K = len(ue_list)
        # 随机分配切片
        s_u = self.rng.choice(num_slices, size=K)
        
        # 计算全局信道矩阵 G [K, B]
        ue_arr = np.array(ue_list)
        G = np.zeros((K, B))
        for k in range(K):
            dists = np.linalg.norm(bs_xy - ue_arr[k], axis=1)
            G[k, :] = calc_pathloss_linear(dists, self.cfg, self.rng)
            
        return bs_xy, ue_arr, np.array(b_u_list), s_u, G

# ==========================================
# 3. 无线环境模型 (继承您的 FormulationV2 逻辑)
# ==========================================
class WirelessEnvNumpy:
    """
    基于 Numpy 手动推导梯度的环境。
    完全复用您 FormulationV2 中的数学逻辑。
    """
    def __init__(self, B, K, S, topo_data, cfg: EnvCfg, alpha_reg=1e-3):
        self.cfg = cfg
        self.B = B
        self.K = K
        self.S = S
        self.alpha_reg = alpha_reg # 正则化系数
        self.eps = 1e-9
        
        # 拓扑数据
        self.bs_xy = topo_data[0]
        self.ue_xy = topo_data[1]
        self.b_u = topo_data[2]   # (K,)
        self.s_u = topo_data[3]   # (K,)
        self.G = topo_data[4]     # (K, B)
        
        # 常量
        self.Wb = np.full(B, cfg.bandwidth_Hz)
        self.Pmax = np.full(B, cfg.Pmax_W)
        self.N0 = thermal_noise_psd_w_per_hz(cfg.noise_figure_dB)
        
        # 固定权重
        self.w_u = np.ones(K)
        self.Rmin_u = np.full(K, 0.5e6) # 0.5 Mbps QoS
        
        # 预先计算 theta, phi (假设均分，如您代码所示)
        self.theta = self._init_equal_shares(self.b_u, self.s_u)
        self.phi = self.theta.copy() # 简单起见假设相同

    def _init_equal_shares(self, b_u, s_u):
        val = np.zeros(self.K)
        for b in range(self.B):
            for s in range(self.S):
                idx = np.where((b_u==b) & (s_u==s))[0]
                if len(idx) > 0:
                    val[idx] = 1.0 / len(idx)
        return val

    # ---------- 核心计算 (前向) ----------
    def compute_rates(self, rho, eta):
        """
        rho: (S,)
        eta: (S, B)
        返回: Ru, sinr, (A_u, I_u, D_u)
        """
        # 1. 噪声功率 sigma^2 = N0 * rho_s * W_b -> (S, B)
        sigma2 = self.N0 * rho[:, None] * self.Wb[None, :]
        
        # 2. 信号部分
        # A_u = theta * Pmax[b_u] * G[u, b_u]
        A_u = self.theta * self.Pmax[self.b_u] * self.G[np.arange(self.K), self.b_u]
        # Numerator = eta[s_u, b_u] * A_u
        num = eta[self.s_u, self.b_u] * A_u
        
        # 3. 干扰部分 (Slice-Specific Interference)
        # I_total_at_u = sum_{b'} eta[s_u, b'] * Pmax[b'] * G[u, b']
        # 这里的干扰源是同切片的所有基站
        I_u = np.zeros(self.K)
        
        # 向量化计算干扰: P_tx_per_slice_bs = eta * Pmax
        P_tx_sb = eta * self.Pmax[None, :] # (S, B)
        
        # 对每个用户计算接收到的总能量 (同切片)
        # Rx_power[u] = sum_b ( P_tx[s_u, b] * G[u, b] )
        # 这一步可以用矩阵乘法优化，但为了清晰用循环（Numpy自动广播）
        # P_tx_users = P_tx_sb[self.s_u, :] # (K, B)
        # Rx_power = np.sum(P_tx_users * self.G, axis=1) # (K,)
        
        # 或者更快的写法：
        P_tx_users = P_tx_sb[self.s_u] # (K, B)
        I_u = np.einsum('kb,kb->k', P_tx_users, self.G) # 总接收功率
        
        # 减去有用信号得到干扰
        signal_power = eta[self.s_u, self.b_u] * self.Pmax[self.b_u] * self.G[np.arange(self.K), self.b_u]
        I_u -= signal_power
        
        # 4. SINR & Rate
        D_u = I_u + sigma2[self.s_u, self.b_u]
        sinr = num / np.maximum(D_u, 1e-15)
        
        # Rate = phi * rho * W * log2(1+SINR)
        Ru = self.phi * rho[self.s_u] * self.Wb[self.b_u] * np.log2(1.0 + sinr)
        
        return Ru, sinr, (A_u, I_u, D_u)

    # ---------- 梯度计算 (反向 - 手动推导) ----------
    def analytical_gradients(self, rho, eta):
        """
        手动计算 Utility 对 rho 和 eta 的梯度。
        Utility = sum(w_u * log(Ru)).
        这完全复刻了您 FormulationV2 中的逻辑。
        """
        Ru, sinr, (A_u, I_u, D_u) = self.compute_rates(rho, eta)
        
        # dObj/dRu = w / Ru
        dObj_dRu = self.w_u / (self.eps + Ru)
        
        # 公共系数
        ln2 = np.log(2.0)
        # dRu/dSINR term
        term1 = dObj_dRu * (self.phi * rho[self.s_u] * self.Wb[self.b_u]) / (ln2 * (1.0 + sinr))
        
        grad_rho = np.zeros(self.S)
        grad_eta = np.zeros((self.S, self.B))
        
        # --- 1. Gradient wrt Eta (Power) ---
        # 分两部分：增加自己(Signal) + 干扰别人(Interference)
        
        # Part A: 作为 Signal (对本小区用户的贡献)
        # dSINR / d_eta_self = A_u / D_u
        grad_eta_signal = term1 * (A_u / D_u)
        # 累加到对应的 (s, b)
        np.add.at(grad_eta, (self.s_u, self.b_u), grad_eta_signal)
        
        # Part B: 作为 Interference (对邻区同切片用户的负贡献)
        # dSINR_j / d_eta_i = - (SINR_j / D_j) * dI_j/d_eta_i
        # dI_j / d_eta_i = Pmax[i] * G[j, i]
        # 这里的 j 是受害用户 index, i 是干扰基站 index (对应 eta 的列)
        
        # 这是一个大的散射操作。为了效率：
        # 干扰系数 coeff_j = term1[j] * (-SINR[j] / D[j]) * Pmax[?]
        # 实际上：dObj/d_eta[s, b] += sum_{k in slice s, b_k != b} ( coeff[k] * Pmax[b] * G[k, b] )
        
        interf_factor = -term1 * (eta[self.s_u, self.b_u] * A_u) / (D_u**2) # (K,)
        
        # 对每个基站 b，计算它对所有同切片用户造成的干扰梯度
        for b in range(self.B):
            dI_db = self.Pmax[b] * self.G[:, b] # (K,) 所有用户受到来自 b 的能量增益
            
            # 只有同切片用户受影响，且排除自己服务的用户 (已经在 Part A 算过)
            # mask: s_u == s AND b_u != b
            # 但我们可以更简单：对每个 slice s，计算所有属于 s 的用户的贡献
            
            for s in range(self.S):
                # 找出属于 slice s 的用户
                user_mask = (self.s_u == s) & (self.b_u != b)
                if np.any(user_mask):
                    grad_val = np.sum(interf_factor[user_mask] * dI_db[user_mask])
                    grad_eta[s, b] += grad_val

        # --- 2. Gradient wrt Rho (Bandwidth) ---
        # Part A: Direct effect on Rate (Linear)
        # dRu/drho = Ru / rho
        grad_rho_direct = dObj_dRu * (Ru / rho[self.s_u])
        np.add.at(grad_rho, self.s_u, grad_rho_direct)
        
        # Part B: Indirect effect via Noise (sigma2) in Denominator
        # dSINR/drho = - (Num / D^2) * (dD/drho)
        # dD/drho = N0 * Wb
        grad_rho_noise = term1 * (- (eta[self.s_u, self.b_u] * A_u) / (D_u**2)) * (self.N0 * self.Wb[self.b_u])
        np.add.at(grad_rho, self.s_u, grad_rho_noise)
        
        # Regularization (可选)
        grad_eta -= self.alpha_reg
        grad_rho -= self.alpha_reg
        
        return grad_rho, grad_eta, Ru

    # ==========================================
    # 4. 分布式接口 (User Requested)
    # ==========================================
    def get_distributed_gradients(self, rho_global, eta_matrix):
        """
        根据当前的全局 rho 和所有人的 eta，计算每个 Agent 需要的梯度。
        
        Args:
            rho_global: (S,) 全局共识后的带宽比例
            eta_matrix: (S, B) 所有 Agent 的功率决策
            
        Returns:
            local_grads: List of (S,) arrays. Agent b 关于自己变量 eta[:, b] 的梯度。
            coupled_grad: (S,) array. 关于全局变量 rho 的梯度。
            metrics: 包含 rates, violations 等
        """
        # 1. 调用手动推导的梯度函数
        g_rho, g_eta, Ru = self.analytical_gradients(rho_global, eta_matrix)
        
        # 2. 拆分给各个 Agent
        local_grads = []
        for b in range(self.B):
            # 提取第 b 列，这就是 Agent b 对 Utility 的贡献 (含利他项)
            # 这就是 dL / d(eta_b)
            g_b = g_eta[:, b].copy()
            local_grads.append(g_b)
            
        # 3. 耦合梯度 (用于 Consensus / Manager)
        # 这就是 dL / d(rho)
        coupled_grad = g_rho.copy()
        
        # 4. 计算 Metrics 用于检查约束
        # QoS Violation
        qos_viol = np.maximum(0, self.Rmin_u - Ru)
        # Global Constraint Violation (sum rho <= 1)
        rho_viol = max(0, np.sum(rho_global) - 1.0)
        # Local Constraint Violation (sum eta <= 1)
        eta_viol = np.maximum(0, np.sum(eta_matrix, axis=0) - 1.0)
        
        metrics = {
            'qos_viol': qos_viol, # (K,)
            'rho_viol': rho_viol, # Scalar
            'eta_viol': eta_viol, # (B,)
            'avg_rate': np.mean(Ru)
        }
        
        return local_grads, coupled_grad, metrics

# ==========================================
# 5. 使用示例
# ==========================================
if __name__ == "__main__":
    # 1. 生成数据 (19 BS)
    cfg = EnvCfg()
    topo_gen = StandardTopology(cfg)
    bs_xy = topo_gen.generate_hex_bs(num_rings=2) # 19 BS
    # 生成用户数据
    topo_data = topo_gen.generate_ues_robust(bs_xy, K_per_bs=5, num_slices=3)
    
    print(f"Topology: {len(bs_xy)} BS, {len(topo_data[1])} UEs")
    
    # 2. 初始化环境
    env = WirelessEnvNumpy(
        B=len(bs_xy), 
        K=len(topo_data[1]), 
        S=3, 
        topo_data=topo_data, 
        cfg=cfg
    )
    
    # 3. 模拟一次 DMC 迭代中的梯度获取
    # 假设当前状态
    current_rho = np.ones(3) / 3.0
    current_eta = np.ones((3, 19)) * 0.2
    
    # 获取梯度
    l_grads, c_grad, metrics = env.get_distributed_gradients(current_rho, current_eta)
    
    print("\n--- Gradient Output Check ---")
    print(f"Coupled Gradient (d_rho): {c_grad}")
    print(f"Agent 5 Local Gradient (d_eta_5): {l_grads[5]}")
    print(f"Avg Rate: {metrics['avg_rate'] / 1e6:.2f} Mbps")
