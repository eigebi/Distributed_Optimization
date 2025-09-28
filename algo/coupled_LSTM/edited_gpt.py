import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List

@dataclass
class EnvCfg:
    area_size_m: Tuple[float, float] = (2000.0, 2000.0)
    fc_GHz: float = 3.5
    bandwidth_Hz: float = 20e6
    noise_figure_dB: float = 7.0
    pathloss_exponent: float = 3.5
    shadowing_sigma_dB: float = 6.0
    min_distance_m: float = 10.0
    seed: int = 2025

def thermal_noise_power(bandwidth_Hz: float, noise_figure_dB: float, T_kelvin: float = 290.0):
    k = 1.380649e-23
    F = 10 ** (noise_figure_dB / 10.0)
    noise_W = k * T_kelvin * bandwidth_Hz * F
    return noise_W

def generate_ppp_positions(N: int, K: int, area_size_m=(2000.0, 2000.0), seed: int = 2025):
    rng = np.random.default_rng(seed)
    W, H = area_size_m
    bs_xy = rng.uniform([0, 0], [W, H], size=(N, 2))
    ue_xy = rng.uniform([0, 0], [W, H], size=(K, 2))
    return bs_xy, ue_xy

def pathloss_dB_log_distance(d_m, fc_GHz=3.5, n=3.5, shadowing_sigma_dB=6.0, rng=None):
    d_m = np.maximum(d_m, 1.0)
    PL0 = 32.4 + 20.0 * np.log10(fc_GHz)
    PL_dB = PL0 + 10.0 * n * np.log10(d_m)
    if shadowing_sigma_dB>0:
        if rng is None: rng = np.random.default_rng()
        PL_dB = PL_dB + rng.normal(0.0, shadowing_sigma_dB, size=np.shape(d_m))
    return PL_dB

def gain_matrix_from_positions(bs_xy, ue_xy, cfg: EnvCfg):
    rng = np.random.default_rng(cfg.seed)
    dx = ue_xy[:, [0]] - bs_xy[None, :, 0]
    dy = ue_xy[:, [1]] - bs_xy[None, :, 1]
    d = np.sqrt(dx * dx + dy * dy)
    d = np.maximum(d, cfg.min_distance_m)
    PL_dB = pathloss_dB_log_distance(d, cfg.fc_GHz, cfg.pathloss_exponent, cfg.shadowing_sigma_dB, rng)
    G = 10 ** (-PL_dB / 10.0)
    return G  # shape (K,B)

class PerUEFormulation:
    """
    Per-UE utility & QoS. Interference is power-aware:
      I_u = sum_{b' != b(u)} Pmax[b'] * (sum_s eta[s,b']) * g[u,b'] + sigma^2
    Signal for UE u (served by BS b, slice s):
      P_sig_u = (eta[s,b] * Pmax[b] / n_sb) * g[u,b]
      BW_u    = W * (rho[s,b] / n_sb)
      R_u     = BW_u * log2(1 + P_sig_u / I_u)
    """
    def __init__(self, B:int, K:int, slice_probs=(0.5,0.3,0.2), seed:int=2025, cfg:EnvCfg=EnvCfg()):
        self.cfg = cfg
        self.B, self.K, self.S = B, K, 3
        self.rng = np.random.default_rng(seed)
        # Geometry & channels
        self.bs_xy, self.ue_xy = generate_ppp_positions(B, K, cfg.area_size_m, seed=seed)
        self.G = gain_matrix_from_positions(self.bs_xy, self.ue_xy, cfg)  # (K,B)
        # Associations
        self.ue2bs = np.argmax(self.G, axis=1)  # strongest-BS association
        probs = np.array(slice_probs, float); probs/=probs.sum()
        self.ue2slice = self.rng.choice(self.S, size=K, p=probs)
        # Masks & groups
        self.a_sb = np.zeros((self.S, self.B), dtype=np.float32)
        self.U_sb: Dict[Tuple[int,int], np.ndarray] = {}
        for s in range(self.S):
            for b in range(self.B):
                mask = np.where((self.ue2slice==s) & (self.ue2bs==b))[0]
                self.U_sb[(s,b)] = mask
                if mask.size>0: self.a_sb[s,b]=1.0
        # 好像是为了防止某个切片在某个基站下没有用户，从而导致后续计算中除以零（并不一定每个基站都分到了用户，虽然概率很小）
        if np.all(self.a_sb.sum(axis=0)==0):
            self.a_sb[:,:]=1.0
        # System params
        self.W = cfg.bandwidth_Hz
        self.sigma2 = thermal_noise_power(cfg.bandwidth_Hz, cfg.noise_figure_dB)
        self.Pmax = np.full(self.B, 40.0, dtype=np.float32)  # W
        # UE-level weights and QoS

        self.w_u = np.ones(self.K, dtype=np.float32) # to determine

        # per slice min rate requirements
        Rmin_map = {0: 2e6, 1: 1e6, 2: 1e5}  # eMBB, URLLC, mMTC (example bps) 
        self.Rmin_u = np.array([Rmin_map[int(self.ue2slice[u])] for u in range(self.K)], dtype=np.float32)
        self.eps = 1e-9

    def split_x(self, x: np.ndarray):
        S, B = self.S, self.B
        assert x.shape[0] == S*2*B
        rho = x[:S*B].reshape(S,B).copy()
        eta = x[S*B:].reshape(S,B).copy()
        return rho, eta

    def merge(self, rho, eta):
        return np.concatenate([rho.flatten(), eta.flatten()])

    def per_ue_quantities(self, rho: np.ndarray, eta: np.ndarray):
        K,B,S = self.K,self.B,self.S
        n_sb = np.zeros((S,B), dtype=np.int32)
        for s in range(S):
            for b in range(B):
                n_sb[s,b] = max(1, self.U_sb[(s,b)].size)
        ell_b = np.sum(self.a_sb * eta, axis=0)
        s_u = self.ue2slice
        b_u = self.ue2bs
        n_u = np.array([n_sb[s_u[u], b_u[u]] for u in range(K)], dtype=np.float32)
        rho_u = np.array([rho[s_u[u], b_u[u]] for u in range(K)], dtype=np.float32)
        eta_u = np.array([eta[s_u[u], b_u[u]] for u in range(K)], dtype=np.float32)
        P_sig = (eta_u * self.Pmax[b_u] / n_u) * self.G[np.arange(K), b_u]
        mask_mat = np.ones((K,B), dtype=bool)
        mask_mat[np.arange(K), b_u] = False
        I_u = (self.Pmax[None,:] * ell_b[None,:] * self.G * mask_mat).sum(axis=1) + self.sigma2
        BW_u = self.W * (rho_u / n_u)
        SNR_u = P_sig / I_u
        R_u = BW_u * (np.log1p(SNR_u) / np.log(2.0))
        return dict(n_sb=n_sb, s_u=s_u, b_u=b_u, n_u=n_u, rho_u=rho_u, eta_u=eta_u,
                    P_sig=P_sig, I_u=I_u, BW_u=BW_u, SNR_u=SNR_u, R_u=R_u)

    def objective_and_grads(self, x: np.ndarray):
        rho, eta = self.split_x(x)
        q = self.per_ue_quantities(rho, eta)
        R_u = q['R_u']
        w = self.w_u; eps = self.eps
        f = float(np.sum(w * np.log(eps + R_u)))

        K, B, S = self.K, self.B, self.S
        dfdR = w / (eps + R_u)

        dR_drho = np.zeros((S,B), dtype=np.float64)
        for s in range(S):
            for b in range(B):
                U = self.U_sb[(s,b)]
                if U.size == 0: continue
                n = max(1, U.size)
                term = (self.W / n) * (np.log1p(q['SNR_u'][U]) / np.log(2.0))
                dR_drho[s,b] = np.sum(dfdR[U] * term)

        dR_deta = np.zeros((S,B), dtype=np.float64)
        u_idx = np.arange(K)
        s_u, b_u = q['s_u'], q['b_u']
        n_u = q['n_u']
        I_u = q['I_u']
        A_u = (self.Pmax[b_u] * self.G[u_idx, b_u]) / n_u
        denom_u = I_u + q['eta_u'] * A_u
        coef_self = (self.W * q['rho_u'] / n_u) * (1.0/np.log(2.0)) * (A_u / denom_u)
        for s in range(S):
            for b in range(B):
                U = self.U_sb[(s,b)]
                if U.size == 0: continue
                dR_deta[s,b] += np.sum(dfdR[U] * coef_self[U])

        factor_u = (self.W * q['rho_u'] / n_u) * (1.0/np.log(2.0)) * ( - (q['eta_u']*A_u) / (I_u * denom_u) )
        for b in range(B):
            U_notb = np.where(b_u != b)[0]
            if U_notb.size == 0: continue
            dI_du = self.Pmax[b] * self.G[U_notb, b]
            contrib = np.sum(dfdR[U_notb] * factor_u[U_notb] * dI_du)
            for s in range(S):
                dR_deta[s,b] += contrib

        grad = np.concatenate([dR_drho.flatten(), dR_deta.flatten()]).astype(np.float64)
        return f, grad

    def per_ue_constraints_and_jacobians(self, x: np.ndarray):
        rho, eta = self.split_x(x)
        q = self.per_ue_quantities(rho, eta)
        R_u = q['R_u']
        h_u = R_u - self.Rmin_u  # want h_u >= 0

        K,B,S = self.K, self.B, self.S
        J_rho = np.zeros((K, S, B), dtype=np.float64)
        J_eta = np.zeros((K, S, B), dtype=np.float64)

        u_idx = np.arange(K)
        s_u, b_u = q['s_u'], q['b_u']
        n_u = q['n_u']; I_u = q['I_u']
        A_u = (self.Pmax[b_u] * self.G[u_idx, b_u]) / n_u
        denom_u = I_u + q['eta_u'] * A_u

        term_u = (self.W / n_u) * (np.log1p(q['SNR_u']) / np.log(2.0))
        for u in range(K):
            J_rho[u, s_u[u], b_u[u]] = term_u[u]

        coef_self = (self.W * q['rho_u'] / n_u) * (1.0/np.log(2.0)) * (A_u / denom_u)
        for u in range(K):
            J_eta[u, s_u[u], b_u[u]] += coef_self[u]

        factor_u = (self.W * q['rho_u'] / n_u) * (1.0/np.log(2.0)) * ( - (q['eta_u']*A_u) / (I_u * denom_u) )
        for b in range(B):
            U_notb = np.where(b_u != b)[0]
            if U_notb.size == 0: continue
            dI_du = self.Pmax[b] * self.G[U_notb, b]
            contrib_vec = factor_u[U_notb] * dI_du
            for idx, u in enumerate(U_notb):
                for s in range(S):
                    J_eta[u, s, b] += contrib_vec[idx]

        return h_u.astype(np.float64), J_rho, J_eta

class SliceAgentView:
    def __init__(self, form: PerUEFormulation, s_index: int):
        self.F = form
        self.s = s_index
        self.S, self.B = form.S, form.B

    def objective_and_grad_slice(self, x: np.ndarray):
        f, grad = self.F.objective_and_grads(x)
        S,B = self.S, self.B
        g_rho = grad[:S*B].reshape(S,B)[self.s,:]
        g_eta = grad[S*B:].reshape(S,B)[self.s,:]
        g_slice = np.concatenate([g_rho, g_eta]).astype(np.float64)
        return f, g_slice

    def constraints_and_jacobians_slice(self, x: np.ndarray):
        h_u, J_rho, J_eta = self.F.per_ue_constraints_and_jacobians(x)
        S,B = self.S, self.B
        J_slice = np.concatenate([J_rho[:,self.s,:], J_eta[:,self.s,:]], axis=1)  # (K, 2B)
        return h_u, J_slice
