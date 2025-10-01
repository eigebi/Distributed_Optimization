import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict

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
    return k * T_kelvin * bandwidth_Hz * F

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
    return G  # (K,B)

class PerUEFormulation:
    """Per-UE utility & QoS with power-aware interference, and per-UE weights theta.

    Objective with resource cost:
        f = sum_u w_u * log(eps + R_u)
            - alpha_rho * sum_{s,b} rho_{s,b}
            - alpha_eta * sum_{s,b} eta_{s,b}
    """
    def __init__(self, B:int, K:int, slice_probs=(0.5,0.3,0.2), seed:int=2025, cfg:EnvCfg=EnvCfg(),
                 alpha_rho: float = 1e-3, alpha_eta: float = 1e-3):
        self.cfg = cfg
        self.B, self.K, self.S = B, K, 3
        self.rng = np.random.default_rng(seed)
        self.bs_xy, self.ue_xy = generate_ppp_positions(B, K, cfg.area_size_m, seed=seed)
        self.G = gain_matrix_from_positions(self.bs_xy, self.ue_xy, cfg)  # (K,B)
        self.ue2bs = np.argmax(self.G, axis=1)
        probs = np.array(slice_probs, float); probs/=probs.sum()
        self.ue2slice = self.rng.choice(self.S, size=K, p=probs)
        # masks & groups
        self.a_sb = np.zeros((self.S, self.B), dtype=np.float64)
        self.U_sb: Dict[Tuple[int,int], np.ndarray] = {}
        for s in range(self.S):
            for b in range(self.B):
                U = np.where((self.ue2slice==s) & (self.ue2bs==b))[0]
                self.U_sb[(s,b)] = U
                if U.size>0: self.a_sb[s,b]=1.0
        if np.all(self.a_sb.sum(axis=0)==0): self.a_sb[:,:]=1.0
        # per-UE allocation weights theta (default equal within (s,b))
        self.theta = np.zeros((self.S, self.B, self.K), dtype=np.float64)
        for s in range(self.S):
            for b in range(self.B):
                U = self.U_sb[(s,b)]
                if U.size>0:
                    self.theta[s,b,U] = 1.0/float(U.size)
        # system params
        self.W = cfg.bandwidth_Hz
        self.sigma2 = thermal_noise_power(cfg.bandwidth_Hz, cfg.noise_figure_dB)
        self.Pmax = np.full(self.B, 40.0, dtype=np.float64)  # W
        # UE weights & QoS
        self.w_u = np.ones(self.K, dtype=np.float64)
        Rmin_map={0: 1e3, 1: 1e2, 2: 1e2}
        self.Rmin_u = np.array([Rmin_map[int(self.ue2slice[u])] for u in range(self.K)], dtype=np.float64)
        self.eps = 1e-9
        # resource cost weights
        self.alpha_rho = float(alpha_rho)
        self.alpha_eta = float(alpha_eta)

    def set_theta(self, theta_sbk):
        assert theta_sbk.shape == (self.S, self.B, self.K)
        self.theta = theta_sbk.astype(np.float64)

    def split_x(self, x: np.ndarray):
        S,B = self.S, self.B
        assert x.shape[0]==S*2*B
        rho = x[:S*B].reshape(S,B).copy()
        eta = x[S*B:].reshape(S,B).copy()
        return rho, eta

    def merge(self, rho, eta):
        return np.concatenate([rho.flatten(), eta.flatten()])

    def per_ue_quantities(self, rho: np.ndarray, eta: np.ndarray):
        K,B,S = self.K,self.B,self.S
        idx = np.arange(K)
        s_u = self.ue2slice
        b_u = self.ue2bs
        theta_u = np.array([self.theta[s_u[u], b_u[u], u] for u in range(K)])
        BW_u = self.W * theta_u * rho[s_u, b_u]
        P_sig = theta_u * eta[s_u, b_u] * self.Pmax[b_u] * self.G[idx, b_u]
        ell_b = np.sum(self.a_sb * eta, axis=0)
        mask = np.ones((K,B), dtype=bool); mask[idx, b_u]=False
        I_u = (self.Pmax[None,:]*ell_b[None,:]*self.G*mask).sum(axis=1) + self.sigma2
        SNR_u = P_sig / I_u
        R_u = BW_u * (np.log1p(SNR_u) / np.log(2.0))
        return dict(s_u=s_u, b_u=b_u, theta_u=theta_u, BW_u=BW_u, P_sig=P_sig, I_u=I_u, SNR_u=SNR_u, R_u=R_u)

    def objective_and_grads(self, x: np.ndarray):
        rho, eta = self.split_x(x)
        q = self.per_ue_quantities(rho, eta)
        R_u = q['R_u']; w=self.w_u; eps=self.eps
        f = float(np.sum(w*np.log(eps+R_u)))
        # subtract resource costs
        f -= self.alpha_rho * float(np.sum(rho))
        f -= self.alpha_eta * float(np.sum(eta))

        K,B,S = self.K,self.B,self.S
        idx = np.arange(K); s_u=q['s_u']; b_u=q['b_u']
        theta_u=q['theta_u']; I_u=q['I_u']
        dfdR = w / (eps + R_u)
        # dR/drho
        dR_drho = np.zeros((S,B), dtype=np.float64)
        term = self.W * theta_u * (np.log1p(q['SNR_u']) / np.log(2.0))
        for u in range(K):
            dR_drho[s_u[u], b_u[u]] += dfdR[u]*term[u]
        # dR/deta
        dR_deta = np.zeros((S,B), dtype=np.float64)
        A_u = theta_u*self.Pmax[b_u]*self.G[idx,b_u]
        denom = I_u + A_u*eta[s_u,b_u]
        coef_self = (self.W*theta_u*rho[s_u,b_u])*(1.0/np.log(2.0))*(A_u/denom)
        for u in range(K):
            dR_deta[s_u[u], b_u[u]] += dfdR[u]*coef_self[u]
        factor = (self.W*theta_u*rho[s_u,b_u])*(1.0/np.log(2.0))*( - (eta[s_u,b_u]*A_u) / (I_u*denom) )
        for b in range(B):
            U = np.where(b_u!=b)[0]
            if U.size==0: continue
            dI = self.Pmax[b]*self.G[U,b]
            contrib = np.sum(dfdR[U]*factor[U]*dI)
            for s in range(S):
                dR_deta[s,b] += contrib
        # add cost gradients (-alpha on each element)
        dR_drho -= self.alpha_rho
        dR_deta -= self.alpha_eta
        grad = np.concatenate([dR_drho.flatten(), dR_deta.flatten()]).astype(np.float64)
        return -f, -grad

    def per_ue_constraints_and_jacobians(self, x: np.ndarray):
        rho, eta = self.split_x(x)
        q = self.per_ue_quantities(rho, eta)
        R_u = q['R_u']
        h_u = R_u - self.Rmin_u
        K,B,S = self.K,self.B,self.S
        idx = np.arange(K); s_u=q['s_u']; b_u=q['b_u']
        theta_u=q['theta_u']; I_u=q['I_u']
        A_u = theta_u*self.Pmax[b_u]*self.G[idx,b_u]
        denom = I_u + A_u*(eta[s_u,b_u])
        J_rho = np.zeros((K,S,B), dtype=np.float64)
        J_eta = np.zeros((K,S,B), dtype=np.float64)
        term = self.W*theta_u*(np.log1p(q['SNR_u'])/np.log(2.0))
        for u in range(K):
            J_rho[u, s_u[u], b_u[u]] = term[u]
        coef_self = (self.W*theta_u*rho[s_u,b_u])*(1.0/np.log(2.0))*(A_u/denom)
        for u in range(K):
            J_eta[u, s_u[u], b_u[u]] += coef_self[u]
        factor = (self.W*theta_u*rho[s_u,b_u])*(1.0/np.log(2.0))*( - (eta[s_u,b_u]*A_u) / (I_u*denom) )
        for b in range(B):
            U = np.where(b_u!=b)[0]
            if U.size==0: continue
            dI = self.Pmax[b]*self.G[U,b]
            contrib = factor[U]*dI
            for k,u in enumerate(U):
                for s in range(S):
                    J_eta[u, s, b] += contrib[k]
        return h_u.astype(np.float64), J_rho, J_eta

if __name__ == "__main__":
    # Minimal usage example
    B, K = 4, 50
    F = PerUEFormulation(B=B, K=K, alpha_rho=1e-3, alpha_eta=1e-3)

    # init rho, eta (equal split over active (s,b))
    rho0 = 0.9 * F.a_sb / np.maximum(1, F.a_sb.sum(axis=0, keepdims=True))
    eta0 = 0.9 * F.a_sb / np.maximum(1, F.a_sb.sum(axis=0, keepdims=True))
    x0 = F.merge(rho0, eta0)

    f, g = F.objective_and_grads(x0)
    print("Objective:", f, "||grad||:", np.linalg.norm(g))

    h, Jrho, Jeta = F.per_ue_constraints_and_jacobians(x0)
    print("Min per-UE margin:", float(h.min()))
    print("Shapes -> h:", h.shape, "Jrho:", Jrho.shape, "Jeta:", Jeta.shape)
