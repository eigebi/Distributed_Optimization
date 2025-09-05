from dataclasses import dataclass
import numpy as np
from typing import Tuple, Dict, Any

@dataclass
class ScenarioCfg:
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
    N0_W_per_Hz = k * T_kelvin
    F = 10 ** (noise_figure_dB / 10.0)
    noise_W = N0_W_per_Hz * bandwidth_Hz * F
    noise_dBm = 10 * np.log10(noise_W) + 30
    return noise_W, noise_dBm

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
    if shadowing_sigma_dB > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        PL_dB = PL_dB + rng.normal(0.0, shadowing_sigma_dB, size=np.shape(d_m))
    return PL_dB

def gain_matrix_from_positions(bs_xy, ue_xy, cfg: ScenarioCfg):
    rng = np.random.default_rng(cfg.seed)
    dx = ue_xy[:, [0]] - bs_xy[None, :, 0]
    dy = ue_xy[:, [1]] - bs_xy[None, :, 1]
    d = np.sqrt(dx * dx + dy * dy)
    d = np.maximum(d, cfg.min_distance_m)
    PL_dB = pathloss_dB_log_distance(d, cfg.fc_GHz, cfg.pathloss_exponent, cfg.shadowing_sigma_dB, rng=rng)
    G = 10 ** (-PL_dB / 10.0)
    return G

def associate_users_to_bs(G: np.ndarray) -> np.ndarray:
    return np.argmax(G, axis=1)

def assign_users_to_slices(K: int, probs=(0.5, 0.3, 0.2), seed: int = 2025) -> np.ndarray:
    rng = np.random.default_rng(seed + 777)
    p = np.array(probs, dtype=float); p = p / p.sum()
    return rng.choice(3, size=K, p=p)

def compute_bar_g_and_alpha(G: np.ndarray,
                            ue2bs: np.ndarray,
                            ue2slice: np.ndarray,
                            Pmax_W: np.ndarray,
                            S: int = 3):
    K, B = G.shape
    bar_g = np.zeros((S, B))
    U_sb: Dict[Tuple[int, int], np.ndarray] = {}
    for s in range(S):
        for b in range(B):
            mask = np.where((ue2slice == s) & (ue2bs == b))[0]
            U_sb[(s, b)] = mask
            bar_g[s, b] = float(np.mean(G[mask, b])) if mask.size > 0 else 1e-15

    alpha = np.zeros((B, B))
    for b in range(B):
        users_in_b = np.where(ue2bs == b)[0]
        if users_in_b.size == 0:
            avg_g = np.mean(G, axis=0)
            for bp in range(B):
                if bp == b: continue
                alpha[b, bp] = Pmax_W[bp] * float(avg_g[bp])
        else:
            avg_coupling = np.mean(G[users_in_b, :], axis=0)
            for bp in range(B):
                if bp == b: continue
                alpha[b, bp] = Pmax_W[bp] * float(avg_coupling[bp])
    return bar_g, alpha, U_sb

@dataclass
class QoSParams:
    Rreq_URLLC_perBS: np.ndarray  # (B,)

@dataclass
class ProblemData:
    W: float
    sigma2_W: float
    Pmax_W: np.ndarray
    a_sb: np.ndarray
    bar_g: np.ndarray
    alpha: np.ndarray
    weights: np.ndarray
    beta: np.ndarray
    delta: np.ndarray
    qos: QoSParams
    eps: float = 1e-9

class RANProblem:
    def __init__(self, data: ProblemData):
        self.d = data
        self.S, self.B = data.a_sb.shape

    def cell_load(self, rho: np.ndarray) -> np.ndarray:
        return np.sum(self.d.a_sb * rho, axis=0)

    def interference(self, ell: np.ndarray) -> np.ndarray:
        return self.d.alpha.dot(ell) + self.d.sigma2_W

    def SNR_like(self, eta: np.ndarray, I: np.ndarray) -> np.ndarray:
        return (eta * self.d.Pmax_W[None, :] * self.d.bar_g) / I[None, :]

    def rate(self, rho: np.ndarray, eta: np.ndarray) -> np.ndarray:
        ell = self.cell_load(rho)
        I = self.interference(ell)
        S = self.SNR_like(eta, I)
        Phi = np.log1p(S) / np.log(2.0)
        return self.d.W * rho * Phi

    def objective_and_grad(self, rho: np.ndarray, eta: np.ndarray):
        d = self.d
        ell = self.cell_load(rho); I = self.interference(ell)
        S = self.SNR_like(eta, I); Phi = np.log1p(S) / np.log(2.0)
        R = d.W * rho * Phi

        util = np.log(d.eps + R)
        f = (np.sum(d.weights * util)
             - np.sum(d.beta[:, None] * eta * d.Pmax_W[None, :])
             - np.sum(d.delta[:, None] * rho * d.W))

        d_util_dR = d.weights / (d.eps + R)
        denom = I[None, :] + eta * d.Pmax_W[None, :] * d.bar_g
        dR_deta = d.W * rho * (1.0 / np.log(2.0)) * (d.Pmax_W[None, :] * d.bar_g) / denom
        grad_eta = d_util_dR * dR_deta - d.beta[:, None] * d.Pmax_W[None, :]

        own_term = d.W * Phi
        Splus = 1.0 + S
        coeff = d.W * (1.0/np.log(2.0)) * ( - (eta * d.Pmax_W[None,:] * d.bar_g) / (I[None,:]**2) ) / Splus
        G_couple_per_b = np.sum(d_util_dR * coeff * rho, axis=0)
        coupling_scalar_per_c = d.alpha.T.dot(G_couple_per_b)
        grad_rho = d_util_dR * own_term - (d.a_sb * coupling_scalar_per_c[None, :]) - d.delta[:, None] * d.W

        extras = dict(R=R, S=S, Phi=Phi, I=I, ell=ell, util=util, d_util_dR=d_util_dR)
        return float(f), grad_rho, grad_eta, extras

    def constraints_and_jacobians(self, rho: np.ndarray, eta: np.ndarray):
        d = self.d
        Sidx = dict(H=0, L=1, M=2)
        R = self.rate(rho, eta)

        g_rho = np.sum(d.a_sb * rho, axis=0) - 1.0
        g_eta = np.sum(d.a_sb * eta, axis=0) - 1.0
        J_g_rho = d.a_sb.copy()
        J_g_eta = d.a_sb.copy()

        h_L = R[Sidx['L'], :] - d.qos.Rreq_URLLC_perBS

        ell = self.cell_load(rho); I = self.interference(ell)
        S_like = self.SNR_like(eta, I)
        Phi = np.log1p(S_like) / np.log(2.0)
        denom = I[None,:] + eta * d.Pmax_W[None,:] * d.bar_g
        dR_deta = d.W * rho * (1.0/np.log(2.0)) * (d.Pmax_W[None,:] * d.bar_g) / denom
        own_term = d.W * Phi
        Splus = 1.0 + S_like
        coeff = d.W * (1.0/np.log(2.0)) * ( - (eta * d.Pmax_W[None,:] * d.bar_g) / (I[None,:]**2) ) / Splus

        B = self.B
        grad_hL_eta = np.zeros((B, self.S, B))
        grad_hL_rho = np.zeros((B, self.S, B))
        for b in range(B):
            grad_hL_eta[b, Sidx['L'], b] = dR_deta[Sidx['L'], b]
            grad_hL_rho[b, Sidx['L'], b] += own_term[Sidx['L'], b]
            coupling_scalar = coeff[Sidx['L'], b] * rho[Sidx['L'], b]
            grad_hL_rho[b, :, :] -= d.a_sb * (d.alpha[b, :][None, :]) * coupling_scalar

        return dict(
            g_rho=g_rho, g_eta=g_eta, J_g_rho=J_g_rho, J_g_eta=J_g_eta,
            h_L=h_L, grad_hL_eta=grad_hL_eta, grad_hL_rho=grad_hL_rho
        )

def build_demo_instance(N=6, K=120, slice_probs=(0.5, 0.3, 0.2), seed=2025):
    cfg = ScenarioCfg(seed=seed)
    bs_xy, ue_xy = generate_ppp_positions(N, K, cfg.area_size_m, seed=cfg.seed)
    G = gain_matrix_from_positions(bs_xy, ue_xy, cfg)
    ue2bs = associate_users_to_bs(G)
    ue2slice = assign_users_to_slices(K, probs=slice_probs, seed=seed)

    B = N; S = 3
    Pmax_W = np.full(B, 40.0)
    bar_g, alpha, U_sb = compute_bar_g_and_alpha(G, ue2bs, ue2slice, Pmax_W, S=S)

    a_sb = np.zeros((S, B))
    for s in range(S):
        for b in range(B):
            a_sb[s, b] = 1.0 if U_sb[(s, b)].size > 0 else 0.0
    if np.all(a_sb.sum(axis=0) == 0):
        a_sb[:, :] = 1.0

    weights = np.ones((S, B))
    beta = np.array([0.3, 0.3, 0.8])
    delta = np.array([0.05, 0.05, 0.05])

    noise_W, noise_dBm = thermal_noise_power(cfg.bandwidth_Hz, cfg.noise_figure_dB)

    L_bits = 1024.0
    lam = 30.0
    tau = 0.005
    Rreq_L_perBS = np.zeros(B)
    for b in range(B):
        U_L_b = np.where((ue2slice == 1) & (ue2bs == b))[0]
        Lambda_L_b = len(U_L_b) * lam
        Rreq_L_perBS[b] = L_bits * (Lambda_L_b + 1.0 / tau)

    qos = QoSParams(Rreq_URLLC_perBS=Rreq_L_perBS)

    data = ProblemData(
        W=cfg.bandwidth_Hz,
        sigma2_W=noise_W,
        Pmax_W=Pmax_W,
        a_sb=a_sb,
        bar_g=bar_g,
        alpha=alpha,
        weights=weights,
        beta=beta,
        delta=delta,
        qos=qos,
        eps=1e-9
    )
    problem = RANProblem(data)

    info = dict(cfg=cfg, bs_xy=bs_xy, ue_xy=ue_xy, G=G, ue2bs=ue2bs, ue2slice=ue2slice,
                U_sb=U_sb, bar_g=bar_g, alpha=alpha, noise_W=noise_W, noise_dBm=noise_dBm)
    return problem, info

if __name__ == '__main__':
    prob, info = build_demo_instance(N=6, K=120, slice_probs=(0.5, 0.3, 0.2), seed=2025)
    S, B = prob.S, prob.B
    rho0 = np.where(prob.d.a_sb > 0, 1.0 / np.maximum(1, np.sum(prob.d.a_sb, axis=0, keepdims=True)), 0.0)
    eta0 = rho0.copy()
    f0, grad_rho0, grad_eta0, extras0 = prob.objective_and_grad(rho0, eta0)
    cons0 = prob.constraints_and_jacobians(rho0, eta0)
    print("Objective:", f0)
    print("h_L:", cons0['h_L'])
