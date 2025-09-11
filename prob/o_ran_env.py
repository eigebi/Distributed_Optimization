from dataclasses import dataclass
import numpy as np
from typing import Tuple, Dict, Any

from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.linalg import block_diag

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
        users_in_b = np.where(ue2bs == b)[0] # if no user in b, all the users cause interference to b
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

from dataclasses import dataclass

@dataclass
class QoSParams:
    Rmin_eMBB: float
    Rreq_URLLC_perBS: np.ndarray

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
        return np.sum(self.d.a_sb * rho, axis=0) # spectrum load per BS

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

        h_H = np.sum(R[Sidx['H'], :]) - d.qos.Rmin_eMBB
        h_L = R[Sidx['L'], :] - d.qos.Rreq_URLLC_perBS

        ell = self.cell_load(rho); I = self.interference(ell)
        S_like = self.SNR_like(eta, I)
        Phi = np.log1p(S_like) / np.log(2.0)
        denom = I[None,:] + eta * d.Pmax_W[None,:] * d.bar_g
        dR_deta = d.W * rho * (1.0/np.log(2.0)) * (d.Pmax_W[None,:] * d.bar_g) / denom
        own_term = d.W * Phi
        Splus = 1.0 + S_like
        coeff = d.W * (1.0/np.log(2.0)) * ( - (eta * d.Pmax_W[None,:] * d.bar_g) / (I[None,:]**2) ) / Splus

        grad_hH_eta = np.zeros_like(eta)
        grad_hH_rho = np.zeros_like(rho)
        G_couple_H_per_b = coeff[Sidx['H'], :] * rho[Sidx['H'], :]
        coupling_H_per_c = d.alpha.T.dot(G_couple_H_per_b)
        grad_hH_eta[Sidx['H'], :] = dR_deta[Sidx['H'], :]
        grad_hH_rho[Sidx['H'], :] += own_term[Sidx['H'], :]
        grad_hH_rho -= d.a_sb * coupling_H_per_c[None, :]

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
            h_H=h_H, h_L=h_L,
            grad_hH_eta=grad_hH_eta, grad_hH_rho=grad_hH_rho,
            grad_hL_eta=grad_hL_eta, grad_hL_rho=grad_hL_rho
        )

def build_demo_instance(N=6, K=40, slice_probs=(0.5, 0.3, 0.2), seed=2025):
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

    L_bits = 256.0
    lam = 30.0
    tau = 0.01
    Rreq_L_perBS = np.zeros(B)
    for b in range(B):
        U_L_b = np.where((ue2slice == 1) & (ue2bs == b))[0]
        Lambda_L_b = len(U_L_b) * lam
        Rreq_L_perBS[b] = L_bits * (Lambda_L_b + 1.0 / tau)

    SNR_opt = (Pmax_W * np.max(bar_g, axis=0)) / noise_W
    Rcap_per_bs = cfg.bandwidth_Hz * np.log2(1.0 + np.maximum(SNR_opt, 1e-12))
    Rmin_eMBB = 0.10 * np.sum(Rcap_per_bs)

    qos = QoSParams(Rmin_eMBB=Rmin_eMBB, Rreq_URLLC_perBS=Rreq_L_perBS)

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
                U_sb=U_sb, bar_g=bar_g, alpha=alpha, noise_W=noise_W, noise_dBm=noise_dBm,
                Rmin_eMBB=Rmin_eMBB, Rreq_L_perBS=Rreq_L_perBS)
    return problem, info



def pack_vars(rho, eta): return np.concatenate([rho.flatten(), eta.flatten()])
def unpack_vars(x):
    n=S*B; rho=x[:n].reshape(S,B); eta=x[n:].reshape(S,B); return rho,eta
from scipy.optimize import Bounds, NonlinearConstraint




if __name__ == '__main__':
    prob, info = build_demo_instance(N=6, K=120, slice_probs=(0.5, 0.3, 0.2), seed=2025)
    S, B = prob.S, prob.B
    rho0 = np.where(prob.d.a_sb > 0, 1.0 / np.maximum(1, np.sum(prob.d.a_sb, axis=0, keepdims=True)), 0.0)
    eta0 = rho0.copy()
    x0 = pack_vars(rho0, eta0)

    lambda_util = 1.0  # 只缩放效用(∑w log(ε+R))，成本项不缩放

    def objective_with_lambda(x):
        rho,eta = unpack_vars(x)
        f_base, grad_rho_base, grad_eta_base, extras = prob.objective_and_grad(rho, eta)
        # 将 base 梯度拆回： grad_base = grad_util - grad_cost
        cost_grad_eta = prob.d.beta[:,None]*prob.d.Pmax_W[None,:]
        cost_grad_rho = prob.d.delta[:,None]*prob.d.W
        # 目标：lambda*util - cost
        util = extras['util']
        f_lambda = (lambda_util*np.sum(prob.d.weights*util)
                    - np.sum(prob.d.beta[:,None]*eta*prob.d.Pmax_W[None,:])
                    - np.sum(prob.d.delta[:,None]*rho*prob.d.W))
        # 梯度：lambda*grad_util - grad_cost = lambda*(grad_base+cost_grad) - cost_grad
        grad_rho = lambda_util*(grad_rho_base + cost_grad_rho) - cost_grad_rho
        grad_eta = lambda_util*(grad_eta_base + cost_grad_eta) - cost_grad_eta
        grad = pack_vars(grad_rho, grad_eta)
        return -f_lambda, -grad  # SciPy 是最小化

    # bounds: 0 ≤ rho ≤ a, 0 ≤ eta ≤ a
    lb = np.zeros_like(x0)
    ub = np.concatenate([prob.d.a_sb.flatten(), prob.d.a_sb.flatten()])
    bounds = Bounds(lb, ub)

    # per-BS 资源约束
    def con_g_rho(x):
        rho,eta = unpack_vars(x); return prob.constraints_and_jacobians(rho,eta)['g_rho']
    def jac_g_rho(x):
        rho,eta = unpack_vars(x); J = prob.constraints_and_jacobians(rho,eta)['J_g_rho']
        J_full = np.zeros((B, 2*S*B))
        for b in range(B):
            for s in range(S):
                J_full[b, s*B+b] = J[s,b]
        return J_full
    def con_g_eta(x):
        rho,eta = unpack_vars(x); return prob.constraints_and_jacobians(rho,eta)['g_eta']
    def jac_g_eta(x):
        rho,eta = unpack_vars(x); J = prob.constraints_and_jacobians(rho,eta)['J_g_eta']
        J_full = np.zeros((B, 2*S*B))
        for b in range(B):
            for s in range(S):
                J_full[b, S*B + (s*B+b)] = J[s,b]
        return J_full
    nlc_g_rho = NonlinearConstraint(con_g_rho, -np.inf*np.ones(B), np.zeros(B), jac=jac_g_rho)
    nlc_g_eta = NonlinearConstraint(con_g_eta, -np.inf*np.ones(B), np.zeros(B), jac=jac_g_eta)

    # eMBB sum-rate
    def con_h_H(x):
        rho,eta = unpack_vars(x); return np.array([prob.constraints_and_jacobians(rho,eta)['h_H']])
    def jac_h_H(x):
        rho,eta = unpack_vars(x); cons=prob.constraints_and_jacobians(rho,eta)
        Jrho, Jeta = cons['grad_hH_rho'], cons['grad_hH_eta']
        J_full = np.zeros((1, 2*S*B))
        for s in range(S):
            for b in range(B):
                J_full[0, s*B+b] = Jrho[s,b]
                J_full[0, S*B + (s*B+b)] = Jeta[s,b]
        return J_full
    nlc_h_H = NonlinearConstraint(con_h_H, np.zeros(1), np.inf*np.ones(1), jac=jac_h_H)

    # URLLC per-BS
    def con_h_L(x):
        rho,eta = unpack_vars(x); return prob.constraints_and_jacobians(rho,eta)['h_L']
    def jac_h_L(x):
        rho,eta = unpack_vars(x); cons=prob.constraints_and_jacobians(rho,eta)
        Jrho, Jeta = cons['grad_hL_rho'], cons['grad_hL_eta']  # (B,S,B)
        J_full = np.zeros((B, 2*S*B))
        for b in range(B):
            for s in range(S):
                for c in range(B):
                    J_full[b, s*B + c] += Jrho[b,s,c]
                    J_full[b, S*B + (s*B+c)] += Jeta[b,s,c]
        return J_full
    nlc_h_L = NonlinearConstraint(con_h_L, np.zeros(B), np.inf*np.ones(B), jac=jac_h_L)

    # solve
    res = minimize(fun=lambda x: objective_with_lambda(x)[0],
                x0=x0,
                jac=lambda x: objective_with_lambda(x)[1],
                method='trust-constr',
                bounds=bounds,
                constraints=[nlc_g_rho, nlc_g_eta, nlc_h_H, nlc_h_L],
                options=dict(verbose=3, maxiter=20000, xtol=1e-4, gtol=1e-2, initial_tr_radius=0.8, initial_constr_penalty=0.2))
    

    rho_opt, eta_opt = unpack_vars(res.x)
    cons_final = prob.constraints_and_jacobians(rho_opt, eta_opt)
    print("\n=== Optimization result ===")
    print("success:", res.success, "| status:", res.status)
    print("message:", res.message)
    print("final objective (max):", -res.fun)
    print("max g_rho (≤0):", np.max(cons_final['g_rho']))
    print("max g_eta (≤0):", np.max(cons_final['g_eta']))
    print("h_H (≥0):", cons_final['h_H'])
    print("min h_L (≥0):", np.min(cons_final['h_L']))