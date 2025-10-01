
import numpy as np
from dataclasses import dataclass
from typing import Optional
from o_ran_env import PerUEFormulation, EnvCfg

@dataclass
class InnerResult:
    feasible: bool
    P_b: np.ndarray         # (B,) minimal BS total powers found
    p_u: np.ndarray         # (K,) per-UE powers (sum per BS equals P_b)
    iters: int
    max_violation: float    # max(P_b - Pmax)_+

def compute_gamma_targets(F: PerUEFormulation, rho: np.ndarray) -> np.ndarray:
    """
    For each UE u, compute target SINR gamma_u from its Rmin and allocated bandwidth W_u = theta * rho * W.
    gamma_u = 2^{ Rmin_u / W_u } - 1.  If W_u is ~0, set gamma_u very large to mark infeasibility.
    """
    S, B, K = F.S, F.B, F.K
    s_u = F.ue2slice
    b_u = F.ue2bs
    idx = np.arange(K)

    theta_u = np.array([F.theta[s_u[u], b_u[u], u] for u in range(K)], dtype=float)
    W_u = F.W * theta_u * rho[s_u, b_u]
    epsW = 1e-12
    W_eff = np.maximum(W_u, epsW)
    gamma_u = np.exp2(F.Rmin_u / W_eff) - 1.0
    return gamma_u

@dataclass
class InnerParams:
    max_iter: int = 2000
    tol: float = 1e-7

def inner_min_power(F: PerUEFormulation, rho: np.ndarray, params: InnerParams = InnerParams()) -> InnerResult:
    """
    Fixed-point iteration over BS total powers P_b:
      For each UE u served by b_u,
        p_u = (gamma_u / g_{u,b_u}) * ( sum_{b'!=b_u} g_{u,b'} P_{b'} + sigma^2 )
      Then P_b = sum_{u: b_u=b} p_u
    This is a standard monotone interference mapping; if feasible, converges to minimal P_b.
    """
    B, K = F.B, F.K
    s_u = F.ue2slice
    b_u = F.ue2bs
    idx = np.arange(K)

    gamma_u = compute_gamma_targets(F, rho)      # (K,)
    g_ub = F.G[idx, b_u]                         # (K,) serving-link gain
    G_other = F.G.copy()
    for u in range(K):
        G_other[u, b_u[u]] = 0.0                 # zero out serving BS for interference sum

    P_b = np.zeros(B, dtype=float)
    p_u = np.zeros(K, dtype=float)
    sigma2 = F.sigma2

    for it in range(1, params.max_iter+1):
        I_u = (G_other @ P_b) + sigma2
        denom = np.maximum(g_ub, 1e-20)
        p_req = gamma_u * I_u / denom
        P_new = np.zeros_like(P_b)
        for b in range(B):
            P_new[b] = p_req[b_u == b].sum()

        if np.linalg.norm(P_new - P_b, ord=np.inf) <= params.tol * (1.0 + np.linalg.norm(P_b, ord=np.inf)):
            P_b = P_new
            p_u = p_req
            break
        P_b = P_new
        p_u = p_req

    viol = np.maximum(P_b - F.Pmax, 0.0)
    feasible = bool(np.all(viol <= 1e-9))
    return InnerResult(feasible=feasible, P_b=P_b, p_u=p_u, iters=it, max_violation=float(viol.max() if viol.size>0 else 0.0))

@dataclass
class OuterResult:
    success: bool
    rho: np.ndarray
    inner: InnerResult
    obj: float
    msg: str

def outer_optimize_rho_min_total_power(
    F: PerUEFormulation,
    rho0: Optional[np.ndarray] = None,
    max_outer: int = 200,
    step: float = 0.2,
    rho_min: float = 1e-4,
    verbose: bool = True
) -> OuterResult:
    """
    Simple centralized outer loop that adjusts rho on each BS to reduce overload (P_b>Pmax) and total power.
    Heuristic: projected gradient-like update driven by inner BS overload and per-UE bandwidth scarcity.
    Keeps sum_s rho_{s,b} <= 1 and rho >= rho_min on active (s,b); unused (no UE) entries are 0.
    """
    S, B = F.S, F.B

    # Initialize rho0: equal split over active (s,b)
    if rho0 is None:
        rho = np.zeros((S,B), dtype=float)
        for b in range(B):
            active = np.where(F.a_sb[:,b] > 0)[0]
            if active.size > 0:
                rho[active, b] = (1.0 - 1e-2) / active.size
    else:
        rho = rho0.copy()

    rho *= F.a_sb

    best = None
    for it in range(1, max_outer+1):
        inner = inner_min_power(F, rho)
        obj = float(inner.P_b.sum()) + 1e-6 * float(rho.sum())

        if verbose and (it % 10 == 1 or it == 1):
            print(f"[outer {it}] obj={obj:.6f} feasible={inner.feasible} max_over={inner.max_violation:.3e}")

        if inner.feasible and (best is None or obj < best.obj):
            best = OuterResult(True, rho.copy(), inner, obj, "feasible snapshot")

        # Build gradient-like update for rho
        grad_rho = np.zeros_like(rho)
        overload = np.maximum(inner.P_b - F.Pmax, 0.0)
        for b in range(F.B):
            active_s = np.where(F.a_sb[:,b] > 0)[0]
            if active_s.size == 0: continue
            if overload[b] > 0:
                counts = np.array([F.U_sb[(s,b)].size for s in active_s], dtype=float)
                if counts.sum() == 0: counts = np.ones_like(counts)
                for j,s in enumerate(active_s):
                    grad_rho[s,b] += overload[b] * (counts[j] / (counts.sum()+1e-9))
            else:
                for s in active_s:
                    grad_rho[s,b] -= 0.1 * rho[s,b]

        rho = rho + step * grad_rho
        rho = np.maximum(rho, 0.0) * F.a_sb
        for b in range(F.B):
            s_active = np.where(F.a_sb[:,b] > 0)[0]
            ssum = rho[s_active, b].sum()
            if ssum > 1.0:
                rho[s_active, b] = rho[s_active, b] / ssum
            if s_active.size>0:
                rho[s_active, b] = np.maximum(rho[s_active, b], rho_min)
                ssum = rho[s_active, b].sum()
                if ssum > 1.0:
                    rho[s_active, b] = rho[s_active, b] / ssum

    if best is None:
        inner = inner_min_power(F, rho)
        return OuterResult(False, rho, inner, float('inf'), "no feasible point found")
    return best

if __name__ == "__main__":
    B, K = 15, 300
    F = PerUEFormulation(B=B, K=K, cfg=EnvCfg(area_size_m=(2000,2000)))
    # Example: moderate per-UE thresholds (bps)
    # F.Rmin_u[:] = 5e5

    res = outer_optimize_rho_min_total_power(F, verbose=True, max_outer=100, step=0.1)
    print("Success:", res.success, res.msg)
    print("Total power (W):", res.inner.P_b.sum())
    print("Max BS power:", float(res.inner.P_b.max()))
    print("Any overload:", res.inner.max_violation)
    print("Best rho shape:", res.rho.shape)
        # ------------------------------
    # Layered solve: outer rho, inner minimal power -> map to eta, then evaluate
    # ------------------------------

    def map_power_to_eta(F, p_u, kappa_psd: float = 3.0):
        """
        Map per-UE powers to slice-BS power shares eta_{s,b} with an optional PSD cap:
            eta_{s,b} = p_{s,b} / Pmax[b],   p_{s,b} = sum_{u in U_{s,b}} p_u
            eta_{s,b} <= kappa_psd * rho_{s,b}
        Finally enforce sum_s eta_{s,b} <= 1 by normalization per BS if needed.
        """
        S, B, K = F.S, F.B, F.K
        eta = np.zeros((S, B), dtype=float)
        # aggregate p_{s,b}
        for s in range(S):
            for b in range(B):
                U = F.U_sb[(s, b)]
                if U.size > 0:
                    eta[s, b] = p_u[U].sum() / F.Pmax[b]  # provisional eta

        # PSD guard: eta_{s,b} <= kappa * rho_{s,b}
        # reuse the best rho from outer loop
        rho_star = res.rho
        eta = np.minimum(eta, kappa_psd * rho_star)

        # Enforce per-BS sum_s eta_{s,b} <= 1
        for b in range(B):
            ssum = eta[:, b].sum()
            if ssum > 1.0:
                eta[:, b] /= ssum
        return eta

    def evaluate_per_ue_rates(F, rho, eta):
        """
        Compute per-UE rate under (rho, eta) using the same physical model as in your formulation:
            W_u = theta_u * rho_{s,b} * W
            P_sig_u = theta_u * eta_{s,b} * Pmax[b] * g[u,b]
            I_u = sum_{b' != b} Pmax[b'] * (sum_s eta_{s,b'}) * g[u,b'] + sigma^2
            R_u = W_u * log2(1 + P_sig_u / I_u)
        Returns: R_u (K,), min_margin (min(R_u - Rmin_u))
        """
        K, B, S = F.K, F.B, F.S
        idx = np.arange(K)
        s_u = F.ue2slice
        b_u = F.ue2bs

        # per-UE bandwidth and signal power
        theta_u = np.array([F.theta[s_u[u], b_u[u], u] for u in range(K)], dtype=float)
        W_u = F.W * theta_u * rho[s_u, b_u]
        P_sig = theta_u * eta[s_u, b_u] * F.Pmax[b_u] * F.G[idx, b_u]

        # interference with BS-load (sum over slices) on each b'
        ell_b = np.sum(eta, axis=0)  # sum_s eta_{s,b}
        mask = np.ones((K, B), dtype=bool)
        mask[idx, b_u] = False
        I_u = (F.Pmax[None, :] * ell_b[None, :] * F.G * mask).sum(axis=1) + F.sigma2

        SNR_u = P_sig / np.maximum(I_u, 1e-20)
        R_u = W_u * (np.log1p(SNR_u) / np.log(2.0))
        min_margin = float(np.min(R_u - F.Rmin_u))
        return R_u, min_margin

    # 1) 外层结果（rho*、最小功率）
    rho_star = res.rho
    p_u_star = res.inner.p_u
    P_b_star = res.inner.P_b

    print("\n--- Layered mapping to (eta) and evaluation ---")
    print("Feasible (inner):", res.inner.feasible, "Max overload:", res.inner.max_violation)
    print("Total power (inner minimal): {:.3f} W, max per-BS: {:.3f} W".format(P_b_star.sum(), P_b_star.max()))

    # 2) 从 per-UE 功率聚合得到 eta，并做 PSD 护栏与归一化
    eta_star = map_power_to_eta(F, p_u_star, kappa_psd=3.0)  # kappa_psd可调 = 峰均功率比上限

    # 3) 回代计算 per-UE 速率，检查 QoS
    R_u, min_margin = evaluate_per_ue_rates(F, rho_star, eta_star)
    print("Achieved per-UE rates: mean = {:.3f} Mbps, 5%-tile = {:.3f} Mbps".format(
        np.mean(R_u)/1e6, np.percentile(R_u, 5)/1e6))
    print("QoS min margin (bps):", min_margin)

    # 4) 打印部分统计
    ell_b = eta_star.sum(axis=0)
    print("Per-BS sum eta (should <= 1): max = {:.3f}".format(ell_b.max()))
    sr_b = rho_star.sum(axis=0)
    print("Per-BS sum rho (should <= 1): max = {:.3f}".format(sr_b.max()))

    # 5) 可选：把得到的 (rho*, eta*) 当作原 per-UE 优化器的 warm-start
    #    （如果你后续要再跑 utility maximization 或分布式算法）
    # x_warm = np.concatenate([rho_star.flatten(), eta_star.flatten()])
    # ... feed x_warm into your SLSQP/ADMM/LSTM pipeline ...
