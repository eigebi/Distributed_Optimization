# power_control_bench_ieee.py
# Pure-power channel-gain environment + baselines + Interference Pricing + Pure-power WMMSE
# + DMC with FULL global copies per agent (X: KxKxN) and FULL gradients wrt global Z (KxN)

import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Utils
# ============================================================

def proj_l1_ball_nonneg(v: np.ndarray, z: float) -> np.ndarray:
    """
    Euclidean projection onto {x >= 0, sum(x) <= z}.
    If sum(max(v,0)) <= z -> max(v,0).
    Else -> projection onto simplex {x>=0, sum x = z}.
    """
    v = np.asarray(v, dtype=np.float64)
    v_pos = np.maximum(v, 0.0)
    if v_pos.sum() <= z:
        return v_pos

    u = np.sort(v_pos)[::-1]
    cssv = np.cumsum(u)
    idx = np.arange(1, len(u) + 1)
    rho_idx = np.nonzero(u * idx > (cssv - z))[0]
    if len(rho_idx) == 0:
        return np.zeros_like(v_pos)
    rho = rho_idx[-1]
    theta = (cssv[rho] - z) / (rho + 1.0)
    return np.maximum(v_pos - theta, 0.0)


def safe_log2_1p(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(x, 0.0)) / np.log(2.0)


# ============================================================
# Environment: K user-pairs, N channels (power-domain IC)
# ============================================================

class PowerControlEnv:
    """
    Pure power-domain IC (TIN = treat interference as noise):

      SINR_{k,n}(P) = g_{kk,n} P_{k,n} / (sigma2 + sum_{j!=k} g_{jk,n} P_{j,n})
      R_k(P) = sum_n log2(1 + SINR_{k,n})

    Here g[j,k,n] is a NONNEGATIVE power gain (dimensionless):
      g[j,k,n] = gain from Tx j -> Rx k on subchannel n
    No phase is modeled.

    Units:
      P in Watts, noise_power in Watts, g dimensionless -> SINR dimensionless -> rate in bps/Hz.
    """

    def __init__(
        self,
        K: int = 20,
        N: int = 8,
        area_side: float = 1000.0,
        pathloss_exp: float = 3.5,
        noise_power: float = 1e-9,
        P_max: float = 1.0,
        R_min: float = 1.0,
        seed: int = 0,
        direct_boost_amp: float = 2.0,
    ):
        self.K = int(K)
        self.N = int(N)
        self.area_side = float(area_side)
        self.pathloss_exp = float(pathloss_exp)
        self.noise_power = float(noise_power)
        self.P_max = float(P_max)
        self.R_min = float(R_min)
        self.seed = int(seed)
        self.direct_boost_amp = float(direct_boost_amp)

        rng = np.random.default_rng(self.seed)

        # Tx/Rx positions: each pair close -> direct links stronger on average
        tx = rng.uniform(0, self.area_side, size=(self.K, 2))
        rx = tx + rng.normal(scale=self.area_side * 0.03, size=(self.K, 2))
        rx = np.clip(rx, 0.0, self.area_side)
        self.tx_pos = tx
        self.rx_pos = rx

        # Distances d_{j->k}
        d = np.zeros((self.K, self.K), dtype=np.float64)
        for j in range(self.K):
            diff = self.rx_pos - self.tx_pos[j]
            d[j, :] = np.sqrt((diff**2).sum(axis=1)) + 1.0  # +1m to avoid singularity

        # Pathloss (dimensionless)
        L = d ** (-self.pathloss_exp)  # (K,K)

        # Small-scale fading power: if h ~ CN(0,1), then |h|^2 ~ Exp(1)
        fading_pow = rng.exponential(scale=1.0, size=(self.K, self.K, self.N))  # >=0

        # Power gains g[j,k,n] = L[j,k] * fading_pow[j,k,n]
        self.g = L[:, :, None] * fading_pow

        # Optional boost for direct links (interpret as amplitude boost -> power gain boost squared)
        if self.direct_boost_amp != 1.0:
            boost_pow = self.direct_boost_amp ** 2
            for k in range(self.K):
                self.g[k, k, :] *= boost_pow

    # ---- constraints / projection ----
    def project_budget(self, P: np.ndarray) -> np.ndarray:
        """
        Row-wise projection: for each user k, project P[k,:] onto {p>=0, sum p <= P_max}.
        Works for shape (K,N) only.
        """
        P = np.asarray(P, dtype=np.float64)
        assert P.shape == (self.K, self.N)
        out = np.zeros_like(P)
        for k in range(self.K):
            out[k, :] = proj_l1_ball_nonneg(P[k, :], self.P_max)
        return out

    # ---- SINR / rates ----
    def sinr(self, P: np.ndarray) -> np.ndarray:
        P = np.asarray(P, dtype=np.float64)
        assert P.shape == (self.K, self.N)

        # I_total[k,n] = sigma2 + sum_j g[j,k,n] P[j,n]
        I_total = self.noise_power + np.einsum("jkn,jn->kn", self.g, P)  # (K,N)
        sig = self.g[np.arange(self.K), np.arange(self.K), :] * P        # (K,N)
        denom = np.maximum(I_total - sig, 1e-18)                         # sigma2 + sum_{j!=k}
        return sig / denom

    def rates_per_user(self, P: np.ndarray) -> np.ndarray:
        return safe_log2_1p(self.sinr(P)).sum(axis=1)  # (K,)

    def sum_rate(self, P: np.ndarray) -> float:
        return float(np.sum(self.rates_per_user(P)))

    def min_rate(self, P: np.ndarray) -> float:
        return float(np.min(self.rates_per_user(P)))

    def g_constraints(self, P: np.ndarray) -> np.ndarray:
        # QoS: R_k(P) >= R_min  <=>  g_k(P) = R_min - R_k(P) <= 0
        return self.R_min - self.rates_per_user(P)

    def qos_violation(self, P: np.ndarray) -> float:
        v = np.maximum(self.g_constraints(P), 0.0)
        return float(np.mean(v**2))

    # ---- gradients for power-domain model ----
    def grad_rates_all(self, P: np.ndarray) -> np.ndarray:
        """
        dR_dP[k,i,n] = ∂R_k / ∂P_{i,n}, shape (K,K,N)
        under TIN SINR model.
        """
        P = np.asarray(P, dtype=np.float64)
        K, N = self.K, self.N
        assert P.shape == (K, N)

        I_total = self.noise_power + np.einsum("jkn,jn->kn", self.g, P)  # (K,N)
        gkk = self.g[np.arange(K), np.arange(K), :]                      # (K,N)
        sig = gkk * P                                                    # (K,N)
        D = np.maximum(I_total - sig, 1e-18)                             # (K,N)
        sinr = sig / D

        coeff = (1.0 / np.log(2.0)) * (1.0 / np.maximum(1.0 + sinr, 1e-18))  # (K,N)

        dR_dP = np.zeros((K, K, N), dtype=np.float64)

        # ∂sinr_k / ∂P_{k} = gkk / D
        for k in range(K):
            d_sinr_d_pk = gkk[k, :] / np.maximum(D[k, :], 1e-18)
            dR_dP[k, k, :] = coeff[k, :] * d_sinr_d_pk

        # ∂sinr_k / ∂P_{i} (i!=k) = -(gkk * Pk) * gik / D^2
        for k in range(K):
            for i in range(K):
                if i == k:
                    continue
                gik = self.g[i, k, :]  # Tx i -> Rx k
                d_sinr_d_pi = -(gkk[k, :] * P[k, :]) * gik / np.maximum(D[k, :]**2, 1e-18)
                dR_dP[k, i, :] = coeff[k, :] * d_sinr_d_pi

        return dR_dP

    def grad_sum_rate(self, P: np.ndarray) -> np.ndarray:
        # ∇ sum_k R_k = sum_k ∇R_k
        return np.sum(self.grad_rates_all(P), axis=0)  # (K,N)

    def grad_local_rate_block(self, k: int, P: np.ndarray) -> np.ndarray:
        # ∇_{P[k,:]} R_k  (this is only the "own-block" gradient, not full)
        return self.grad_rates_all(P)[k, k, :]  # (N,)

    def local_constraint_value(self, k: int, P: np.ndarray) -> float:
        return float(self.g_constraints(P)[k])

    def local_constraint_grad_block(self, k: int, P: np.ndarray) -> np.ndarray:
        # ∇_{P[k,:]} g_k = - ∇_{P[k,:]} R_k
        return -self.grad_local_rate_block(k, P)

    # ============================================================
    # NEW hooks for FULL-gradient DMC (agent sees gradient wrt ALL users' powers)
    # ============================================================
    def grad_local_objective_full(self, a: int, Z: np.ndarray) -> np.ndarray:
        """
        Local objective for agent a:
          minimize f_a(Z) = -R_a(Z)   (equivalently maximize R_a)
        Return ∇ f_a(Z) w.r.t. FULL Z, shape (K,N).
        """
        dR = self.grad_rates_all(Z)          # (K,K,N)
        return -dR[a, :, :]                  # (K,N)

    def local_constraint_grad_full(self, a: int, Z: np.ndarray) -> np.ndarray:
        """
        QoS constraint for agent a: g_a(Z) = R_min - R_a(Z) <= 0
        Return ∇ g_a(Z) w.r.t. FULL Z, shape (K,N).
        """
        dR = self.grad_rates_all(Z)
        return -dR[a, :, :]                  # (K,N)


# ============================================================
# Baselines
# ============================================================

def _init_power(env: PowerControlEnv, seed: int, scale: float = 0.01) -> np.ndarray:
    rng = np.random.default_rng(seed)
    P = scale * (env.P_max / env.N) * (1.0 + 0.1 * rng.standard_normal((env.K, env.N)))
    return env.project_budget(P)


def run_pga_sumrate(env: PowerControlEnv, iters: int = 200, step: float = 0.002, seed: int = 0):
    """
    Projected Gradient Ascent:
      max sum_rate(P) s.t. per-user power budgets
    """
    P = _init_power(env, seed=seed, scale=0.01)
    hist = {"sum_rate": [], "min_rate": [], "viol": []}

    hist["sum_rate"].append(env.sum_rate(P))
    hist["min_rate"].append(env.min_rate(P))
    hist["viol"].append(env.qos_violation(P))

    for _ in range(int(iters)):
        grad = env.grad_sum_rate(P)
        P = env.project_budget(P + step * grad)
        hist["sum_rate"].append(env.sum_rate(P))
        hist["min_rate"].append(env.min_rate(P))
        hist["viol"].append(env.qos_violation(P))
    return P, hist


def run_penalty_pga(env: PowerControlEnv, iters: int = 250, step: float = 0.002, rho: float = 80.0, seed: int = 0):
    """
    Penalty-based PGA for QoS:
      max sum_rate(P) - rho * sum_k [g_k(P)]_+^2
    """
    P = _init_power(env, seed=seed, scale=0.01)
    hist = {"sum_rate": [], "min_rate": [], "viol": []}

    hist["sum_rate"].append(env.sum_rate(P))
    hist["min_rate"].append(env.min_rate(P))
    hist["viol"].append(env.qos_violation(P))

    for _ in range(int(iters)):
        gval = env.g_constraints(P)     # (K,)
        gp = np.maximum(gval, 0.0)

        grad = env.grad_sum_rate(P)

        if np.any(gp > 0):
            # objective includes -rho * sum gp^2
            # gradient adds -2 rho gp_k ∇g_k, sum over k
            grad_g = -env.grad_rates_all(P)  # (K,K,N) = ∇g_k
            grad_pen = np.einsum("k,kij->ij", -2.0 * rho * gp, grad_g)  # (K,N)
            grad = grad + grad_pen

        P = env.project_budget(P + step * grad)

        hist["sum_rate"].append(env.sum_rate(P))
        hist["min_rate"].append(env.min_rate(P))
        hist["viol"].append(env.qos_violation(P))
    return P, hist


def run_primal_dual(env: PowerControlEnv, iters: int = 350, step_p: float = 0.01, step_lam: float = 0.2, seed: int = 0):
    """
    Primal-dual gradient for QoS:
      max sum_rate(P) s.t. g_k(P)<=0 and budget
    L(P,lam) = sum_rate(P) - sum_k lam_k g_k(P)

    Updates:
      P <- proj( P + step_p * (∇sum_rate - sum_k lam_k ∇g_k) )
      lam <- [lam + step_lam * g(P)]_+
    """
    P = _init_power(env, seed=seed, scale=0.01)
    lam = np.zeros(env.K, dtype=np.float64)

    hist = {"sum_rate": [], "min_rate": [], "viol": [], "lam_norm": []}

    hist["sum_rate"].append(env.sum_rate(P))
    hist["min_rate"].append(env.min_rate(P))
    hist["viol"].append(env.qos_violation(P))
    hist["lam_norm"].append(float(np.linalg.norm(lam)))

    for _ in range(int(iters)):
        grad = env.grad_sum_rate(P)

        grad_g = -env.grad_rates_all(P)               # (K,K,N)
        grad_l = np.einsum("k,kij->ij", lam, grad_g)  # (K,N)
        grad = grad - grad_l

        P = env.project_budget(P + step_p * grad)

        gval = env.g_constraints(P)
        lam = np.maximum(lam + step_lam * gval, 0.0)

        hist["sum_rate"].append(env.sum_rate(P))
        hist["min_rate"].append(env.min_rate(P))
        hist["viol"].append(env.qos_violation(P))
        hist["lam_norm"].append(float(np.linalg.norm(lam)))

    return P, hist


def run_interference_pricing_sumrate(
    env: PowerControlEnv,
    iters: int = 150,
    eta: float = 0.6,
    seed: int = 0,
):
    """
    Interference Pricing baseline (no QoS):
      max sum_rate(P) s.t. per-user budget
    Uses ONLY power gains g.

    Price at Rx k:
      pi[k,n] = (1/ln2) * sig / ( q (q+sig) )
      sig = g[k,k,n] P[k,n]
      q   = sigma2 + sum_{j!=k} g[j,k,n] P[j,n]

    Tx i cost:
      c[i,n] = sum_{k != i} pi[k,n] * g[i,k,n]

    User-i best response:
      p_n(mu) = ( 1/((c_n + mu) ln2) - d_n/gii )_+
      where d_n = q[i,n], gii = g[i,i,n]
    """
    iters = max(int(iters), 0)
    eta = float(eta)
    if not (0.0 < eta <= 1.0):
        raise ValueError("eta must be in (0,1].")

    rng = np.random.default_rng(seed)
    K, N = env.K, env.N
    g = env.g
    sigma2 = env.noise_power
    ln2 = np.log(2.0)

    P = 0.01 * (env.P_max / env.N) * (1.0 + 0.1 * rng.standard_normal((K, N)))
    P = env.project_budget(P)

    hist = {"sum_rate": [], "min_rate": [], "viol": []}
    hist["sum_rate"].append(env.sum_rate(P))
    hist["min_rate"].append(env.min_rate(P))
    hist["viol"].append(env.qos_violation(P))

    for _ in range(iters):
        total_rx = sigma2 + np.einsum("jkn,jn->kn", g, P)  # (K,N)
        sig = g[np.arange(K), np.arange(K), :] * P         # (K,N)
        q = np.maximum(total_rx - sig, 1e-18)              # (K,N)

        pi = (1.0 / ln2) * (sig / np.maximum(q * (q + sig), 1e-18))  # (K,N)

        # Gauss-Seidel best responses
        for i in range(K):
            d_in = q[i, :]                                  # (N,)
            gii = np.maximum(g[i, i, :], 1e-18)             # (N,)

            c_in = np.einsum("kn,kn->n", pi, g[i, :, :])    # (N,)
            c_in = c_in - pi[i, :] * g[i, i, :]             # exclude k=i
            c_in = np.maximum(c_in, 0.0)

            p0 = 1.0 / ((c_in + 1e-18) * ln2) - d_in / gii
            p0 = np.maximum(p0, 0.0)

            if p0.sum() <= env.P_max + 1e-12:
                p_new = p0
            else:
                lo, hi = 0.0, 1.0
                for _ in range(80):
                    p_hi = np.maximum(1.0 / ((c_in + hi) * ln2) - d_in / gii, 0.0)
                    if p_hi.sum() <= env.P_max:
                        break
                    hi *= 2.0
                for _ in range(60):
                    mu = 0.5 * (lo + hi)
                    pm = np.maximum(1.0 / ((c_in + mu) * ln2) - d_in / gii, 0.0)
                    if pm.sum() > env.P_max:
                        lo = mu
                    else:
                        hi = mu
                p_new = np.maximum(1.0 / ((c_in + hi) * ln2) - d_in / gii, 0.0)

            P[i, :] = (1.0 - eta) * P[i, :] + eta * p_new

        P = env.project_budget(P)

        hist["sum_rate"].append(env.sum_rate(P))
        hist["min_rate"].append(env.min_rate(P))
        hist["viol"].append(env.qos_violation(P))

    return P, hist


# ============================================================
# Pure-power WMMSE reference (final only)
# ============================================================
def wmmse_power_reference_final(
    env: PowerControlEnv,
    max_iter: int = 300,
    tol: float = 1e-10,
    n_starts: int = 5,
    seed: int = 0,
    P_init: np.ndarray | None = None,
):
    """
    Pure-power WMMSE (no phase). Multi-start. Optionally include explicit init P_init.
    IMPORTANT FIX:
      - Do NOT project onto simplex inside iterations (breaks WMMSE monotonicity).
      - Enforce sum-power by bisection on mu per user; only do numerical clipping/scaling.
    Returns best final P under env.sum_rate(P).
    """
    K, N = env.K, env.N
    g = env.g
    sigma2 = env.noise_power
    rng = np.random.default_rng(seed)

    eps = 1e-18
    best_P = None
    best_sr = -np.inf

    # ---- build starts ----
    starts = []
    if P_init is not None:
        P0 = np.asarray(P_init, dtype=np.float64).copy()
        # only minimal sanitization: nonneg + per-user scaling if exceeds budget
        P0 = np.maximum(P0, 0.0)
        row_sum = P0.sum(axis=1, keepdims=True)
        scale = np.minimum(1.0, env.P_max / np.maximum(row_sum, eps))
        P0 = P0 * scale
        starts.append(P0)

    for _ in range(max(0, int(n_starts) - len(starts))):
        # random-ish init to break symmetry
        P0 = (env.P_max / env.N) * np.ones((K, N))
        P0 *= (1.0 + 0.05 * rng.standard_normal((K, N)))
        P0 = np.maximum(P0, 0.0)
        row_sum = P0.sum(axis=1, keepdims=True)
        scale = np.minimum(1.0, env.P_max / np.maximum(row_sum, eps))
        P0 = P0 * scale
        starts.append(P0)

    # ---- WMMSE loop for each start ----
    for P in starts:
        prev_sr = env.sum_rate(P)

        for _ in range(int(max_iter)):
            # I_total[k,n] = sigma2 + sum_j g[j,k,n] P[j,n]
            I_total = sigma2 + np.einsum("jkn,jn->kn", g, P)  # (K,N)

            gkk = g[np.arange(K), np.arange(K), :]            # (K,N)
            hkk = np.sqrt(np.maximum(gkk, eps))               # amplitude-equivalent

            v = np.sqrt(np.maximum(P, 0.0))                   # (K,N), real nonneg
            hv = hkk * v                                      # (K,N)

            # MMSE receiver u = (hkk v_k) / I_total
            u = hv / np.maximum(I_total, eps)

            # MSE: e = 1 - 2u(hv) + u^2 I_total
            e = 1.0 - 2.0 * u * hv + (u**2) * I_total
            e = np.maximum(e, 1e-12)
            w = 1.0 / e

            a = w * (u**2)                                    # (K,N)

            # denom0[k,n] = sum_i g[k,i,n] * a[i,n]
            denom0 = np.einsum("kin,in->kn", g, a)            # (K,N)

            # num[k,n] = w_k u_k hkk
            num = w * u * hkk                                 # (K,N)

            P_new = np.zeros_like(P)

            for k in range(K):
                d0 = np.maximum(denom0[k, :], eps)
                b = np.maximum(num[k, :], 0.0)               # should be nonneg anyway

                # candidate mu=0
                pk0 = (b / d0) ** 2
                s0 = pk0.sum()

                if s0 <= env.P_max + 1e-12:
                    pk = pk0
                else:
                    # bisection on mu >= 0: sum (b/(d0+mu))^2 = Pmax
                    lo, hi = 0.0, 1.0
                    for _ in range(80):
                        pk_hi = (b / (d0 + hi)) ** 2
                        if pk_hi.sum() <= env.P_max:
                            break
                        hi *= 2.0

                    for _ in range(60):
                        mid = 0.5 * (lo + hi)
                        pk_mid = (b / (d0 + mid)) ** 2
                        if pk_mid.sum() > env.P_max:
                            lo = mid
                        else:
                            hi = mid

                    pk = (b / (d0 + hi)) ** 2

                # numerical guard: clip and (if still slightly above) scale down
                pk = np.maximum(pk, 0.0)
                s = pk.sum()
                if s > env.P_max * (1.0 + 1e-10):
                    pk *= (env.P_max / max(s, eps))

                P_new[k, :] = pk

            P = P_new

            sr = env.sum_rate(P)
            if abs(sr - prev_sr) <= tol * max(1.0, abs(prev_sr)):
                break
            prev_sr = sr

        sr_final = env.sum_rate(P)
        if sr_final > best_sr:
            best_sr = sr_final
            best_P = P.copy()

    return best_P


# ============================================================
# DMC: FULL global copies per agent (X: KxKxN) + FULL gradients wrt global Z (KxN)
# ============================================================

class DMCInterface:
    def dmc_step(self, P: np.ndarray, Z: np.ndarray, multipliers: dict, env, **kwargs):
        raise NotImplementedError


class DMC_GlobalCopy_FullGradient(DMCInterface):
    """
    DMC with full global copy per agent:
      Z: (K,N)
      X[a,:,:]: (K,N)  => overall X: (K,K,N)
      Gamma[a,:,:]: (K,N) => overall Gamma: (K,K,N)
      Lam[a]: scalar (one QoS constraint per agent; can be extended)

    Uses env hooks:
      - env.grad_local_objective_full(a,Z)  -> (K,N)
      - env.local_constraint_value(a,Z)     -> scalar
      - env.local_constraint_grad_full(a,Z) -> (K,N)
      - env.project_budget(Z)               -> (K,N)
    """

    def __init__(
        self,
        rho: float = 100.0,
        theta: float = 0.05,
        beta: float = 100.0,
        eta: float = 0.0,
        lam_clip: float = 1e6,
        project_X_each_step: bool = True,
    ):
        self.rho = float(rho)
        self.theta = float(theta)
        self.beta = float(beta)
        self.eta = float(eta)
        self.lam_clip = float(lam_clip)
        self.project_X_each_step = bool(project_X_each_step)

    def _init_state_if_needed(self, Z: np.ndarray, multipliers: dict):
        K, N = Z.shape

        if "X" not in multipliers:
            multipliers["X"] = np.repeat(Z[None, :, :], K, axis=0)  # (K,K,N)
        if "Gamma" not in multipliers:
            multipliers["Gamma"] = np.zeros((K, K, N), dtype=np.float64)
        if "Lam" not in multipliers:
            multipliers["Lam"] = np.zeros((K,), dtype=np.float64)
        if "t" not in multipliers:
            multipliers["t"] = 0

        X = np.asarray(multipliers["X"], dtype=np.float64)
        Gm = np.asarray(multipliers["Gamma"], dtype=np.float64)
        Lm = np.asarray(multipliers["Lam"], dtype=np.float64)

        if X.shape != (K, K, N):
            X = np.repeat(Z[None, :, :], K, axis=0)
        if Gm.shape != (K, K, N):
            Gm = np.zeros((K, K, N), dtype=np.float64)
        if Lm.shape != (K,):
            Lm = np.zeros((K,), dtype=np.float64)

        multipliers["X"] = X
        multipliers["Gamma"] = Gm
        multipliers["Lam"] = Lm
        return multipliers

    def dmc_step(self, P: np.ndarray, Z: np.ndarray, multipliers: dict, env, **kwargs):
        Z = np.asarray(Z, dtype=np.float64)
        K, N = Z.shape

        multipliers = self._init_state_if_needed(Z, multipliers)
        X = multipliers["X"]          # (K,K,N)
        Gamma = multipliers["Gamma"]  # (K,K,N)
        Lam = multipliers["Lam"]      # (K,)
        t = int(multipliers.get("t", 0))

        rho = float(kwargs.get("rho", self.rho))
        theta = float(kwargs.get("theta", self.theta))
        beta = float(kwargs.get("beta", self.beta))
        eta = float(kwargs.get("eta", self.eta))

        # ---- Step 0: consensus update for Z ----
        agg = np.sum(rho * X + Gamma, axis=0)          # (K,N)
        Z_next = (beta * Z + agg) / (K * rho + beta)   # (K,N)
        Z_next = env.project_budget(Z_next)

        # ---- Step 1: local updates (FULL gradient wrt global Z) ----
        X_next = np.empty_like(X)
        Gamma_next = np.empty_like(Gamma)
        Lam_next = Lam.copy()

        for a in range(K):
            grad_f = env.grad_local_objective_full(a, Z_next)      # (K,N)
            g_val = env.local_constraint_value(a, Z_next)          # scalar
            grad_g = env.local_constraint_grad_full(a, Z_next)     # (K,N)

            dx = grad_f + Lam[a] * grad_g                          # (K,N)

            Xa = Z_next - (dx + Gamma[a, :, :]) / max(rho, 1e-12)  # (K,N)
            if self.project_X_each_step:
                Xa = env.project_budget(Xa)

            lam_a = Lam[a] + (theta * g_val) / (1.0 + theta * eta)
            lam_a = np.clip(max(lam_a, 0.0), 0.0, self.lam_clip)
            Lam_next[a] = lam_a

            Gamma_next[a, :, :] = Gamma[a, :, :] + rho * (Xa - Z_next)
            X_next[a, :, :] = Xa

        multipliers["X"] = X_next
        multipliers["Gamma"] = Gamma_next
        multipliers["Lam"] = Lam_next
        multipliers["t"] = t + 1

        # For compatibility output, return mean copy
        P_next = np.mean(X_next, axis=0)  # (K,N)
        return P_next, Z_next, multipliers


def run_dmc(env: PowerControlEnv, dmc: DMCInterface, iters: int = 200, **kwargs):
    """
    Run DMC. History logged on consensus variable Z.
    """
    K, N = env.K, env.N
    Z = 0.01 * (env.P_max / env.N) * np.ones((K, N), dtype=np.float64)
    Z = env.project_budget(Z)

    P = Z.copy()
    multipliers = {}

    hist = {"sum_rate": [], "min_rate": [], "viol": []}
    hist["sum_rate"].append(env.sum_rate(Z))
    hist["min_rate"].append(env.min_rate(Z))
    hist["viol"].append(env.qos_violation(Z))

    for _ in range(int(iters)):
        P, Z, multipliers = dmc.dmc_step(P=P, Z=Z, multipliers=multipliers, env=env, **kwargs)

        # safety
        Z = env.project_budget(Z)
        P = env.project_budget(P)

        hist["sum_rate"].append(env.sum_rate(Z))
        hist["min_rate"].append(env.min_rate(Z))
        hist["viol"].append(env.qos_violation(Z))

    return Z, hist


# ============================================================
# Plotting (IEEE double-column friendly)
# ============================================================

def set_ieee_plot_style():
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.linewidth": 1.0,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    })


def plot_histories_ieee(env: PowerControlEnv, histories: dict, wmmse_P: np.ndarray | None, out_dir: str, prefix: str):
    os.makedirs(out_dir, exist_ok=True)

    wmmse_sr = env.sum_rate(wmmse_P) if wmmse_P is not None else None
    wmmse_minr = env.min_rate(wmmse_P) if wmmse_P is not None else None
    wmmse_viol = env.qos_violation(wmmse_P) if wmmse_P is not None else None

    fig_w, fig_h = 3.5, 2.4

    # 1) Sum-rate
    plt.figure(figsize=(fig_w, fig_h))
    for name, hist in histories.items():
        plt.plot(np.asarray(hist["sum_rate"], float), label=name)
    if wmmse_sr is not None:
        plt.axhline(wmmse_sr, linestyle="--", label="WMMSE (final ref.)")
    plt.xlabel("Iteration")
    plt.ylabel("Sum-rate (bps/Hz)")
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_sumrate.pdf"))
    plt.savefig(os.path.join(out_dir, f"{prefix}_sumrate.png"))
    plt.close()

    # 2) QoS violation (log scale)
    plt.figure(figsize=(fig_w, fig_h))
    for name, hist in histories.items():
        y = np.maximum(np.asarray(hist["viol"], float), 1e-18)
        plt.plot(y, label=name)
    if wmmse_viol is not None:
        plt.axhline(max(wmmse_viol, 1e-18), linestyle="--", label="WMMSE (final ref.)")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Mean QoS viol. (log)")
    plt.grid(True, which="both")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_violation.pdf"))
    plt.savefig(os.path.join(out_dir, f"{prefix}_violation.png"))
    plt.close()

    # 3) Min-rate
    plt.figure(figsize=(fig_w, fig_h))
    for name, hist in histories.items():
        plt.plot(np.asarray(hist["min_rate"], float), label=name)
    if wmmse_minr is not None:
        plt.axhline(wmmse_minr, linestyle="--", label="WMMSE (final ref.)")
    plt.axhline(env.R_min, linestyle=":", label=r"$R_{\min}$")
    plt.xlabel("Iteration")
    plt.ylabel("Min user rate (bps/Hz)")
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_minrate.pdf"))
    plt.savefig(os.path.join(out_dir, f"{prefix}_minrate.png"))
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    set_ieee_plot_style()

    env = PowerControlEnv(
        K=20,
        N=8,
        noise_power=1e-9,     # W
        P_max=1.0,            # W (per user total)
        R_min=20.0,            # bps/Hz
        seed=1,
        direct_boost_amp=2.0,
    )

    # Sanity: uniform allocation per user
    P_uniform = (env.P_max / env.N) * np.ones((env.K, env.N))
    print("[Sanity] Uniform: sum_rate=", env.sum_rate(P_uniform),
          " min_rate=", env.min_rate(P_uniform),
          " viol=", env.qos_violation(P_uniform))

    # Baselines
    P_pga, hist_pga = run_pga_sumrate(env, iters=200, step=0.002, seed=0)
    P_pen, hist_pen = run_penalty_pga(env, iters=250, step=0.002, rho=80.0, seed=0)
    P_pd,  hist_pd  = run_primal_dual(env, iters=350, step_p=0.01, step_lam=0.2, seed=0)

    # Interference pricing baseline (no QoS)
    P_ip, hist_ip = run_interference_pricing_sumrate(env, iters=150, eta=0.6, seed=0)

    # Pure-power WMMSE reference (final only)
    P_wmmse = wmmse_power_reference_final(env, max_iter=500, tol=1e-10, n_starts=5, seed=0, P_init=P_ip)


    # DMC (FULL gradient, FULL copy) — you can tune rho/theta/beta
    dmc = DMC_GlobalCopy_FullGradient(rho=100.0, theta=0.05, beta=100.0, eta=0.0, project_X_each_step=True)
    P_dmc, hist_dmc = run_dmc(env, dmc, iters=200, rho=100.0, theta=0.05, beta=100.0, eta=0.0)

    histories = {
        "PGA": hist_pga,
        "Penalty-PGA": hist_pen,
        "Primal-Dual": hist_pd,
        "Interf-Pricing": hist_ip,
        "DMC (full-copy)": hist_dmc,
    }

    out_dir = "figs_ieee"
    prefix = f"K{env.K}_N{env.N}_Rmin{env.R_min}"
    plot_histories_ieee(env, histories, wmmse_P=P_wmmse, out_dir=out_dir, prefix=prefix)

    # Table-style final numbers
    def summarize(name, P):
        return (name, env.sum_rate(P), env.min_rate(P), env.qos_violation(P), float(np.sum(P)))

    results = [
        summarize("Uniform", P_uniform),
        summarize("PGA", P_pga),
        summarize("Penalty-PGA", P_pen),
        summarize("Primal-Dual", P_pd),
        summarize("Interf-Pricing", P_ip),
        summarize("DMC", P_dmc),
        summarize("WMMSE (ref.)", P_wmmse),
    ]

    print("\n=== Final values (for paper table) ===")
    for (name, sr, minr, viol, tp) in results:
        print(f"{name:>14} | SR={sr:8.4f} | minR={minr:7.4f} | viol={viol:9.2e} | Pow={tp:7.4f}")

    print(f"\nSaved figures to: {out_dir}/")
    print("Note: WMMSE is PURE-POWER and used as FINAL reference line only (QoS not enforced).")

    print("median gkk =", np.median(env.g[np.arange(env.K), np.arange(env.K), :]))
    print("median cross =", np.median(env.g[np.arange(env.K)[:, None], np.arange(env.K)[None, :], :]))
    print("noise =", env.noise_power)


if __name__ == "__main__":
    main()
