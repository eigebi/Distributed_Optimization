# power_control_bench_method4_mc.py
# Method-4 QoS calibration (SNR-proxy) + baselines + DMC (fixed/adaptive)
# Figures you want:
#   Fig-A (single instance, T=100): 1 figure with 2 subplots
#       (a) Sum-rate vs iteration (+ WMMSE final as dashed hline)
#       (b) QoS violation vs iteration (mean(max(g,0))) in log-scale
#           + WMMSE final violation as dashed hline
#       Curves: Pricing, DMC-fixed, DMC-adaptive (and WMMSE dashed reference lines)
#
#   Fig-B (scale with Monte-Carlo, each K run S=20 seeds): 1 figure with 2 subplots
#       (a) Mean(SR / SR_WMMSE) vs K
#       (b) Mean( mean(max(g,0)) / R_min ) vs K  (and WMMSE's own ratio as dashed)
#       x-axis uses integer ticks (no decimals)
#
# Pure-power (channel gains only), no phase.

import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Utils
# ============================================================

def safe_log2_1p(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(x, 0.0)) / np.log(2.0)

def proj_l1_ball_nonneg(v: np.ndarray, z: float) -> np.ndarray:
    """
    Euclidean projection onto {x >= 0, sum(x) <= z}.
    If sum(max(v,0)) <= z -> max(v,0).
    Else -> projection onto simplex {x>=0, sum x = z}.
    """
    v = np.asarray(v, dtype=np.float64)
    v_pos = np.maximum(v, 0.0)
    s = v_pos.sum()
    if s <= z:
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

def row_project_budget(P: np.ndarray, P_max: float) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    K, N = P.shape
    out = np.zeros_like(P)
    for k in range(K):
        out[k, :] = proj_l1_ball_nonneg(P[k, :], P_max)
    return out


# ============================================================
# Environment (pure power gains)
# ============================================================

class PowerControlEnv:
    """
    Pure power-domain K-user IC on N subchannels, TIN.
      SINR_{k,n} = g_{kk,n} P_{k,n} / (sigma2 + sum_{j!=k} g_{jk,n} P_{j,n})
      R_k = sum_n log2(1 + SINR_{k,n})

    QoS constraint:
      g_k(P) = R_min - R_k(P) <= 0

    Violation metric you want:
      viol(P) = mean_k max(g_k(P), 0)   (NOT squared)
    """

    def __init__(
        self,
        K: int,
        N: int,
        noise_power: float,
        P_max: float,
        seed: int = 0,
        area_side: float = 500.0,
        pathloss_exp: float = 3.5,
        direct_boost_amp: float = 2.0,
        edge_frac: float = 0.2,
        edge_distance_boost: float = 2.0,
    ):
        self.K = int(K)
        self.N = int(N)
        self.noise_power = float(noise_power)
        self.P_max = float(P_max)

        self.area_side = float(area_side)
        self.pathloss_exp = float(pathloss_exp)
        self.direct_boost_amp = float(direct_boost_amp)

        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)

        # positions
        tx = rng.uniform(0, self.area_side, size=(self.K, 2))
        rx = tx + rng.normal(scale=self.area_side * 0.15, size=(self.K, 2))
        rx = np.clip(rx, 0.0, self.area_side)

        # choose edge users (fixed fraction)
        num_edge = int(np.round(edge_frac * self.K))
        edge_idx = rng.choice(self.K, size=max(num_edge, 1), replace=False) if self.K > 0 else np.array([], int)
        self.edge_mask = np.zeros(self.K, dtype=bool)
        self.edge_mask[edge_idx] = True

        # distances d_{j->k}
        d = np.zeros((self.K, self.K), dtype=np.float64)
        for j in range(self.K):
            diff = rx - tx[j]
            d[j, :] = np.sqrt((diff**2).sum(axis=1)) + 1.0

        # make direct links of edge users worse
        for k in range(self.K):
            if self.edge_mask[k]:
                d[k, k] *= float(edge_distance_boost)

        # pathloss
        L = d ** (-self.pathloss_exp)  # (K,K)

        # fading power ~ Exp(1)
        fading_pow = rng.exponential(scale=1.0, size=(self.K, self.K, self.N))
        self.g = L[:, :, None] * fading_pow

        # boost direct links
        if self.direct_boost_amp != 1.0:
            boost_pow = self.direct_boost_amp ** 2
            for k in range(self.K):
                self.g[k, k, :] *= boost_pow

        # QoS target set by calibration
        self.R_min = 0.0

    def project_budget(self, P: np.ndarray) -> np.ndarray:
        assert P.shape == (self.K, self.N)
        return row_project_budget(P, self.P_max)

    def sinr(self, P: np.ndarray) -> np.ndarray:
        P = np.asarray(P, dtype=np.float64)
        assert P.shape == (self.K, self.N)
        I_total = self.noise_power + np.einsum("jkn,jn->kn", self.g, P)  # (K,N)
        sig = self.g[np.arange(self.K), np.arange(self.K), :] * P        # (K,N)
        denom = np.maximum(I_total - sig, 1e-18)
        return sig / denom

    def rates_per_user(self, P: np.ndarray) -> np.ndarray:
        return safe_log2_1p(self.sinr(P)).sum(axis=1)

    def sum_rate(self, P: np.ndarray) -> float:
        return float(np.sum(self.rates_per_user(P)))

    def min_rate(self, P: np.ndarray) -> float:
        return float(np.min(self.rates_per_user(P)))

    def g_constraints(self, P: np.ndarray) -> np.ndarray:
        return self.R_min - self.rates_per_user(P)  # (K,)

    def viol(self, P: np.ndarray) -> float:
        g = self.g_constraints(P)
        return float(np.mean(np.maximum(g, 0.0)))

    def satisfy_ratio(self, P: np.ndarray) -> float:
        g = self.g_constraints(P)
        return float(np.mean(g <= 0.0))

    # gradients: dR_dP[k,i,n] = ∂R_k / ∂P_{i,n}
    def grad_rates_all(self, P: np.ndarray) -> np.ndarray:
        P = np.asarray(P, dtype=np.float64)
        K, N = self.K, self.N
        assert P.shape == (K, N)

        I_total = self.noise_power + np.einsum("jkn,jn->kn", self.g, P)  # (K,N)
        gkk = self.g[np.arange(K), np.arange(K), :]                      # (K,N)
        sig = gkk * P                                                    # (K,N)
        D = np.maximum(I_total - sig, 1e-18)
        sinr = sig / D
        coeff = (1.0 / np.log(2.0)) * (1.0 / np.maximum(1.0 + sinr, 1e-18))  # (K,N)

        dR = np.zeros((K, K, N), dtype=np.float64)

        # own derivative
        for k in range(K):
            d_sinr_d_pk = gkk[k, :] / np.maximum(D[k, :], 1e-18)
            dR[k, k, :] = coeff[k, :] * d_sinr_d_pk

        # cross derivative
        for k in range(K):
            for i in range(K):
                if i == k:
                    continue
                gik = self.g[i, k, :]
                d_sinr_d_pi = -(gkk[k, :] * P[k, :]) * gik / np.maximum(D[k, :]**2, 1e-18)
                dR[k, i, :] = coeff[k, :] * d_sinr_d_pi

        return dR

    # full-gradient hooks for DMC:
    # local objective: f_a(Z) = -R_a(Z)
    def grad_local_objective_full(self, a: int, Z: np.ndarray) -> np.ndarray:
        dR = self.grad_rates_all(Z)   # (K,K,N)
        return -dR[a, :, :]           # (K,N)

    # QoS constraint: g_a(Z) = R_min - R_a(Z)
    def local_constraint_value(self, a: int, Z: np.ndarray) -> float:
        return float(self.g_constraints(Z)[a])

    def local_constraint_grad_full(self, a: int, Z: np.ndarray) -> np.ndarray:
        dR = self.grad_rates_all(Z)
        return -dR[a, :, :]           # (K,N)


# ============================================================
# Method-4 QoS calibration (SNR proxy)
# ============================================================

def calibrate_Rmin_method4(env: PowerControlEnv, alpha: float = 0.55, q_user: float = 0.12):
    """
    Proxy:
      SNR_{k,n} = gkk * (Pmax/N) / sigma2
      Rtilde_k = sum_n log2(1 + SNR_{k,n})
    Set:
      R_min = alpha * Quantile_{q_user}({Rtilde_k})
    """
    K, N = env.K, env.N
    gkk = env.g[np.arange(K), np.arange(K), :]  # (K,N)
    P_unif = env.P_max / N
    snr = gkk * P_unif / max(env.noise_power, 1e-18)
    Rtilde = safe_log2_1p(snr).sum(axis=1)      # (K,)
    base = float(np.quantile(Rtilde, q_user))
    env.R_min = float(alpha * base)
    return env.R_min


# ============================================================
# Baselines: Interference Pricing + WMMSE(final)
# ============================================================

def run_interference_pricing_sumrate(env: PowerControlEnv, iters: int = 100, eta: float = 0.6, seed: int = 0):
    """
    Interference Pricing for sum-rate (no QoS enforcement).
    History logged for sum-rate / min-rate / viol.
    """
    iters = int(max(iters, 0))
    if not (0.0 < eta <= 1.0):
        raise ValueError("eta must be in (0,1].")

    rng = np.random.default_rng(seed)
    K, N = env.K, env.N
    g = env.g
    sigma2 = env.noise_power
    ln2 = np.log(2.0)

    P = 0.01 * (env.P_max / N) * (1.0 + 0.1 * rng.standard_normal((K, N)))
    P = env.project_budget(P)

    hist = {"sum_rate": [], "min_rate": [], "viol": []}

    def log_point(Pcur):
        hist["sum_rate"].append(env.sum_rate(Pcur))
        hist["min_rate"].append(env.min_rate(Pcur))
        hist["viol"].append(env.viol(Pcur))

    log_point(P)

    for _ in range(iters):
        total_rx = sigma2 + np.einsum("jkn,jn->kn", g, P)   # (K,N)
        sig = g[np.arange(K), np.arange(K), :] * P          # (K,N)
        q = np.maximum(total_rx - sig, 1e-18)

        pi = (1.0 / ln2) * (sig / np.maximum(q * (q + sig), 1e-18))  # (K,N)

        for i in range(K):
            d_in = q[i, :]
            gii = np.maximum(g[i, i, :], 1e-18)

            c_in = np.einsum("kn,kn->n", pi, g[i, :, :])
            c_in = c_in - pi[i, :] * g[i, i, :]
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
                    mid = 0.5 * (lo + hi)
                    p_mid = np.maximum(1.0 / ((c_in + mid) * ln2) - d_in / gii, 0.0)
                    if p_mid.sum() > env.P_max:
                        lo = mid
                    else:
                        hi = mid
                p_new = np.maximum(1.0 / ((c_in + hi) * ln2) - d_in / gii, 0.0)

            P[i, :] = (1.0 - eta) * P[i, :] + eta * p_new

        P = env.project_budget(P)
        log_point(P)

    return P, hist


def wmmse_power_reference_final(env: PowerControlEnv, max_iter: int = 400, tol: float = 1e-10, n_starts: int = 3, seed: int = 0, P_init=None):
    """
    Pure-power WMMSE reference for sum-rate (no QoS).
    Final solution ONLY (use as dashed horizontal reference).
    """
    K, N = env.K, env.N
    g = env.g
    sigma2 = env.noise_power
    rng = np.random.default_rng(seed)
    eps = 1e-18

    best_P, best_sr = None, -np.inf
    starts = []

    if P_init is not None:
        P0 = np.maximum(np.asarray(P_init, float), 0.0)
        rs = P0.sum(axis=1, keepdims=True)
        P0 *= np.minimum(1.0, env.P_max / np.maximum(rs, eps))
        starts.append(P0)

    for _ in range(max(0, int(n_starts) - len(starts))):
        P0 = (env.P_max / N) * (1.0 + 0.05 * rng.standard_normal((K, N)))
        P0 = np.maximum(P0, 0.0)
        rs = P0.sum(axis=1, keepdims=True)
        P0 *= np.minimum(1.0, env.P_max / np.maximum(rs, eps))
        starts.append(P0)

    for P in starts:
        prev_sr = env.sum_rate(P)
        for _ in range(int(max_iter)):
            I_total = sigma2 + np.einsum("jkn,jn->kn", g, P)

            gkk = g[np.arange(K), np.arange(K), :]
            hkk = np.sqrt(np.maximum(gkk, eps))
            v = np.sqrt(np.maximum(P, 0.0))
            hv = hkk * v

            u = hv / np.maximum(I_total, eps)
            e = 1.0 - 2.0 * u * hv + (u**2) * I_total
            e = np.maximum(e, 1e-12)
            w = 1.0 / e

            a = w * (u**2)
            denom0 = np.einsum("kin,in->kn", g, a)     # (K,N)
            num = w * u * hkk                          # (K,N)

            P_new = np.zeros_like(P)
            for k in range(K):
                d0 = np.maximum(denom0[k, :], eps)
                b = np.maximum(num[k, :], 0.0)
                pk0 = (b / d0) ** 2
                if pk0.sum() <= env.P_max + 1e-12:
                    pk = pk0
                else:
                    lo, hi = 0.0, 1.0
                    for _ in range(80):
                        if ((b / (d0 + hi)) ** 2).sum() <= env.P_max:
                            break
                        hi *= 2.0
                    for _ in range(60):
                        mid = 0.5 * (lo + hi)
                        if ((b / (d0 + mid)) ** 2).sum() > env.P_max:
                            lo = mid
                        else:
                            hi = mid
                    pk = (b / (d0 + hi)) ** 2
                s = pk.sum()
                if s > env.P_max * (1.0 + 1e-10):
                    pk *= env.P_max / max(s, eps)
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
# DMC (full copy per agent) : fixed vs adaptive schedules
# ============================================================

class DMC_GlobalCopy_FullGradient:
    """
    Z: (K,N)
    X[a,:,:]: (K,N) => X: (K,K,N)
    Gamma[a,:,:]: (K,N) => Gamma: (K,K,N)
    Lam[a]: scalar QoS multiplier for agent a
    """

    def __init__(self, rho=10.0, theta=0.5, beta=50.0, eta=0.0, lam_clip=1e6, project_X_each_step=True):
        self.rho = float(rho)
        self.theta = float(theta)
        self.beta = float(beta)
        self.eta = float(eta)
        self.lam_clip = float(lam_clip)
        self.project_X_each_step = bool(project_X_each_step)

    def init_state(self, Z):
        K, N = Z.shape
        X = np.repeat(Z[None, :, :], K, axis=0)
        Gamma = np.zeros((K, K, N), dtype=np.float64)
        Lam = np.zeros((K,), dtype=np.float64)
        return {"X": X, "Gamma": Gamma, "Lam": Lam, "t": 0}

    def step(self, env: PowerControlEnv, Z: np.ndarray, state: dict, rho=None, theta=None, beta=None, eta=None):
        Z = np.asarray(Z, dtype=np.float64)
        K, N = Z.shape

        X = state["X"]
        Gamma = state["Gamma"]
        Lam = state["Lam"]

        rho = float(self.rho if rho is None else rho)
        theta = float(self.theta if theta is None else theta)
        beta = float(self.beta if beta is None else beta)
        eta = float(self.eta if eta is None else eta)

        # ---- consensus update ----
        agg = np.sum(rho * X + Gamma, axis=0)          # (K,N)
        Z_next = (beta * Z + agg) / (K * rho + beta)
        Z_next = env.project_budget(Z_next)

        # ---- local updates ----
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

            # dual ascent on constraint (project to R_+)
            denom = 1.0 + theta * eta
            lam_a = (Lam[a] + theta * g_val) / denom
            lam_a = np.clip(max(lam_a, 0.0), 0.0, self.lam_clip)
            Lam_next[a] = lam_a

            Gamma_next[a, :, :] = Gamma[a, :, :] + rho * (Xa - Z_next)
            X_next[a, :, :] = Xa

        state["X"] = X_next
        state["Gamma"] = Gamma_next
        state["Lam"] = Lam_next
        state["t"] = int(state["t"]) + 1

        return Z_next, state


def run_dmc(env: PowerControlEnv, dmc: DMC_GlobalCopy_FullGradient, iters: int, mode: str, sched: dict, seed: int = 0):
    rng = np.random.default_rng(seed)
    K, N = env.K, env.N
    Z = (env.P_max / N) * (1.0 + 0.01 * rng.standard_normal((K, N)))
    Z = np.maximum(Z, 0.0)
    Z = env.project_budget(Z)

    state = dmc.init_state(Z)

    hist = {"sum_rate": [], "min_rate": [], "viol": []}

    def log_point(Zcur):
        hist["sum_rate"].append(env.sum_rate(Zcur))
        hist["min_rate"].append(env.min_rate(Zcur))
        hist["viol"].append(env.viol(Zcur))

    log_point(Z)

    for t in range(1, int(iters) + 1):
        if mode == "fixed":
            Z, state = dmc.step(
                env, Z, state,
                rho=sched.get("rho", None),
                theta=sched.get("theta", None),
                beta=sched.get("beta", None),
                eta=sched.get("eta", None),
            )
        elif mode == "adaptive":
            rho = float(sched.get("rho", dmc.rho))
            theta = float(sched.get("theta", dmc.theta))

            eta_pow = float(sched.get("eta_pow", 0.5))
            eta_scale = float(sched.get("eta_scale", 1.0))
            eta_t = eta_scale * (t ** (-eta_pow))

            beta0 = float(sched.get("beta0", dmc.beta))
            beta_mode = str(sched.get("beta_mode", "1+1/eta"))
            if beta_mode == "1+1/eta":
                beta_t = beta0 * (1.0 + 1.0 / max(eta_t, 1e-12))
            elif beta_mode == "1/eta^2":
                beta_t = beta0 * (1.0 + 1.0 / max(eta_t, 1e-12) ** 2)
            else:
                beta_t = beta0

            Z, state = dmc.step(env, Z, state, rho=rho, theta=theta, beta=beta_t, eta=eta_t)
        else:
            raise ValueError("mode must be 'fixed' or 'adaptive'.")

        Z = env.project_budget(Z)
        log_point(Z)

    return Z, hist


# ============================================================
# Plotting (clear colors, 1 figure with 2 subplots)
# ============================================================

def set_ieee_plot_style():
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 2.0,
        "figure.dpi": 160,
        "savefig.dpi": 300,
    })

def plot_iter_two_panels(histories: dict, wmmse_sr: float, wmmse_viol: float, out_dir: str, prefix: str):
    """
    1 figure, 2 subplots:
      left:  sum-rate vs iteration (+ wmmse_sr dashed line)
      right: viol vs iteration (log) (+ wmmse_viol dashed line)
    """
    os.makedirs(out_dir, exist_ok=True)

    # clear style set:
    style = {
        "Interf-Pricing": dict(color="tab:blue",  linestyle="-"),
        "DMC-fixed":      dict(color="tab:orange",linestyle="-"),
        "DMC-adaptive":   dict(color="tab:green", linestyle="-"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))

    # (a) sum-rate
    ax = axes[0]
    for name, hist in histories.items():
        ax.plot(np.asarray(hist["sum_rate"], float), label=name, **style.get(name, {}))
    ax.axhline(wmmse_sr, linestyle="--", color="tab:gray", label="WMMSE (final)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Sum-rate (bps/Hz)")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="best")

    # (b) violation
    ax = axes[1]
    for name, hist in histories.items():
        y = np.maximum(np.asarray(hist["viol"], float), 1e-18)
        ax.plot(y, label=name, **style.get(name, {}))
    ax.axhline(max(wmmse_viol, 1e-18), linestyle="--", color="tab:gray", label="WMMSE (final)")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Mean $\max(g,0)$ (log)")
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_iter_two_panels.pdf"))
    fig.savefig(os.path.join(out_dir, f"{prefix}_iter_two_panels.png"))
    plt.close(fig)

def plot_scale_two_panels(K_list, sr_ratio_dict, viol_ratio_dict, out_dir: str, prefix: str):
    """
    1 figure, 2 subplots:
      left:  mean SR/SR_WMMSE vs K (WMMSE is implicitly 1)
      right: mean viol/Rmin vs K (plot also WMMSE viol/Rmin as dashed)
    x-axis integer ticks only.
    """
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))

    # left panel: SR ratios
    ax = axes[0]
    for name, ys in sr_ratio_dict.items():
        ax.plot(K_list, ys, marker="o",
                color={"Interf-Pricing":"tab:blue","DMC-fixed":"tab:orange","DMC-adaptive":"tab:green"}.get(name,"tab:purple"),
                linestyle="-",
                label=name)
    ax.axhline(1.0, linestyle="--", color="tab:gray", label="WMMSE (=1)")
    ax.set_xlabel("K (number of users)")
    ax.set_ylabel("Mean SR / SR(WMMSE)")
    ax.set_xticks(list(map(int, K_list)))
    ax.grid(True, alpha=0.4)
    ax.legend(loc="best")

    # right panel: viol ratios
    ax = axes[1]
    for name, ys in viol_ratio_dict.items():
        if name == "WMMSE":
            continue
        ax.plot(K_list, ys, marker="o",
                color={"Interf-Pricing":"tab:blue","DMC-fixed":"tab:orange","DMC-adaptive":"tab:green"}.get(name,"tab:purple"),
                linestyle="-",
                label=name)
    # WMMSE reference viol ratio
    if "WMMSE" in viol_ratio_dict:
        ax.plot(K_list, viol_ratio_dict["WMMSE"], linestyle="--", color="tab:gray", marker="x", label="WMMSE (viol/Rmin)")
    ax.set_yscale("log")
    ax.set_xlabel("K (number of users)")
    ax.set_ylabel(r"Mean $\max(g,0)$ / $R_{\min}$ (log)")
    ax.set_xticks(list(map(int, K_list)))
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_scale_two_panels.pdf"))
    fig.savefig(os.path.join(out_dir, f"{prefix}_scale_two_panels.png"))
    plt.close(fig)


# ============================================================
# Main experiment
# ============================================================

def main():
    set_ieee_plot_style()

    # ------------------------
    # Single-instance behavior (T=100)
    # ------------------------
    T = 100

    K = 20
    N = 4
    noise = 1e-8
    P_max = 1.0

    env = PowerControlEnv(
        K=K, N=N, noise_power=noise, P_max=P_max, seed=1,
        direct_boost_amp=2.0,
        edge_frac=0.2,
        edge_distance_boost=2.0
    )
    calibrate_Rmin_method4(env, alpha=0.55, q_user=0.12)

    # Pricing baseline
    P_ip, hist_ip = run_interference_pricing_sumrate(env, iters=T, eta=0.6, seed=0)

    # WMMSE final reference (no QoS)
    P_wmmse = wmmse_power_reference_final(env, max_iter=400, tol=1e-10, n_starts=3, seed=0, P_init=P_ip)
    wmmse_sr = env.sum_rate(P_wmmse)
    wmmse_viol = env.viol(P_wmmse)

    # DMC fixed vs adaptive
    dmc = DMC_GlobalCopy_FullGradient(rho=10.0, theta=0.5, beta=50.0, eta=0.0, project_X_each_step=True)

    fixed_sched = {"rho": 10.0, "theta": 0.5, "beta": 50.0, "eta": 0.0}
    Z_fixed, hist_fixed = run_dmc(env, dmc, iters=T, mode="fixed", sched=fixed_sched, seed=123)

    adapt_sched = {"rho": 10.0, "theta": 0.5, "beta": 50.0, "eta_pow": 0.5, "eta_scale": 1.0, "beta_mode": "1+1/eta"}
    Z_adapt, hist_adapt = run_dmc(env, dmc, iters=T, mode="adaptive", sched=adapt_sched, seed=123)

    histories = {
        "Interf-Pricing": hist_ip,
        "DMC-fixed": hist_fixed,
        "DMC-adaptive": hist_adapt,
    }

    out_dir = "figs_method4"
    prefix = f"method4_K{K}_N{N}_Rmin{env.R_min:.2f}_T{T}"
    plot_iter_two_panels(histories, wmmse_sr=wmmse_sr, wmmse_viol=wmmse_viol, out_dir=out_dir, prefix=prefix)

    # ------------------------
    # Scale experiment: Monte-Carlo (each K run S=20)
    # Normalization:
    #   rate: SR / SR_WMMSE  (WMMSE=1)
    #   viol: mean(max(g,0)) / Rmin
    # ------------------------
    S = 10
    K_list = [10, 15, 20, 25, 30]

    # store mean ratios across seeds
    sr_ratio = {"Interf-Pricing": [], "DMC-fixed": [], "DMC-adaptive": []}
    viol_ratio = {"Interf-Pricing": [], "DMC-fixed": [], "DMC-adaptive": [], "WMMSE": []}

    # iteration counts for scale runs (keep moderate)
    T_ip = 100
    T_dmc = 100

    eps = 1e-12
    base_seed = 2026

    for Kk in K_list:
        ratios_sr = {k: [] for k in sr_ratio.keys()}
        ratios_vl = {k: [] for k in viol_ratio.keys()}

        for rep in range(S):
            seed = base_seed + 1000 * Kk + rep

            envk = PowerControlEnv(
                K=Kk, N=N, noise_power=noise, P_max=P_max, seed=seed,
                direct_boost_amp=2.0,
                edge_frac=0.2,
                edge_distance_boost=2.0
            )
            calibrate_Rmin_method4(envk, alpha=0.55, q_user=0.12)
            Rmin = envk.R_min

            # Pricing
            P_ip_k, _ = run_interference_pricing_sumrate(envk, iters=T_ip, eta=0.6, seed=0)
            sr_ip = envk.sum_rate(P_ip_k)
            vl_ip = envk.viol(P_ip_k)

            # WMMSE reference
            P_w_k = wmmse_power_reference_final(envk, max_iter=350, tol=1e-10, n_starts=2, seed=0, P_init=P_ip_k)
            sr_w = envk.sum_rate(P_w_k)
            vl_w = envk.viol(P_w_k)

            # DMC runs (same schedules as above; you can tweak if needed)
            dmc_k = DMC_GlobalCopy_FullGradient(rho=10.0, theta=0.5, beta=50.0, eta=0.0, project_X_each_step=True)

            Z_fx_k, _ = run_dmc(envk, dmc_k, iters=T_dmc, mode="fixed", sched=fixed_sched, seed=123)
            sr_fx = envk.sum_rate(Z_fx_k)
            vl_fx = envk.viol(Z_fx_k)

            Z_ad_k, _ = run_dmc(envk, dmc_k, iters=T_dmc, mode="adaptive", sched=adapt_sched, seed=123)
            sr_ad = envk.sum_rate(Z_ad_k)
            vl_ad = envk.viol(Z_ad_k)

            # normalized ratios
            ratios_sr["Interf-Pricing"].append(sr_ip / (sr_w + eps))
            ratios_sr["DMC-fixed"].append(sr_fx / (sr_w + eps))
            ratios_sr["DMC-adaptive"].append(sr_ad / (sr_w + eps))


            vref = max(vl_w, 1e-12)

            ratios_vl["Interf-Pricing"].append(vl_ip / vref)
            ratios_vl["DMC-fixed"].append(vl_fx / vref)
            ratios_vl["DMC-adaptive"].append(vl_ad / vref)
            ratios_vl["WMMSE"].append(vl_w / vref) 
            '''
            ratios_vl["Interf-Pricing"].append(vl_ip / (Rmin + eps))
            ratios_vl["DMC-fixed"].append(vl_fx / (Rmin + eps))
            ratios_vl["DMC-adaptive"].append(vl_ad / (Rmin + eps))
            ratios_vl["WMMSE"].append(vl_w / (Rmin + eps))
            '''

        # mean over S runs
        for name in sr_ratio.keys():
            sr_ratio[name].append(float(np.mean(ratios_sr[name])))
        for name in viol_ratio.keys():
            viol_ratio[name].append(float(np.mean(ratios_vl[name])))

        print(f"[MC scale K={Kk}] "
              f"SR/IP={sr_ratio['Interf-Pricing'][-1]:.3f}, SR/fix={sr_ratio['DMC-fixed'][-1]:.3f}, SR/ad={sr_ratio['DMC-adaptive'][-1]:.3f} | "
              f"viol/IP={viol_ratio['Interf-Pricing'][-1]:.3e}, viol/fix={viol_ratio['DMC-fixed'][-1]:.3e}, viol/ad={viol_ratio['DMC-adaptive'][-1]:.3e}, viol/W={viol_ratio['WMMSE'][-1]:.3e}")

    prefix2 = f"method4_MCscale_N{N}_S{S}_T{T_dmc}"
    plot_scale_two_panels(K_list, sr_ratio, viol_ratio, out_dir=out_dir, prefix=prefix2)

    print(f"\nSaved figures to: {out_dir}/")
    print("  - *_iter_two_panels.(pdf|png)")
    print("  - *_scale_two_panels.(pdf|png)")


if __name__ == "__main__":
    main()
