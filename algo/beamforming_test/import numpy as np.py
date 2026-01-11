import numpy as np
import matplotlib.pyplot as plt
import time


# ============================================================
# 1) Environment (SISO, K user pairs, N subchannels)
# g[j,k,n] = |h_{j->k,n}|^2
# ============================================================
class PowerControlEnv:
    def __init__(
        self,
        K=20,
        N=8,
        P_max=1.0,
        noise_power=1e-9,
        area_side=1000.0,
        pathloss_exp=3.5,
        direct_boost=2.0,
        seed=0,
    ):
        self.K = int(K)
        self.N = int(N)
        self.P_max = float(P_max)
        self.noise_power = float(noise_power)

        rng = np.random.default_rng(seed)

        # Tx positions
        tx = rng.uniform(0.0, area_side, size=(K, 2))
        # Rx positions near its Tx (pairing)
        rx = tx + rng.normal(scale=0.03 * area_side, size=(K, 2))
        rx = np.clip(rx, 0.0, area_side)

        # distance d[j,k] : Tx j -> Rx k
        d = np.zeros((K, K), dtype=np.float64)
        for j in range(K):
            diff = rx - tx[j]
            d[j, :] = np.sqrt((diff**2).sum(axis=1)) + 1.0  # avoid zero

        # pathloss
        L = d ** (-pathloss_exp)  # (K,K)

        # frequency selective Rayleigh fading
        fading = (rng.normal(size=(K, K, N)) + 1j * rng.normal(size=(K, K, N))) / np.sqrt(2.0)
        h = np.sqrt(L)[:, :, None] * fading

        # boost direct links
        for k in range(K):
            h[k, k, :] *= direct_boost

        self.h = h
        self.g = np.abs(h) ** 2  # (K,K,N)

    # ---- projection per user: {p>=0, sum p <= Pmax} ----
    def project_budget(self, P: np.ndarray) -> np.ndarray:
        P = np.maximum(P, 0.0)
        out = np.zeros_like(P)
        for k in range(self.K):
            row = P[k].copy()
            s = row.sum()
            if s <= self.P_max:
                out[k] = row
            else:
                # project onto simplex sum = Pmax
                u = np.sort(row)[::-1]
                cssv = np.cumsum(u)
                idx = np.arange(1, self.N + 1)
                cond = u - (cssv - self.P_max) / idx > 0
                rho = np.where(cond)[0][-1]
                theta = (cssv[rho] - self.P_max) / (rho + 1.0)
                out[k] = np.maximum(row - theta, 0.0)
        return out

    # ---- SINR + rates ----
    def sinr(self, P: np.ndarray) -> np.ndarray:
        # total_rx[k,n] = noise + sum_j g[j,k,n] * P[j,n]
        total_rx = self.noise_power + np.einsum("jkn,jn->kn", self.g, P)
        sig = self.g[np.arange(self.K), np.arange(self.K), :] * P  # (K,N)
        interf = np.maximum(total_rx - sig, 1e-18)
        return sig / interf

    def rates_per_user(self, P: np.ndarray) -> np.ndarray:
        s = self.sinr(P)
        return (np.log1p(np.maximum(s, 0.0)) / np.log(2.0)).sum(axis=1)

    def sum_rate(self, P: np.ndarray) -> float:
        return float(np.sum(self.rates_per_user(P)))


# ============================================================
# 2) Interference Pricing (no QoS)
# ============================================================
def run_interference_pricing(env: PowerControlEnv, iters=150, tol=1e-7, eta=1.0, seed=0, verbose=False):
    """
    Interference Pricing for sum-rate maximization (TIN model), no QoS constraints.

    Notation:
      g[j,k,n] = Tx j -> Rx k gain on channel n.
      P[j,n]   = Tx j power on channel n.
      q[k,n]   = noise + sum_{j!=k} g[j,k,n] P[j,n]

    Price at Rx k:
      pi[k,n] = - dR_{k,n} / d q[k,n]
              = (1/ln2) * (g[k,k,n] P[k,n]) / ( q[k,n] * (q[k,n] + g[k,k,n] P[k,n]) )

    For Tx i, cost on channel n:
      c[i,n] = sum_{k != i} pi[k,n] * g[i,k,n]

    User i update:
      maximize sum_n log2(1 + g[i,i,n] p_n / d_n) - sum_n c_n p_n
      s.t. p_n>=0, sum_n p_n <= Pmax
    Closed form with bisection on mu:
      p_n(mu) = ( 1/((c_n + mu) ln2) - d_n / g[i,i,n] )_+
    """

    rng = np.random.default_rng(seed)
    K, N = env.K, env.N
    g = env.g
    sigma2 = env.noise_power
    ln2 = np.log(2.0)

    # init: small random powers to break symmetry
    P = 0.01 * (env.P_max / N) * (1.0 + 0.1 * rng.standard_normal((K, N)))
    P = env.project_budget(P)

    sr_hist = []
    t_hist = []
    t0 = time.perf_counter()
    prev_sr = -np.inf

    for t in range(iters):
        # total_rx[k,n] = noise + sum_j g[j,k,n] P[j,n]
        total_rx = sigma2 + np.einsum("jkn,jn->kn", g, P)  # (K,N)
        sig = g[np.arange(K), np.arange(K), :] * P         # (K,N)
        q = np.maximum(total_rx - sig, 1e-18)              # (K,N) = noise + sum_{j!=k}

        # price at each Rx k
        # pi[k,n] = (1/ln2) * sig / (q * (q + sig))
        denom = np.maximum(q * (q + sig), 1e-18)
        pi = (1.0 / ln2) * (sig / denom)                   # (K,N)

        # Gauss-Seidel user updates
        for i in range(K):
            d_in = q[i, :]                                 # (N,) noise + interference at Rx i
            gii = np.maximum(g[i, i, :], 1e-18)            # (N,)

            # cost c_in[n] = sum_{k != i} pi[k,n] * g[i,k,n]
            # g[i,:,:] has shape (K,N) : (k,n)
            c_in = np.einsum("kn,kn->n", pi, g[i, :, :])   # sum over k -> (N,)
            c_in = c_in - pi[i, :] * g[i, i, :]            # exclude k=i
            c_in = np.maximum(c_in, 0.0)

            # candidate with mu=0
            p0 = 1.0 / ((c_in + 1e-18) * ln2) - d_in / gii
            p0 = np.maximum(p0, 0.0)

            if p0.sum() <= env.P_max + 1e-12:
                p_new = p0
            else:
                # bisection on mu to satisfy sum p = Pmax
                lo, hi = 0.0, 1.0
                # find a hi large enough so that sum p(hi) <= Pmax
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

            # optional damping for stability
            if eta < 1.0:
                P[i, :] = (1.0 - eta) * P[i, :] + eta * p_new
            else:
                P[i, :] = p_new

        # safety projection (numerical)
        P = env.project_budget(P)

        sr = env.sum_rate(P)
        sr_hist.append(sr)
        t_hist.append(time.perf_counter() - t0)

        if verbose and (t % 10 == 0 or t == iters - 1):
            print(f"iter {t:3d}: sum-rate = {sr:.6f}")

        if abs(sr - prev_sr) <= tol * max(1.0, abs(prev_sr)):
            break
        prev_sr = sr

    elapsed = time.perf_counter() - t0
    return P, np.array(sr_hist), np.array(t_hist), elapsed


# ============================================================
# 3) Demo
# ============================================================
if __name__ == "__main__":
    env = PowerControlEnv(K=20, N=8, P_max=1.0, noise_power=1e-9, seed=1, direct_boost=2.0)

    P_ip, sr_hist, t_hist, elapsed = run_interference_pricing(
        env, iters=150, tol=1e-7, eta=0.6, seed=0, verbose=True
    )

    print(f"\nFinal sum-rate = {env.sum_rate(P_ip):.6f} bps/Hz, elapsed = {elapsed:.3f}s")

    plt.figure(figsize=(3.6, 2.4))
    plt.plot(sr_hist)
    plt.xlabel("Iteration")
    plt.ylabel("Sum-rate (bps/Hz)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(3.6, 2.4))
    plt.plot(t_hist, sr_hist)
    plt.xlabel("Wall-clock time (s)")
    plt.ylabel("Sum-rate (bps/Hz)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
