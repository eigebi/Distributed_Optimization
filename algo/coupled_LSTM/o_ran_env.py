
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

# =====================
# Basic configuration
# =====================

@dataclass
class EnvCfg:
    area_size_m: Tuple[float, float] = (2000.0, 2000.0)
    fc_GHz: float = 3.5
    bandwidth_Hz: float = 20e6          # W_b for all BSs (can be per-BS if needed)
    noise_figure_dB: float = 7.0
    pathloss_exponent: float = 3.5
    shadowing_sigma_dB: float = 6.0
    min_distance_m: float = 10.0
    seed: int = 2025
    Pmax_W: float = 40.0                # BS max transmit power (W) default ~46 dBm
    temperature_K: float = 290.0

def thermal_noise_psd_w_per_hz(noise_figure_dB: float, T_kelvin: float = 290.0):
    """Return thermal noise power spectral density (W/Hz) with noise figure."""
    k_B = 1.380649e-23  # Boltzmann
    N0 = k_B * T_kelvin
    NF = 10.0 ** (noise_figure_dB / 10.0)
    return N0 * NF

def free_space_PL_dB(d_m: np.ndarray, fc_GHz: float):
    """Friis reference at 1 m with frequency fc (GHz)."""
    # FSPL(d) = (4*pi*d/lambda)^2  -> in dB: 32.45 + 20log10(fc_MHz) + 20log10(d_km)
    fc_MHz = fc_GHz * 1e3
    d_km = np.maximum(d_m, 1.0) / 1e3
    return 32.45 + 20.0 * np.log10(fc_MHz) + 20.0 * np.log10(d_km)

def pathloss_dB_log_distance(d_m: np.ndarray, fc_GHz: float, n: float, shadow_sigma_dB: float, rng: np.random.Generator):
    """Log-distance model with lognormal shadowing; PL(d0=1m) from FSPL, slope n."""
    d0 = 1.0
    PL0 = free_space_PL_dB(np.array([d0]), fc_GHz)[0]
    d = np.maximum(d_m, d0)
    PL = PL0 + 10.0 * n * np.log10(d / d0)
    if shadow_sigma_dB > 0:
        PL += rng.normal(0.0, shadow_sigma_dB, size=PL.shape)
    return PL

def deploy_bs_grid(B: int, area: Tuple[float, float]):
    """Place B base stations on a simple grid covering the area."""
    G = int(np.ceil(np.sqrt(B)))
    xs = np.linspace(0.1, 0.9, G)
    ys = np.linspace(0.1, 0.9, G)
    xv, yv = np.meshgrid(xs, ys)
    pts = np.stack([xv.ravel(), yv.ravel()], axis=1)[:B]
    w, h = area
    return np.column_stack([pts[:,0]*w, pts[:,1]*h])  # (B,2)

def uniform_ues(K: int, area: Tuple[float, float], rng: np.random.Generator):
    w, h = area
    return np.column_stack([rng.uniform(0,w,size=K), rng.uniform(0,h,size=K)])  # (K,2)

def nearest_bs_assign(ue_xy: np.ndarray, bs_xy: np.ndarray):
    """Assign each UE to its nearest BS (Voronoi)."""
    K, B = ue_xy.shape[0], bs_xy.shape[0]
    dx = ue_xy[:,None,0] - bs_xy[None,:,0]
    dy = ue_xy[:,None,1] - bs_xy[None,:,1]
    d2 = dx*dx + dy*dy
    b_u = d2.argmin(axis=1)
    return b_u

def channel_gains(cfg: EnvCfg, ue_xy: np.ndarray, bs_xy: np.ndarray, rng: np.random.Generator):
    """Return large-scale power gain matrix G[u,b]."""
    K, B = ue_xy.shape[0], bs_xy.shape[0]
    dx = ue_xy[:,None,0] - bs_xy[None,:,0]
    dy = ue_xy[:,None,1] - bs_xy[None,:,1]
    d = np.sqrt(dx*dx + dy*dy)
    d = np.maximum(d, cfg.min_distance_m)
    PL_dB = pathloss_dB_log_distance(d, cfg.fc_GHz, cfg.pathloss_exponent, cfg.shadowing_sigma_dB, rng)
    G = 10 ** (-PL_dB / 10.0)
    return G  # (K,B)

# =====================
# Formulation (rho_s global, eta_{s,b} per BS)
# =====================

class FormulationV2:
    """
    Model per the user's latest spec:
    - Power fractions: eta[s,b] with sum_s eta[s,b] <= 1 (for each b).
    - Spectrum fractions: rho[s] (global across BS) with sum_s rho[s] <= 1.
    - No inter-slice interference (slices are orthogonal in frequency).
    - Interference comes only from same slice, other BS.
    - Per-UE parameters (not optimized): theta[u] (power share within slice/BS), phi[u] (spectrum share).
    - SINR(u) = (theta_u * eta_{s(u),b(u)} * Pmax_{b(u)} * g_{b(u)->u})
                / ( sum_{b' != b(u)} eta_{s(u),b'} * Pmax_{b'} * g_{b'->u} + sigma2_{s(u),b(u)} )
      where sigma2_{s,b} = N0 * rho_s * W_b
    - Rate: R_u = phi_u * rho_{s(u)} * W_{b(u)} * log2(1 + SINR_u).
    """
    def __init__(self, B: int, K: int, S: int = 3,
                 cfg: Optional[EnvCfg] = None,
                 slice_probs: Optional[np.ndarray] = None,
                 alpha_eta: float = 1e-3,
                 alpha_rho: float = 1e-3,
                 eps: float = 1e-9,
                 seed: int = 2025):
        self.cfg = cfg or EnvCfg(seed=seed)
        self.rng = np.random.default_rng(self.cfg.seed)
        self.B, self.K, self.S = B, K, S
        # Network topology
        self.bs_xy = deploy_bs_grid(B, self.cfg.area_size_m)            # (B,2)
        self.ue_xy = uniform_ues(K, self.cfg.area_size_m, self.rng)     # (K,2)
        self.b_u = nearest_bs_assign(self.ue_xy, self.bs_xy)            # (K,)
        # Slices for UEs
        if slice_probs is None:
            slice_probs = np.ones(S) / S
        self.slice_probs = np.asarray(slice_probs, dtype=float) / np.sum(slice_probs)
        self.s_u = self.rng.choice(S, size=K, p=self.slice_probs)       # (K,)
        # Channel gains (large-scale)
        self.G = channel_gains(self.cfg, self.ue_xy, self.bs_xy, self.rng)   # (K,B)
        # Per-BS bandwidth and max power
        self.Wb = np.full(B, self.cfg.bandwidth_Hz, dtype=float)        # (B,)
        self.Pmax = np.full(B, self.cfg.Pmax_W, dtype=float)            # (B,)
        # Noise PSD (W/Hz)
        self.N0 = thermal_noise_psd_w_per_hz(self.cfg.noise_figure_dB, self.cfg.temperature_K)
        # Per-UE non-optimized parameters
        self.theta = self._init_theta_equal()                           # (K,)
        self.phi = self._init_phi_equal()                               # (K,)
        # Weights and QoS
        self.w_u = np.ones(K, dtype=float)
        self.Rmin_u = np.zeros(K, dtype=float)
        # Objective regularization
        self.alpha_eta = float(alpha_eta)
        self.alpha_rho = float(alpha_rho)
        self.eps = float(eps)

    # ---------- UE-level shares (non-optimized) ----------
    def _init_theta_equal(self):
        """Equal power shares per (b,s) group, normalized to sum<=1 in each group."""
        theta = np.zeros(self.K, dtype=float)
        for b in range(self.B):
            for s in range(self.S):
                idx = np.where((self.b_u==b) & (self.s_u==s))[0]
                if idx.size > 0:
                    theta[idx] = 1.0 / idx.size
        return theta

    def _init_phi_equal(self):
        """Equal spectrum shares per (b,s) group, normalized to sum<=1 in each group."""
        phi = np.zeros(self.K, dtype=float)
        for b in range(self.B):
            for s in range(self.S):
                idx = np.where((self.b_u==b) & (self.s_u==s))[0]
                if idx.size > 0:
                    phi[idx] = 1.0 / idx.size
        return phi

    def set_theta(self, theta_u: np.ndarray):
        assert theta_u.shape == (self.K,)
        self.theta = np.clip(theta_u, 0.0, 1.0)

    def set_phi(self, phi_u: np.ndarray):
        assert phi_u.shape == (self.K,)
        self.phi = np.clip(phi_u, 0.0, 1.0)

    # ---------- Decision variables packing ----------
    def var_shapes(self):
        """Return shapes for decision variables: rho[S], eta[S,B]"""
        return (self.S,), (self.S, self.B)

    def merge(self, rho: np.ndarray, eta: np.ndarray) -> np.ndarray:
        return np.concatenate([rho.ravel(), eta.ravel()])

    def split(self, x: np.ndarray):
        rho = x[:self.S].reshape(self.S)
        eta = x[self.S:].reshape(self.S, self.B)
        return rho, eta

    # ---------- Core computations ----------
    def sigma2_sb(self, rho: np.ndarray) -> np.ndarray:
        """Noise power per (s,b): sigma^2_{s,b} = N0 * rho_s * W_b"""
        return (self.N0 * rho[:,None] * self.Wb[None,:])  # (S,B)

    def sinr_per_ue(self, rho: np.ndarray, eta: np.ndarray):
        """Compute SINR_u per spec (no robust)."""
        K, B, S = self.K, self.B, self.S
        theta = self.theta
        b_u, s_u, G = self.b_u, self.s_u, self.G
        Pmax = self.Pmax
        sigma2 = self.sigma2_sb(rho)  # (S,B)

        # Build A_u = theta_u * Pmax_{b_u} * g_{b_u,u}
        A_u = theta * Pmax[b_u] * G[np.arange(K), b_u]   # (K,)

        # Numerator: eta_{s_u, b_u} * A_u
        num = eta[s_u, b_u] * A_u

        # Interference for each UE: sum over b' != b_u of eta_{s_u, b'} * Pmax_{b'} * g_{b',u}
        # We'll compute for each b, but we need per UE excluding serving b.
        I = np.zeros(K, dtype=float)
        for b in range(B):
            # contribution of BS b to all UEs
            contrib = eta[s_u, b] * Pmax[b] * G[:, b]
            I += contrib
        # subtract the serving BS contribution (which should not be in interference)
        I -= eta[s_u, b_u] * Pmax[b_u] * G[np.arange(K), b_u]

        # Denominator D = I + sigma2_{s_u, b_u}
        D = I + sigma2[s_u, b_u]
        D = np.maximum(D, 1e-15)  # numeric guard

        sinr = num / D
        return sinr, (A_u, I, D)

    def rate_per_ue(self, rho: np.ndarray, eta: np.ndarray):
        """R_u = phi_u * rho_{s(u)} * W_{b(u)} * log2(1 + sinr_u)."""
        sinr, aux = self.sinr_per_ue(rho, eta)
        phi, b_u, s_u = self.phi, self.b_u, self.s_u
        Wb = self.Wb
        Ru = phi * rho[s_u] * Wb[b_u] * np.log2(1.0 + sinr)
        return Ru, sinr, aux

    # ---------- Objective & gradients ----------
    def objective_and_grads(self, x: np.ndarray):
        """Return objective value and gradient wrt flattened x=[rho;eta].
        Objective: sum_u w_u * log(eps + R_u) - alpha_eta * sum_{s,b} eta_{s,b} - alpha_rho * sum_s rho_s
        """
        rho, eta = self.split(x)
        Ru, sinr, (A_u, I_u, D_u) = self.rate_per_ue(rho, eta)

        w = self.w_u
        eps = self.eps
        Wb = self.Wb
        b_u, s_u = self.b_u, self.s_u
        phi = self.phi
        Pmax = self.Pmax
        N0 = self.N0

        f = np.sum(w * np.log(eps + Ru)) - self.alpha_eta * np.sum(eta) - self.alpha_rho * np.sum(rho)

        # Precompute helpful terms
        dObj_dRu = w / (eps + Ru)                             # (K,)
        log_term = 1.0 / (np.log(2.0) * (1.0 + sinr))         # (K,)
        coeff = dObj_dRu * (phi * rho[s_u] * Wb[b_u]) * log_term  # multiplies d(sinr)
        # For rho derivative we also need the direct derivative of Ru wrt rho (linear factor)
        direct_rho_coeff = dObj_dRu * (phi * Wb[b_u]) * np.log2(1.0 + sinr)   # multiplies delta rho[s_u]

        # ---------- Grad wrt eta[s,b] ----------
        grad_eta = np.zeros((self.S, self.B), dtype=float)
        # Two cases: (1) b == b_u (numerator); (2) b != b_u (denominator via interference)
        # Case (1): d sinr / d eta[s_u, b_u] = A_u / D_u
        contrib_self = np.zeros_like(grad_eta)
        for u in range(self.K):
            s = s_u[u]; b = b_u[u]
            d_sinr = A_u[u] / D_u[u]
            grad_eta[s, b] += coeff[u] * d_sinr

        # Case (2): for each (s,b), for UEs with same slice s but served by other b_u != b:
        # d sinr / d eta[s, b] = -(eta[s,b_u] * A_u / D_u^2) * (Pmax[b] * G[u,b])
        for b in range(self.B):
            mask = (b_u != b)
            if not np.any(mask):
                continue
            # Only UEs whose slice equals s we will scatter-add
            u_idx = np.where(mask)[0]
            if u_idx.size == 0:
                continue
            # quantities per selected UEs
            s_sel = s_u[u_idx]
            dI = Pmax[b] * self.G[u_idx, b]     # (|U|,)
            d_sinr = -(eta[s_sel, b_u[u_idx]] * A_u[u_idx] / (D_u[u_idx]**2)) * dI
            for j, u in enumerate(u_idx):
                grad_eta[s_sel[j], b] += coeff[u] * d_sinr[j]

        # Regularizer
        grad_eta -= self.alpha_eta

        # ---------- Grad wrt rho[s] ----------
        grad_rho = np.zeros(self.S, dtype=float)
        # Two contributions:
        #  (i) direct: R_u has factor rho[s_u]
        # (ii) indirect via sigma2_{s,b_u} in D_u: d sinr / d rho[s_u] = -(eta[s_u,b_u]*A_u / D_u^2) * (N0 * Wb[b_u])
        for u in range(self.K):
            s = s_u[u]
            # (i) direct
            grad_rho[s] += direct_rho_coeff[u]
            # (ii) indirect
            d_sinr = -(eta[s, b_u[u]] * A_u[u] / (D_u[u]**2)) * (N0 * Wb[b_u[u]])
            grad_rho[s] += coeff[u] * d_sinr

        # Regularizer
        grad_rho -= self.alpha_rho

        # Pack gradients
        g = np.concatenate([grad_rho.ravel(), grad_eta.ravel()])
        return f, g

    # ---------- Constraints ----------
    def capacity_constraints(self, x: np.ndarray):
        """Return capacity constraints residuals >= 0:
           c_eta[b] = 1 - sum_s eta[s,b]  (for each b)
           c_rho    = 1 - sum_s rho[s]
        """
        rho, eta = self.split(x)
        c_eta = 1.0 - eta.sum(axis=0)        # (B,)
        c_rho = 1.0 - rho.sum()              # scalar
        return c_eta, c_rho

    def per_ue_rate_constraints(self, x: np.ndarray, use_lb: bool = False):
        """Return vector h_u = R_u - Rmin_u (>=0 required)."""
        rho, eta = self.split(x)
        Ru, sinr, _ = self.rate_per_ue(rho, eta)
        return Ru - self.Rmin_u

    # ---------- Helpers ----------
    def init_feasible(self):
        """Return a simple feasible x (rho, eta) using equal shares within active sets."""
        # Build active masks to avoid empty slices at BSs: here we keep all active.
        rho = np.full(self.S, 1.0 / self.S, dtype=float) * 0.9
        eta = np.full((self.S, self.B), 1.0 / self.S, dtype=float) * 0.9
        # small shrink to keep strictly feasible
        # normalize per-BS
        eta /= np.maximum(1.0, eta.sum(axis=0, keepdims=True))
        return self.merge(rho, eta)

    def snapshot(self):
        """Return a dictionary of key arrays for analysis/debug."""
        return dict(
            B=self.B, K=self.K, S=self.S,
            bs_xy=self.bs_xy, ue_xy=self.ue_xy,
            b_u=self.b_u, s_u=self.s_u,
            G=self.G, Wb=self.Wb, Pmax=self.Pmax, N0=self.N0,
            theta=self.theta, phi=self.phi,
        )

    def per_ue_constraints_and_jacobians(self, x: np.ndarray):
        """Return (h, Jrho, Jeta) where
           h[u]   = R_u - Rmin_u,
           Jrho   shape (K, S)    = d h / d rho_s,
           Jeta   shape (K, S, B) = d h / d eta_{s,b}.
        """
        rho, eta = self.split(x)
        Ru, sinr, aux = self.rate_per_ue(rho, eta)
        A_u, I_u, D_u = aux

        K, S, B = self.K, self.S, self.B
        b_u, s_u = self.b_u, self.s_u
        phi, Wb = self.phi, self.Wb
        Pmax, N0 = self.Pmax, self.N0

        ln2 = np.log(2.0)
        L_u = np.log1p(sinr) / ln2                  # log2(1+sinr)
        dL_dsinr = 1.0 / (ln2 * (1.0 + sinr))       # d/dsinr log2(1+sinr)

        # h = R - Rmin
        h = Ru - self.Rmin_u

        # ---- Jacobian wrt rho[S] ----
        Jrho = np.zeros((K, S), dtype=float)
        # direct term: dR/d rho[s_u] = phi*Wb * L_u
        direct = phi * Wb[b_u] * L_u
        # indirect via sigma^2_{s,b_u} = N0 * rho_s * Wb[b_u]:
        # d sinr / d rho[s_u] = - (eta[s_u,b_u]*A_u / D_u**2) * (N0 * Wb[b_u])
        d_sinr_drho = -( (eta[s_u, b_u] * A_u) / (D_u**2) ) * (N0 * Wb[b_u])
        indirect = (phi * rho[s_u] * Wb[b_u]) * dL_dsinr * d_sinr_drho
        Jrho[np.arange(K), s_u] = direct + indirect

        # ---- Jacobian wrt eta[S,B] ----
        Jeta = np.zeros((K, S, B), dtype=float)
        coef = (phi * rho[s_u] * Wb[b_u]) * dL_dsinr

        # (a) serving BS contribution
        d_sinr_self = A_u / D_u
        for u in range(K):
            Jeta[u, s_u[u], b_u[u]] += coef[u] * d_sinr_self[u]

        # (b) other BS contributions via interference
        base = -(eta[s_u, b_u] * A_u) / (D_u**2)  # (K,)
        for b in range(B):
            mask = (b_u != b)
            if not np.any(mask):
                continue
            dI = Pmax[b] * self.G[:, b]          # (K,)
            contrib = coef[mask] * (base[mask] * dI[mask])
            ss = s_u[mask]
            uu = np.where(mask)[0]
            for j, u in enumerate(uu):
                Jeta[u, ss[j], b] += contrib[j]

        return h, Jrho, Jeta

if __name__ == "__main__":
    # Quick self-test
    F = FormulationV2(B=15, K=300, S=3, alpha_eta=1, alpha_rho=1)
    x0 = F.init_feasible()
    f0, g0 = F.objective_and_grads(x0)
    c_eta, c_rho = F.capacity_constraints(x0)
    h_u = F.per_ue_rate_constraints(x0)
    print("Objective:", float(f0))
    print("||grad||:", float(np.linalg.norm(g0)))
    print("Cap. min margins -> eta:", float(c_eta.min()), " rho:", float(c_rho))
    print("Min UE margin:", float(h_u.min()))
