import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ============================================================
# Part 0: Smooth nonconvex sparsity constraint (global)
#   g_sparse(w) = sum_j (1 - exp(-alpha * w_j^2)) - tau <= 0
# ============================================================

def sparse_constraint_value_grad(w: np.ndarray,
                                 alpha: float,
                                 tau: float) -> Tuple[float, np.ndarray]:
    """
    Smooth nonconvex sparsity constraint:
        g_sparse(w) = sum_j (1 - exp(-alpha * w_j^2)) - tau <= 0

    Returns:
        g_val: scalar
        grad_w: same shape as w
    """
    exp_term = np.exp(-alpha * (w ** 2))
    g_val = np.sum(1.0 - exp_term) - tau
    grad_w = 2.0 * alpha * w * exp_term
    return g_val, grad_w


# ============================================================
# Part 1: Robust SVM 分布式数据生成器 (保持你的结构 + 加 meta)
# ============================================================

def generate_distributed_robust_svm_data(
    n_samples,
    n_features,
    n_nodes,
    seed=42,
    easy_ratio=0.6,      # 大 margin 样本比例
    border_ratio=0.3,    # 近 margin 样本比例
    hard_ratio=None,     # 难样本比例，默认补齐
    margin_easy=3.0,     # easy 样本的 margin 大小
    margin_border=1.2,   # border 样本的 margin 大小（略大于 1）
    margin_hard=0.5,     # hard 样本朝“错误方向”的 margin
    noise_orth=0.2,      # 在 u 正交空间的噪声幅度
    sigma_noise=0.1,     # 协方差：Sigma_i = sigma_noise^2 I
    delta=0.1            # robust 半径：unc = sqrt(delta/(1-delta))
):
    """
    生成“强特征 + 有少量违约样本”的分布式 Robust SVM 数据集。

    特点：
      - 存在一个明显的 w_true，使得大部分样本在 y w^T x 上 margin 很大；
      - 少量 border/hard 样本，即使用最优 w 也需要 xi > 0；
      - 适合测试：w* 非 0，且 xi 在最优解处是“活跃”的。
    """
    np.random.seed(seed)

    if hard_ratio is None:
        hard_ratio = 1.0 - easy_ratio - border_ratio
    assert abs(easy_ratio + border_ratio + hard_ratio - 1.0) < 1e-8, "ratio 之和必须为 1"

    print(f"[Strong+Slack] Generating data: {n_samples} samples, "
          f"{n_features} features, {n_nodes} nodes.")
    print(f"[Strong+Slack] ratios: easy={easy_ratio}, border={border_ratio}, hard={hard_ratio}")
    print(f"[Strong+Slack] margins: easy={margin_easy}, border={margin_border}, hard={margin_hard}")
    print(f"[Strong+Slack] sigma_noise={sigma_noise}, delta={delta}")

    # 1. 真实权重 w_true（单位向量）
    w_true = np.random.randn(n_features)
    w_true /= np.linalg.norm(w_true)
    u = w_true

    # 2. 决定每类样本的数量
    n_easy = int(round(easy_ratio * n_samples))
    n_border = int(round(border_ratio * n_samples))
    n_hard = n_samples - n_easy - n_border

    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples)

    idx = 0

    # 辅助函数：在 u 的正交空间上生成一个单位向量
    def sample_orth_unit():
        z = np.random.randn(n_features)
        proj = np.dot(z, u) * u
        v = z - proj
        nrm = np.linalg.norm(v)
        if nrm < 1e-8:
            z = np.random.randn(n_features)
            proj = np.dot(z, u) * u
            v = z - proj
            nrm = np.linalg.norm(v)
        return v / nrm

    # 2.1 easy 样本：大 margin，基本不需要 xi
    for _ in range(n_easy):
        y_i = np.random.choice([-1.0, 1.0])
        y[idx] = y_i
        noise_vec = noise_orth * sample_orth_unit()
        x_i = y_i * margin_easy * u + noise_vec
        X[idx, :] = x_i
        idx += 1

    # 2.2 border 样本：margin ≈ 1 附近，有些需要小的 xi
    for _ in range(n_border):
        y_i = np.random.choice([-1.0, 1.0])
        y[idx] = y_i
        noise_vec = noise_orth * sample_orth_unit()
        x_i = y_i * margin_border * u + noise_vec
        X[idx, :] = x_i
        idx += 1

    # 2.3 hard 样本：朝错误方向，必然需要 xi > 0
    for _ in range(n_hard):
        y_i = np.random.choice([-1.0, 1.0])
        y[idx] = y_i
        noise_vec = noise_orth * sample_orth_unit()
        x_i = - y_i * margin_hard * u + noise_vec   # 故意放错边
        X[idx, :] = x_i
        idx += 1

    # 3. 协方差矩阵：各向同性小噪声，Sigma_i = sigma_noise^2 I
    Sqrt_Sigmas = []
    Sigmas = []
    for _ in range(n_samples):
        Sigma = (sigma_noise ** 2) * np.eye(n_features)
        Sqrt_Sigma = sigma_noise * np.eye(n_features)
        Sigmas.append(Sigma)
        Sqrt_Sigmas.append(Sqrt_Sigma)

    # 4. 不确定性常数
    unc_const = np.sqrt(delta / (1.0 - delta))

    # 5. 分发到各个节点
    node_data_list = []
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    split_indices = np.array_split(indices, n_nodes)

    for idx_list in split_indices:
        node_data_list.append({
            'X': X[idx_list],
            'y': y[idx_list],
            'Sigmas': [Sigmas[i] for i in idx_list],
            'Sqrt_Sigmas': [Sqrt_Sigmas[i] for i in idx_list],
            'n_samples': len(idx_list)
        })

    meta = {
        "w_true": w_true,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_nodes": n_nodes
    }

    return node_data_list, unc_const, meta


# ============================================================
# Part 2: DistributedNode (保持你的接口)
# ============================================================

class DistributedNode:
    def __init__(self, node_id, data, C_svm, unc_const, n_features, total_nodes):
        self.id = node_id
        self.X = data['X']               # 本地样本特征 (n_local, d)
        self.y = data['y']               # 本地标签 (n_local,)
        self.Sigmas = data['Sigmas']     # 协方差矩阵
        self.Sqrt_Sigmas = data['Sqrt_Sigmas']  # 协方差平方根
        self.n_local = data['n_samples'] # 本地样本数
        self.d = n_features              # 特征维度 (全局变量 w 的维度)
        self.total_nodes = total_nodes   # 总节点数，用于分摊全局正则项
        
        self.C = C_svm         # 惩罚系数 C
        self.unc = unc_const   # Robust 常数

    # ---------------- 1. 目标函数 ----------------
    # Global Obj: 0.5 * ||w||^2 + C * sum(xi)
    # Local Obj:  (0.5 / N_nodes) * ||w||^2 + C * sum(xi_local)

    def func(self, w, xi):
        reg_term = (0.5 * np.sum(w**2)) / self.total_nodes
        loss_term = self.C * np.sum(xi)
        return reg_term + loss_term

    def grad_w(self, w, xi):
        return w / self.total_nodes

    def grad_xi(self, w, xi):
        return np.full(self.n_local, self.C)

    # ---------------- 2. Robust 约束 g <= 0 ----------------
    # y_i * w^T x_i >= 1 - xi_i + unc * ||S_i w||
    # 等价：
    # g_i(w,xi) = 1 - xi_i + unc * ||S_i w|| - y_i w^T x_i <= 0

    def constraints(self, w, xi):
        g_vals = np.zeros(self.n_local)
        for j in range(self.n_local):
            S = self.Sqrt_Sigmas[j]
            Sw = S @ w
            norm_sw = np.linalg.norm(Sw)
            linear_part = self.y[j] * np.dot(w, self.X[j])
            g_vals[j] = 1.0 - xi[j] + self.unc * norm_sw - linear_part
        return g_vals

    def jacobian_w(self, w, xi):
        jac_w = np.zeros((self.n_local, self.d))
        epsilon = 1e-8
        for j in range(self.n_local):
            S = self.Sqrt_Sigmas[j]
            Sw = S @ w
            norm_sw = np.linalg.norm(Sw)
            if norm_sw < epsilon:
                grad_soc = np.zeros(self.d)
            else:
                grad_soc = (S.T @ Sw) / norm_sw
            jac_w[j, :] = self.unc * grad_soc - self.y[j] * self.X[j]
        return jac_w

    def jacobian_xi(self, w, xi):
        return -1.0 * np.eye(self.n_local)

    def project_xi(self, xi):
        return np.maximum(xi, 0.0)


# ============================================================
# Part 3: 中心化求解器 + 稀疏约束 + 准确率接口
# ============================================================

@dataclass
class ReferenceCentralizedSolverWithSparse:
    """
    集中式求解器：
      min 0.5*||w||^2 + C * sum_i xi_i
      s.t.   robust constraints  g_k(w,xi_k) <= 0
             sparse constraint   g_sparse(w) <= 0 (smooth nonconvex)
             xi_i >= 0
    """
    nodes: List[DistributedNode]
    n_features: int
    alpha_sparse: float = 5.0
    tau_sparse: float = 4.0

    def __post_init__(self):
        self.d = self.n_features
        self.n_total_xi = sum(node.n_local for node in self.nodes)
        self.total_dim = self.d + self.n_total_xi

        # xi 在大向量里的切片
        self.xi_slices = []
        start = self.d
        for node in self.nodes:
            end = start + node.n_local
            self.xi_slices.append(slice(start, end))
            start = end

    def _unpack(self, x_full):
        w = x_full[:self.d]
        xi_list = [x_full[s] for s in self.xi_slices]
        return w, xi_list

    # -------- 目标函数 --------
    def objective(self, x_full):
        w, xi_list = self._unpack(x_full)
        total_loss = 0.0
        for node, xi in zip(self.nodes, xi_list):
            total_loss += node.func(w, xi)
        return total_loss

    def gradient(self, x_full):
        w, xi_list = self._unpack(x_full)
        grad_w_accum = np.zeros(self.d)
        grad_xi_all = []
        for node, xi in zip(self.nodes, xi_list):
            grad_w_accum += node.grad_w(w, xi)
            grad_xi_all.append(node.grad_xi(w, xi))
        return np.concatenate([grad_w_accum] + grad_xi_all)

    # -------- Robust 约束：所有样本拼成一个向量 --------
    def constraints_func(self, x_full):
        w, xi_list = self._unpack(x_full)
        all_cons = []
        for node, xi in zip(self.nodes, xi_list):
            all_cons.append(node.constraints(w, xi))
        return np.concatenate(all_cons)

    def constraints_jacobian(self, x_full):
        w, xi_list = self._unpack(x_full)
        jac_rows = []
        for i, (node, xi) in enumerate(zip(self.nodes, xi_list)):
            jac_w_local = node.jacobian_w(w, xi)    # (n_i, d)
            jac_xi_local = node.jacobian_xi(w, xi)  # (n_i, n_i)
            row_parts = [jac_w_local]
            for j, other in enumerate(self.nodes):
                if i == j:
                    row_parts.append(jac_xi_local)
                else:
                    row_parts.append(np.zeros((node.n_local, other.n_local)))
            jac_rows.append(np.hstack(row_parts))
        return np.vstack(jac_rows)

    # -------- 稀疏非凸约束：g_sparse(w) <= 0 --------
    def sparse_constraint_func(self, x_full):
        w, _ = self._unpack(x_full)
        g_val, _ = sparse_constraint_value_grad(w, self.alpha_sparse, self.tau_sparse)
        return np.array([g_val])

    def sparse_constraint_jacobian(self, x_full):
        w, _ = self._unpack(x_full)
        _, grad_w = sparse_constraint_value_grad(w, self.alpha_sparse, self.tau_sparse)
        grad_full = np.zeros(self.total_dim)
        grad_full[:self.d] = grad_w
        return grad_full[None, :]

    # -------- 集中式求解 --------
    def solve(self,
              x0: Optional[np.ndarray] = None,
              maxiter: int = 200,
              gtol: float = 1e-6):
        print(f"Solving centralized problem with {self.total_dim} variables...")

        if x0 is None:
            x0 = np.zeros(self.total_dim)
            x0[self.d:] = 0.1  # 初始 xi > 0

        # Robust constraints
        cons_robust = NonlinearConstraint(
            self.constraints_func,
            -np.inf,
            0.0,
            jac=self.constraints_jacobian
        )

        # Sparse nonconvex constraint
        cons_sparse = NonlinearConstraint(
            self.sparse_constraint_func,
            -np.inf,
            0.0,
            jac=self.sparse_constraint_jacobian
        )

        # Bounds: w 无界，xi >= 0
        bounds = [(None, None)] * self.d + [(0.0, None)] * self.n_total_xi

        res = minimize(
            self.objective,
            x0,
            method='trust-constr',
            jac=self.gradient,
            constraints=[cons_robust, cons_sparse],
            bounds=bounds,
            options={
                'verbose': 2,
                'maxiter': maxiter,
                'gtol': gtol,
                'xtol': 1e-6,
                'barrier_tol': 1e-6
            }
        )

        w_opt, xi_list_opt = self._unpack(res.x)
        return w_opt, xi_list_opt, res

    # -------- 准确率评估 --------
    def evaluate_accuracy(self, w: np.ndarray) -> float:
        X_all = np.vstack([node.X for node in self.nodes])
        y_all = np.concatenate([node.y for node in self.nodes])
        scores = X_all @ w
        y_pred = np.where(scores >= 0, 1.0, -1.0)
        return (y_pred == y_all).mean()


# ============================================================
# Part 4: 简单 main 示例（可选，调试用）
# ============================================================

if __name__ == "__main__":
    # 缩小一点规模做个验证
    N_SAMPLES = 300
    N_FEATURES = 10
    N_NODES = 3
    C_SVM = 1.0

    data_list, unc_const, meta = generate_distributed_robust_svm_data(
        N_SAMPLES, N_FEATURES, N_NODES, seed=123
    )

    nodes = []
    for i in range(N_NODES):
        nodes.append(DistributedNode(
            node_id=i,
            data=data_list[i],
            C_svm=C_SVM,
            unc_const=unc_const,
            n_features=N_FEATURES,
            total_nodes=N_NODES
        ))

    solver = ReferenceCentralizedSolverWithSparse(
        nodes=nodes,
        n_features=N_FEATURES,
        alpha_sparse=5.0,
        tau_sparse=5.0   # 可以调小/调大观察约束是否激活
    )

    w_star, xi_list_star, res = solver.solve(maxiter=100)

    print("\n" + "=" * 40)
    print("CENTRALIZED SOLUTION (WITH SPARSE CONSTRAINT)")
    print("=" * 40)
    print("Optimal objective:", res.fun)
    print("w_star (first 5 dims):", w_star[:5])

    acc = solver.evaluate_accuracy(w_star)
    print("Train accuracy:", acc)

    g_sparse_val, _ = sparse_constraint_value_grad(
        w_star, solver.alpha_sparse, solver.tau_sparse
    )
    print("g_sparse(w_star) =", g_sparse_val)
