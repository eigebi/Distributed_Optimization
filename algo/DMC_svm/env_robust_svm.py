import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

# ==============================================================================
# Part 1: Robust SVM 分布式数据生成器
# Reference: Section 6 "Experimental Setup" 
# ==============================================================================



####################

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

    print(f"[Strong+Slack] Generating data: {n_samples} samples, {n_features} features, {n_nodes} nodes.")
    print(f"[Strong+Slack] ratios: easy={easy_ratio}, border={border_ratio}, hard={hard_ratio}")
    print(f"[Strong+Slack] margins: easy={margin_easy}, border={margin_border}, hard={margin_hard}")
    print(f"[Strong+Slack] sigma_noise={sigma_noise}, delta={delta}")

    # 1. 真实权重 w_true
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
            # 罕见退化，重新采一次
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

        # 故意把 x_i 放在“错的一边”
        noise_vec = noise_orth * sample_orth_unit()
        # 注意这里是负号：- y_i * margin_hard * u
        x_i = - y_i * margin_hard * u + noise_vec
        X[idx, :] = x_i
        idx += 1

    # 3. 协方差矩阵：各向同性小噪声
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

    return node_data_list, unc_const


# ==============================================================================
# Part 2: 分布式节点接口 (Decoupled Interface for DMC)
# 提供了目标函数、约束、梯度和雅可比矩阵
# ==============================================================================

class DistributedNode:
    def __init__(self, node_id, data, C_svm, unc_const, n_features, total_nodes):
        self.id = node_id
        self.X = data['X']               # 本地样本特征 (n_local, d)
        self.y = data['y']               # 本地标签 (n_local,)
        self.Sigmas = data['Sigmas']     # 协方差矩阵（已裁剪 eigenvalues）
        self.Sqrt_Sigmas = data['Sqrt_Sigmas'] # 用于计算约束值和梯度的平方根矩阵
        self.n_local = data['n_samples'] # 本地样本数
        self.d = n_features              # 特征维度 (全局变量 w 的维度)
        self.total_nodes = total_nodes   # 总节点数，用于分摊全局正则项
        
        self.C = C_svm         # 惩罚系数 c [cite: 2218]
        self.unc = unc_const   # 鲁棒常数

    # --------------------------------------------------------------------------
    # 1. 目标函数 (Objective Function)
    # Global Obj: 0.5 * ||w||^2 + C * sum(xi) [cite: 2219]
    # Local Obj:  (0.5 / N_nodes) * ||w||^2 + C * sum(xi_local)
    # --------------------------------------------------------------------------
    
    def func(self, w, xi):
        """
        计算本地目标函数值。
        w: 全局变量 (d,)
        xi: 本地变量 (n_local,)
        """
        # 将全局正则项 ||w||^2 均摊到每个节点
        reg_term = (0.5 * np.sum(w**2)) / self.total_nodes
        loss_term = self.C * np.sum(xi)
        return reg_term + loss_term

    def grad_w(self, w, xi):
        """
        计算目标函数关于 w 的梯度。
        grad = (1 / N_nodes) * w
        """
        return w / self.total_nodes

    def grad_xi(self, w, xi):
        """
        计算目标函数关于 xi 的梯度。
        grad = C
        """
        return np.full(self.n_local, self.C)

    # --------------------------------------------------------------------------
    # 2. 约束函数 (Constraints)
    # Robust SVM SOCP Constraint: 
    # y_i * w^T * x_i >= 1 - xi_i + unc * ||Sigma^0.5 w|| [cite: 2221]
    # Reformulated as g(x) <= 0:
    # 1 - xi_i + unc * ||Sigma^0.5 w|| - y_i * w^T * x_i <= 0
    # --------------------------------------------------------------------------

    def constraints(self, w, xi):
        """
        计算本地所有样本的不等式约束值 g(w, xi)。
        返回: (n_local,) 的数组，值 > 0 表示违反约束。
        """
        g_vals = np.zeros(self.n_local)
        
        for j in range(self.n_local):
            S = self.Sqrt_Sigmas[j]
            Sw = S @ w
            norm_sw = np.linalg.norm(Sw)
            
            # Term 2: 线性部分 y * w^T * x
            linear_part = self.y[j] * np.dot(w, self.X[j])
            
            # g <= 0
            g_vals[j] = 1.0 - xi[j] + self.unc * norm_sw - linear_part
            
        return g_vals

    def jacobian_w(self, w, xi):
        """
        计算约束关于全局变量 w 的雅可比矩阵 (Jacobian)。
        返回: (n_local, d) 矩阵。
        """
        jac_w = np.zeros((self.n_local, self.d))
        epsilon = 1e-8
        
        for j in range(self.n_local):
            S = self.Sqrt_Sigmas[j]
            Sw = S @ w
            norm_sw = np.linalg.norm(Sw)

            if norm_sw < epsilon:
                grad_soc = np.zeros(self.d)
            else:
                # d/dw ||S w|| = S^T S w / ||S w||
                grad_soc = (S.T @ Sw) / norm_sw
            
            # 完整梯度: unc * grad_soc - y * x
            jac_w[j, :] = self.unc * grad_soc - self.y[j] * self.X[j]
            
        return jac_w

    def jacobian_xi(self, w, xi):
        """
        计算约束关于本地变量 xi 的雅可比矩阵。
        形式: 对角矩阵，对角线元素为 -1。
        返回: (n_local, n_local) 矩阵。
        """
        # g_j = ... - xi_j ... => d/dxi_j = -1
        return -1.0 * np.eye(self.n_local)

    def project_xi(self, xi):
        """
        处理简单的边界约束: xi >= 0 [cite: 2222]
        """
        return np.maximum(xi, 0.0)


class ReferenceCentralizedSolver:
    """
    标准集中式求解器 (Ground Truth Solver)
    用于验证 DistributedNode 接口定义的 Robust SVM 问题。
    """
    def __init__(self, nodes, n_features):
        self.nodes = nodes
        self.d = n_features
        self.n_total_xi = sum(node.n_local for node in nodes)
        self.total_dim = self.d + self.n_total_xi
        
        # 记录每个节点 xi 在大向量中的切片位置
        self.xi_slices = []
        start = self.d
        for node in nodes:
            end = start + node.n_local
            self.xi_slices.append(slice(start, end))
            start = end

    def _unpack(self, x_full):
        """将大向量 X 拆解为 w 和 list of xi"""
        w = x_full[:self.d]
        xi_list = [x_full[s] for s in self.xi_slices]
        return w, xi_list

    def objective(self, x_full):
        """计算全局目标函数值 F(X) = sum f_k"""
        w, xi_list = self._unpack(x_full)
        total_loss = 0.0
        for i, node in enumerate(self.nodes):
            total_loss += node.func(w, xi_list[i])
        return total_loss

    def gradient(self, x_full):
        """计算全局梯度 [grad_w, grad_xi_0, grad_xi_1...]"""
        w, xi_list = self._unpack(x_full)
        
        # 1. 聚合 w 的梯度 (sum of local gradients)
        grad_w_accum = np.zeros(self.d)
        grad_xi_all = []
        
        for i, node in enumerate(self.nodes):
            grad_w_accum += node.grad_w(w, xi_list[i])
            grad_xi_all.append(node.grad_xi(w, xi_list[i]))
            
        return np.concatenate([grad_w_accum] + grad_xi_all)

    def constraints_func(self, x_full):
        """计算所有约束值 (拼接成一个大向量)"""
        w, xi_list = self._unpack(x_full)
        all_cons = []
        for i, node in enumerate(self.nodes):
            all_cons.append(node.constraints(w, xi_list[i]))
        return np.concatenate(all_cons)

    def constraints_jacobian(self, x_full):
        """
        构建稀疏的雅可比矩阵 (这里用 dense 演示)
        Row: All constraints
        Col: [w, xi_0, xi_1, ...]
        """
        w, xi_list = self._unpack(x_full)
        jac_rows = []
        
        for i, node in enumerate(self.nodes):
            # 获取本地雅可比
            # J_w shape: (n_local, d)
            # J_xi shape: (n_local, n_local)
            jac_w_local = node.jacobian_w(w, xi_list[i])
            jac_xi_local = node.jacobian_xi(w, xi_list[i])
            
            # 构建这一行块：[J_w, 0, ..., J_xi, ..., 0]
            row_parts = [jac_w_local]
            
            for j in range(len(self.nodes)):
                if i == j:
                    row_parts.append(jac_xi_local)
                else:
                    other_n = self.nodes[j].n_local
                    row_parts.append(np.zeros((node.n_local, other_n)))
            
            jac_rows.append(np.hstack(row_parts))
            
        return np.vstack(jac_rows)

    def solve(self):
        print(f"Solving centralized problem with {self.total_dim} variables...")

        # 初始猜测
        x0 = np.zeros(self.total_dim)
        x0[self.d:] = 0.1  # Initialize xi > 0
        
        # 1. 复杂不等式约束: g(x) <= 0
        # SciPy NonlinearConstraint: lb <= fun(x) <= ub
        nonlinear_cons = NonlinearConstraint(
            self.constraints_func, 
            -np.inf, 
            0.0, 
            jac=self.constraints_jacobian
        )
        
        # 2. 边界约束: xi >= 0 (w 无约束)
        bounds = [(None, None)] * self.d + [(0.0, None)] * self.n_total_xi
        
        # 使用 trust-constr，因为只有它真正支持 NonlinearConstraint
        res = minimize(
            self.objective, 
            x0, 
            method='trust-constr', 
            jac=self.gradient,
            constraints=[nonlinear_cons],
            bounds=bounds,
            options={
                'verbose': 2,
                'maxiter': 200,
                'gtol': 1e-6,
                'xtol': 1e-6,
                'barrier_tol': 1e-6
            }
        )
        
        w_opt, _ = self._unpack(res.x)
        return w_opt, res.fun


# ==============================================================================
# Part 3: 主程序入口 (验证接口逻辑)
# ==============================================================================

if __name__ == "__main__":
    '''
    # --- 1. 配置参数 (接口测试用，规模可以稍大) ---
    N_SAMPLES = 10   # 样本总数
    N_FEATURES = 10    # 特征维度
    N_NODES = 2        # 分布式节点数
    C_SVM = 1.0        # SVM 正则系数

    cfg = {
        'n_samples': N_SAMPLES,
        'n_features': N_FEATURES,
        'n_nodes': N_NODES,
        'C_svm': C_SVM
    }
    
    print("=== 1. 初始化数据生成 ===")
    data_list, unc_const = generate_distributed_robust_svm_data(N_SAMPLES, N_FEATURES, N_NODES)
    print(f"不确定性常数 (Robust Constant): {unc_const:.4f}")
    
    # --- 2. 实例化分布式节点 ---
    print("\n=== 2. 构建分布式节点接口 ===")
    nodes = []
    for i in range(N_NODES):
        node = DistributedNode(
            node_id=i,
            data=data_list[i],
            C_svm=C_SVM,
            unc_const=unc_const,
            n_features=N_FEATURES,
            total_nodes=N_NODES
        )
        nodes.append(node)
    
    print(f"已创建 {len(nodes)} 个节点对象。每个节点持有约 {N_SAMPLES//N_NODES} 个样本。")

    # --- 3. 模拟 DMC 算法调用流程 (验证单节点接口) ---
    print("\n=== 3. 验证接口调用 (模拟 DMC Solver) ===")
    
    w_global = np.random.randn(N_FEATURES)
    test_node = nodes[0]
    xi_local = np.abs(np.random.randn(test_node.n_local))
    
    print(f"Node {test_node.id} 测试:")
    
    # A. 计算目标函数值
    val = test_node.func(w_global, xi_local)
    print(f"  [Objective] Value: {val:.4f}")
    
    # B. 计算梯度
    g_w = test_node.grad_w(w_global, xi_local)
    g_xi = test_node.grad_xi(w_global, xi_local)
    print(f"  [Gradient w] Shape: {g_w.shape}, Norm: {np.linalg.norm(g_w):.4f}")
    print(f"  [Gradient xi] Shape: {g_xi.shape}, Mean: {np.mean(g_xi):.4f} (Should be {C_SVM})")
    
    # C. 计算约束值
    cons = test_node.constraints(w_global, xi_local)
    max_viol = np.max(cons)
    print(f"  [Constraints] Max Violation: {max_viol:.4f}")
    print(f"  [Constraints] Violated Count: {np.sum(cons > 0)} / {test_node.n_local}")
    
    # D. 计算雅可比矩阵 (Jacobians)
    J_w = test_node.jacobian_w(w_global, xi_local)
    J_xi = test_node.jacobian_xi(w_global, xi_local)
    print(f"  [Jacobian w] Shape: {J_w.shape}")
    print(f"  [Jacobian xi] Shape: {J_xi.shape} (Diagonal: {np.diag(J_xi)[:3]}...)")
    
    print("\n=== 单节点接口验证完成：可以接入 DMC 算法 ===")

    # --- 4. 集中式 ground truth 验证 (规模缩小一点更稳妥) ---
    '''
    print("\n=== 4. 集中式求解器验证 (缩小规模) ===")
    N_SAMPLES = 10   # 样本稍少一点以便求解更稳定
    N_FEATURES = 10
    N_NODES = 2
    C_SVM = 1.0
    
    data_list, unc_const = generate_distributed_robust_svm_data(N_SAMPLES, N_FEATURES, N_NODES, seed=123)
    
    nodes = []
    for i in range(N_NODES):
        nodes.append(DistributedNode(i, data_list[i], C_SVM, unc_const, N_FEATURES, N_NODES))
        
    solver = ReferenceCentralizedSolver(nodes, N_FEATURES)
    w_star, f_star = solver.solve()
    
    print("\n" + "="*30)
    print("GROUND TRUTH SOLUTION")
    print("="*30)
    print(f"Optimal Objective Value: {f_star:.6f}")
    print(f"Optimal w (first 5 dims): {w_star[:5]}")
    
    print("\n现在你可以用同样的接口跑 DMC，并对比 w 是否收敛到 w_star。")
