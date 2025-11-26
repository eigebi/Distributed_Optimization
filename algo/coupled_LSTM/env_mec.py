import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ==========================================
# 1. 系统参数配置 (Configuration)
# ==========================================
@dataclass
class SystemConfig:
    # --- 拓扑规模 ---
    num_edges: int = 100           # Edge Server (Agent) 数量
    tasks_per_edge: int = 10      # 每个 Edge 下的任务数量
    
    # --- 物理常数 ---
    # 能耗系数: P = kappa * f^3 (f in Hz)
    # 实际 kappa 约为 1e-28。
    # 为了数值稳定，我们使用 "GHz" 作为 f 的单位。
    # 换算逻辑: E = 1e-28 * (f_GHz * 1e9)^3 * t = 1e-1 * f_GHz^3 * t
    # 所以 scaled_kappa = 0.1
    kappa_scaled: float = 0.1    
    
    # --- 任务分布范围 (保证有解) ---
    # 任务量 W (Giga Cycles): 0.5 ~ 1.5 Gcycles
    W_min: float = 0.5
    W_max: float = 1.5
    
    # 最大时延 T_max (s): 0.5s ~ 1.0s
    # 这意味着最小需要的 f 约为 1.5/0.5 = 3 GHz，最大可能到 10GHz
    T_min: float = 0.5
    T_max: float = 1.0
    
    # --- 效用与权重 ---
    # 目标: Max alpha * ln(1+f) - beta * E
    alpha_utility: float = 10.0   # 调高 Utility 权重以鼓励 f 增大
    beta_energy: float = 5.0      # 能耗权重
    
    # --- 资源限制 ---
    # 总任务数 = 20。平均每个任务最少需要 ~1.5 GHz。
    # 理论最低总需求 ~30 GHz。
    # 我们给 Cloud 设置 50 GHz，制造适度的资源竞争（Active Constraint）。
    F_cloud_max: float = 2500.0     # (GHz)
    
    # 变量边界 (GHz)
    f_min_bound: float = 0.1
    f_max_bound: float = 10.0
    
    seed: int = 42

# ==========================================
# 2. 集中式 MEC 环境 (Centralized Env)
# ==========================================
class CentralizedMECEnv:
    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        
        self.total_tasks = cfg.num_edges * cfg.tasks_per_edge
        
        # --- 随机生成任务参数 ---
        # W: 任务量 (Gcycles)
        self.W = self.rng.uniform(cfg.W_min, cfg.W_max, self.total_tasks)
        # T_max: 最大时延 (s)
        self.T_max = self.rng.uniform(cfg.T_min, cfg.T_max, self.total_tasks)
        # Alpha: 任务对速率的偏好 (Utility Weight) - 稍微随机化一点
        self.alpha_vec = self.rng.uniform(0.8, 1.2, self.total_tasks) * cfg.alpha_utility
        
        print(f"Environment Initialized: {self.total_tasks} Tasks.")
        print(f"Total Cloud Capacity: {cfg.F_cloud_max} GHz")
        # 检查理论可行性
        min_f_req = self.W / self.T_max
        print(f"Theoretical Min Total Frequency Required: {np.sum(min_f_req):.2f} GHz")
        if np.sum(min_f_req) > cfg.F_cloud_max:
            print("WARNING: Problem might be infeasible (Tight Constraints)!")
        else:
            print("Status: Feasible region exists.")

    def objective(self, x):
        """
        Scipy 只能做 Minimize。
        我们原本的目标: Max Utility - Energy
        转化后的目标: Min Energy - Utility
        x: flat array [f_0, ..., f_N, t_0, ..., t_N]
        """
        N = self.total_tasks
        f = x[:N]
        t = x[N:]
        
        # 1. 计算能耗 (Scaled)
        # E = 0.1 * f^3 * t
        energy = self.cfg.kappa_scaled * (f**3) * t
        
        # 2. 计算效用
        # U = alpha * ln(1 + f)
        utility = self.alpha_vec * np.log(1 + f)
        
        # 3. 组合目标 (加负号因为是 Minimize)
        # Min (beta * Energy - Utility)
        net_cost = np.sum(self.cfg.beta_energy * energy - utility)
        
        return net_cost
    def get_local_utility_gradients(self, f_vec, t_vec, id):
        """
        计算每个任务(Agent)的本地目标函数(Net Cost)关于其本地变量的梯度。
        
        Objective_i = beta * Energy_i - Utility_i
                    = beta * (kappa * f_i^3 * t_i) - alpha_i * ln(1 + f_i)
        
        Args:
            f_vec: (N_tasks,) 当前分配的频率
            t_vec: (N_tasks,) 当前分配的时间
            
        Returns:
            grads: list of dict, 每个元素对应一个任务
                   {'grad_f': scalar, 'grad_t': scalar}
        """
        N = self.total_tasks
        grads = []
        
        # 预取参数
        beta = self.cfg.beta_energy
        kappa = self.cfg.kappa_scaled

        dE_df = kappa * 3 * (f_vec**2) * t_vec
        dU_df = self.alpha_vec[id*self.cfg.tasks_per_edge:(id+1)*self.cfg.tasks_per_edge] / (1.0 + f_vec)
        grad_f_vec = beta * dE_df - dU_df

        dE_dt = kappa * (f_vec**3)
        grad_t_vec = beta * dE_dt

        return grad_f_vec, grad_t_vec
        
        

    def constraints(self):
        """
        定义 Scipy 格式的约束
        Constraints are: fun(x) >= 0
        """
        N = self.total_tasks
        cons = []
        
        # --- C1: Local Non-convex (f * t >= W) ---
        # 形式: f*t - W >= 0
        # 这是一个 array constraint，但在 scipy 里最好拆成一个个
        for i in range(N):
            def constr_comp(x, idx=i):
                f = x[idx]
                t = x[N + idx]
                return f * t - self.W[idx]
            cons.append({'type': 'ineq', 'fun': constr_comp})
            
        # --- C2: Global Coupled (Sum(f) <= F_max) ---
        # 形式: F_max - Sum(f) >= 0
        def constr_global(x):
            f = x[:N]
            return self.cfg.F_cloud_max - np.sum(f)
        cons.append({'type': 'ineq', 'fun': constr_global})
        
        # --- C3: Latency is handled by Bounds, but for safety ---
        # t <= T_max  => T_max - t >= 0
        # (This is optional if bounds are set correctly, but good for stability)
        for i in range(N):
            def constr_time(x, idx=i):
                t = x[N + idx]
                return self.T_max[idx] - t
            cons.append({'type': 'ineq', 'fun': constr_time})
            
        return cons
    def get_local_constraint_gradients(self, f_vec, t_vec,i):
        """
        计算每个任务(Agent)的本地非凸 QoS 约束关于其本地变量的梯度。
        Constraint: g_i = W_i - f_i * t_i <= 0
        
        Args:
            f_vec: (N_tasks,) 当前分配的频率
            t_vec: (N_tasks,) 当前分配的时间
            
        Returns:
            grads: list of dict, 每个元素对应一个任务
                   {'grad_f': scalar, 'grad_t': scalar, 'violation': scalar}
        """
        N = self.total_tasks
        grads = []
        g = self.W[i*self.cfg.tasks_per_edge:(i+1)*self.cfg.tasks_per_edge] -  f_vec * t_vec
        grad_f_vec = -t_vec
        grad_t_vec = -f_vec

        return grad_f_vec, grad_t_vec, g
        
        for i in range(N):
            # 1. 获取当前状态
            f_i = f_vec[i]
            t_i = t_vec[i]
            W_i = self.W[i]
            
            # 2. 计算违约值 (Violation)
            # g(x) = W - f*t
            # 如果 g > 0, 说明 f*t < W (未完成任务), 需要惩罚
            val = W_i - f_i * t_i
            violation = max(0, val)
            
            # 3. 计算梯度 (Jacobian of g(x))
            # dg/df = -t
            grad_f = -t_i
            
            # dg/dt = -f
            grad_t = -f_i
            
            # 如果您需要归一化 (Scaling) 以防止梯度爆炸:
            # 假设我们希望梯度量级在 1.0 左右
            # 当前 f ~ 5, t ~ 0.5. grad_f ~ -0.5, grad_t ~ -5. 
            # 量级不平衡 (f梯度小, t梯度大)
            # 建议不做硬性归一化，而是通过 Adam/RMSprop 优化器自动处理，
            # 或者在 DMC 更新时给 f 和 t 不同的步长。
            
            grads.append({
                'id': i,
                'grad_f': grad_f,
                'grad_t': grad_t,
                'violation': violation,
                'is_active': violation > 0 # 标记约束是否被激活
            })
            
        return grads

    def get_bounds(self):
        """
        定义变量边界
        f: [0.1, 10] GHz
        t: [0.01, T_max] s
        """
        N = self.total_tasks
        bnds = []
        # Bounds for f
        for i in range(N):
            bnds.append((self.cfg.f_min_bound, self.cfg.f_max_bound))
        # Bounds for t
        for i in range(N):
            bnds.append((0.01, self.T_max[i]))
        return tuple(bnds)

# ==========================================
# 3. 求解器 (Centralized Baseline)
# ==========================================
def solve_centralized_scipy(env):
    N = env.total_tasks
    
    # 1. 初始猜测 (Initial Guess)
    # 给一个合理的初值非常重要，特别是对于非凸问题
    # 设 f 为 Cloud Capacity 的平均分，t 为刚好满足 W/f
    f0 = np.ones(N) * (env.cfg.F_cloud_max / N)
    t0 = env.W / f0 
    # 修正 t0 以满足 T_max
    t0 = np.minimum(t0, env.T_max)
    # 再次修正 f0 以满足 W (f = W/t)
    f0 = np.maximum(f0, env.W / t0)
    
    x0 = np.concatenate([f0, t0])
    
    print("\nStarting Scipy SLSQP Optimization...")
    
    # 2. 调用求解器
    # SLSQP 是处理含非线性约束问题的标准算法
    res = minimize(
        fun=env.objective,
        x0=x0,
        bounds=env.get_bounds(),
        constraints=env.constraints(),
        method='SLSQP',
        options={'maxiter': 500, 'ftol': 1e-6, 'disp': True}
    )
    
    # 3. 解析结果
    f_opt = res.x[:N]
    t_opt = res.x[N:]
    
    return {
        'success': res.success,
        'message': res.message,
        'fun': -res.fun, # 转回 Utility - Energy
        'f': f_opt,
        't': t_opt
    }

# ==========================================
# 4. 执行与验证
# ==========================================
if __name__ == "__main__":
    # 配置
    cfg = SystemConfig()
    env = CentralizedMECEnv(cfg)
    
    # 求解
    result = solve_centralized_scipy(env)
    
    print("\n" + "="*40)
    print(f"Optimization Success: {result['success']}")
    print(f"Message: {result['message']}")
    print(f"Final Net Utility: {result['fun']:.4f}")
    print("="*40)
    
    if result['success']:
        # --- 验证结果合理性 ---
        f_vec = result['f']
        t_vec = result['t']
        
        # 1. 检查非凸约束 f*t >= W
        products = f_vec * t_vec
        W_vec = env.W
        violation = np.maximum(0, W_vec - products)
        max_viol = np.max(violation)
        print(f"Max Constraint Violation (f*t - W): {max_viol:.4e}")
        
        # 2. 检查全局资源约束
        total_f = np.sum(f_vec)
        print(f"Total Frequency Used: {total_f:.2f} / {cfg.F_cloud_max} GHz")
        
        # 3. 检查时延约束
        time_viol = np.maximum(0, t_vec - env.T_max)
        print(f"Max Time Violation: {np.max(time_viol):.4e}")

        # --- 绘图 ---
        plt.figure(figsize=(12, 5))
        
        # 图1: 每个任务的资源分配 vs 需求
        plt.subplot(1, 2, 1)
        indices = np.arange(env.total_tasks)
        plt.bar(indices, f_vec, label='Allocated f (GHz)', alpha=0.7)
        # 计算最小需要的 f
        min_f = env.W / env.T_max
        plt.plot(indices, min_f, 'r^', label='Min Required f (W/Tmax)')
        plt.xlabel('Task ID')
        plt.ylabel('Frequency (GHz)')
        plt.title('Resource Allocation per Task')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 图2: 验证 f*t >= W (是否贴紧边界?)
        plt.subplot(1, 2, 2)
        # Ratio = (f*t) / W. If == 1.0, means strictly active constraint.
        ratio = products / W_vec
        plt.plot(indices, ratio, 'o-', color='green')
        plt.axhline(1.0, color='red', linestyle='--', label='Boundary (1.0)')
        plt.xlabel('Task ID')
        plt.ylabel('Ratio (f*t / W)')
        plt.title('Constraint Tightness check')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("\nInterpretation:")
        print("1. If 'Ratio' is close to 1.0, the non-convex constraint is ACTIVE.")
        print("   (This is expected because Energy minimizes when t is large).")
        print("2. If 'Total F' is close to 40 GHz, the global coupling is ACTIVE.")
        print("   (This creates the price competition).")