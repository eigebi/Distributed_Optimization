import numpy as np
from scipy.optimize import minimize

# ==========================================
# 1. 真实物理环境配置 (High Load Scenario)
# ==========================================
NUM_BS = 2
NUM_UE = 2
NUM_ANT = 4           # 4天线，保证有足够的自由度去消干扰
P_MAX = 10.0          # 最大功率 10 Watts (40 dBm)

# [关键修改 1] 提升噪声水平 (模拟城市环境/高底噪)
# -90 dBm = 1e-12 W. 配合真实路损，这需要瓦特级的发射功率
NOISE_POWER = 1e-12   

# [关键修改 2] 设定一个极具挑战性的目标
# 边缘用户，路损大，还要达到 5 dB
TARGET_SINR_DB = 5.0  
TARGET_SINR_LIN = 10**(TARGET_SINR_DB / 10.0)

# 坐标 (单位: 米)
# 两个基站相距 600米
pos_bs = np.array([[0, 0], [600, 0]])
# 两个用户在中间 (300米处) "肉搏"
# 距离各自基站 300米，且互相距离很近
pos_ue = np.array([[295, 0], [305, 0]]) 

np.random.seed(1024) 

# --- 信道生成 (真实路损模型) ---
H = np.zeros((NUM_UE, NUM_BS, NUM_ANT), dtype=complex)

print("--- 物理环境检查 ---")
for u in range(NUM_UE):
    for b in range(NUM_BS):
        d = np.linalg.norm(pos_ue[u] - pos_bs[b])
        
        # [关键修改 3] 使用更真实的 Path Loss
        # 自由空间损耗 FSPL (2GHz): ~ 30 + 20log(d) + ...
        # 简化为 d^-3.8 (典型的城市非视距传播)
        # 300m^-3.8 ≈ 3.8e-10
        path_gain = d ** (-3.8) 
        
        # 瑞利衰落
        fading = (np.random.randn(NUM_ANT) + 1j * np.random.randn(NUM_ANT)) / np.sqrt(2)
        H[u, b, :] = np.sqrt(path_gain) * fading
        
        # 打印路损情况
        gain_db = 10 * np.log10(np.mean(np.abs(H[u, b, :])**2))
        link_type = "主信号" if u == b else "强干扰"
        print(f"{link_type} (BS{b}->UE{u}): 距离={d:.0f}m, 路径增益={gain_db:.1f} dB")

print(f"噪声水平: {10*np.log10(NOISE_POWER):.1f} dBm")
print("----------------------\n")

# ==========================================
# 2. 变量与求解器封装
# ==========================================
def pack(W): return np.concatenate([W.real.flatten(), W.imag.flatten()])
def unpack(x): return (x[:NUM_BS*NUM_ANT] + 1j*x[NUM_BS*NUM_ANT:]).reshape((NUM_BS, NUM_ANT))

def objective(x):
    W = unpack(x)
    return np.sum(np.abs(W)**2) # 最小化总功率

def constraints(x):
    W = unpack(x)
    cons = []
    # P_max 约束 (每个基站)
    for b in range(NUM_BS):
        p_bs = np.sum(np.abs(W[b, :])**2)
        cons.append(P_MAX - p_bs) # P_max - P >= 0
        
    # SINR 约束
    for k in range(NUM_UE):
        my_bs = k
        sig = np.abs(np.vdot(H[k, my_bs, :], W[k, :]))**2
        
        interf_bs = 1 - k
        interf = np.abs(np.vdot(H[k, interf_bs, :], W[interf_bs, :]))**2
        
        # Constraint: Signal >= Target * (Interf + Noise)
        cons.append(sig - TARGET_SINR_LIN * (interf + NOISE_POWER))
    
    return np.array(cons)

# ==========================================
# 3. 初始化 (至关重要)
# ==========================================
# 在这种高压环境下，如果不给一个好的初值，Solver 很难找到可行域
print("--- 初始化 (Full Power MRT) ---")
W_init = np.zeros((NUM_BS, NUM_ANT), dtype=complex)
for k in range(NUM_BS):
    h_main = H[k, k, :]
    # MRT 方向 + 满功率发射 (10W)
    # 既然环境这么恶劣，初始就得全力以赴
    w_init = np.conj(h_main)
    w_init = w_init / np.linalg.norm(w_init) * np.sqrt(P_MAX)
    W_init[k, :] = w_init

x0 = pack(W_init)

# 检查初始是否满足 SINR
print("检查初始点可行性:")
cons_val = constraints(x0)
sinr_margin = cons_val[NUM_BS:] # 后两个是 SINR 约束
if np.all(sinr_margin >= 0):
    print("  -> 初始满功率可以满足 SINR，Solver 将尝试降低功率。")
else:
    print(f"  -> 警告：即使满功率也无法满足 SINR (缺口 {sinr_margin})。")
    print("     Solver 将尝试调整相位来消除干扰以满足约束。")

# ==========================================
# 4. 求解
# ==========================================
print("\n--- 启动 SLSQP 优化器 ---")
res = minimize(objective, x0, method='SLSQP', 
               constraints={'type': 'ineq', 'fun': constraints},
               options={'maxiter': 1000, 'ftol': 1e-9, 'disp': True})

# ==========================================
# 5. 结果分析
# ==========================================
print(f"\n优化成功? {res.success}")
print(f"退出信息: {res.message}")

W_opt = unpack(res.x)
total_p = np.sum(np.abs(W_opt)**2)

print(f"\n[最终状态]")
print(f"总功耗: {total_p:.4f} W (Limit: {2*P_MAX} W)")

for k in range(NUM_UE):
    sig = np.abs(np.vdot(H[k, k, :], W_opt[k, :]))**2
    interf = np.abs(np.vdot(H[k, 1-k, :], W_opt[1-k, :]))**2
    sinr_val = sig / (interf + NOISE_POWER)
    sinr_db = 10*np.log10(sinr_val)
    
    p_bs = np.sum(np.abs(W_opt[k, :])**2)
    
    print(f"用户 {k} (BS{k}):")
    print(f"  - 发射功率: {p_bs:.4f} W")
    print(f"  - 接收信号: {sig:.2e}")
    print(f"  - 接收干扰: {interf:.2e}")
    print(f"  - 最终 SINR: {sinr_db:.2f} dB (Target: {TARGET_SINR_DB} dB)")