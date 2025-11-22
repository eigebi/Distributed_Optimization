import matplotlib.pyplot as plt
import numpy as np

def plot_network_topology(bs_xy, ue_xy, s_u, b_u=None, title="Network Topology with Slicing"):
    """
    绘制基站和用户的分布图，不同 Slice 的用户用不同颜色标示。
    
    Args:
        bs_xy: (B, 2) 基站坐标
        ue_xy: (K, 2) 用户坐标
        s_u:   (K,) 每个用户的切片索引 (0, 1, 2...)
        b_u:   (K,) 每个用户归属的基站索引 (可选，用于画连线)
    """
    plt.figure(figsize=(10, 8))
    
    # 1. 绘制连接线 (UE -> Serving BS)
    # 画在最底层 (zorder=1)，颜色淡一点
    if b_u is not None:
        print("Plotting associations...")
        for k in range(len(ue_xy)):
            bs_idx = b_u[k]
            ue_pos = ue_xy[k]
            bs_pos = bs_xy[bs_idx]
            # 画线
            plt.plot([ue_pos[0], bs_pos[0]], [ue_pos[1], bs_pos[1]], 
                     color='gray', alpha=0.2, linewidth=0.5, zorder=1)

    # 2. 绘制用户 (UE) - 按 Slice 分色
    # zorder=2
    unique_slices = np.unique(s_u)
    # 使用 tab10 色板，足够区分 10 个以内的 slice
    cmap = plt.get_cmap('tab10') 
    
    for i, s_idx in enumerate(unique_slices):
        mask = (s_u == s_idx)
        points = ue_xy[mask]
        plt.scatter(points[:, 0], points[:, 1], 
                    color=cmap(i), s=20, alpha=0.8, 
                    label=f'Slice {s_idx} User', zorder=2)

    # 3. 绘制基站 (BS)
    # zorder=3 (最上层)
    plt.scatter(bs_xy[:, 0], bs_xy[:, 1], 
                c='black', marker='^', s=150, 
                label='Base Station', edgecolors='white', zorder=3)
    
    # 标上基站 ID
    for i, (x, y) in enumerate(bs_xy):
        plt.text(x, y + 50, f'BS{i}', fontsize=9, ha='center', color='black', fontweight='bold')

    # 4. 图表装饰
    plt.title(title)
    plt.xlabel("X Position (meters)")
    plt.ylabel("Y Position (meters)")
    plt.legend(loc='upper right', shadow=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal') # 保持 1:1 比例，这样六边形才正
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 测试脚本 (结合之前的生成代码)
# ==========================================
if __name__ == "__main__":
    # 假设你已经运行了之前的生成代码，这里为了演示独立运行，
    # 我把生成部分简单 mock 一下，或者你可以直接 import 之前的模块
    
    # 1. 引入之前的模块 (假设保存为 env_numpy.py)
    # from env_numpy import EnvCfg, StandardTopology
    
    # 为了演示方便，直接复制之前的调用逻辑：
    from env_slicing import EnvCfg, StandardTopology # 请确保文件名对应
    
    cfg = EnvCfg()
    topo = StandardTopology(cfg, isd=500.0)
    
    # 生成 19 个基站 (2圈)
    print("Generating 19-BS topology...")
    bs_xy = topo.generate_hex_bs(num_rings=2)
    
    # 生成用户 (每个基站 10 个用户，3 个切片)
    # 注意：generate_ues_robust 返回的是 tuple
    print("Generating UEs with Slice info...")
    topo_data = topo.generate_ues_robust(bs_xy, K_per_bs=10, num_slices=3)
    
    # 解包数据
    # topo_data[0] 是 bs_xy (可能没变)
    # topo_data[1] 是 ue_xy
    # topo_data[2] 是 b_u
    # topo_data[3] 是 s_u
    ue_xy = topo_data[1]
    b_u = topo_data[2]
    s_u = topo_data[3]
    
    # 调用绘图
    plot_network_topology(bs_xy, ue_xy, s_u, b_u, title="19-Cell Hexagonal Network with 3 Slices")
