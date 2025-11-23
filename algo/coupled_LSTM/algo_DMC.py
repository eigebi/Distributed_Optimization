import numpy as np
from env_test import WirelessEnvNumpy, StandardTopology, EnvCfg

if __name__ == "__main__":
    cfg = EnvCfg()
    topo = StandardTopology(cfg)
    bs_xy = topo.generate_hex_bs(num_rings=2)
    data = topo.generate_ues_robust(bs_xy, K_per_bs=12, num_slices=3)
    
    env = WirelessEnvNumpy(len(bs_xy), len(data[1]), 3, data, cfg)
    print(f"Topology: {env.B} BS, {env.K} UEs")

    M_g = 10
    sigma = 100
    L_f = 100
    N = 1 # number of problems/env 暂时先考虑单个环境
    tau = 1
    rho = (N*tau*L_f)**2+np.sqrt((N*tau*L_f)**4+7*N*N*tau*tau*L_f*L_f*L_f)
    theta = rho*rho/(15*M_g*M_g*(2*rho+7*L_f))


    ub = None

    z = np.random.randn(env.K*2)
    z = np.clip(z,0,ub)
    z_graph = None #应该是在env里有类似的能转换

    dx = np.zeros((env.B,env.K*2), dtype=np.float32)
    for i in range(env.B):
        dx[i] = env.get_net_utility_gradients(z[:env.K], z[env.K:], i)