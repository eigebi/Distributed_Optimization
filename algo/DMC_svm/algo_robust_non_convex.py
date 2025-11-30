import numpy as np
from env_robust_non_convex import *
import matplotlib.pyplot as plt

class DMCSolver:
    def __init__(self, env,cfg):
        self.node = env

        self.N = cfg["n_nodes"] # agents数量
        self.K = cfg["n_features"]  # w维度
        self.M = [env[i].n_local for i in range(self.N)]  # 每个agent本地数据量, 也是xi的维度

       
        self.agents_x = np.ones((self.N,self.K))*0.1 # 每个 Agent 的变量向量化存储
        self.agent_xi = [np.ones((self.M[i])) for i in range(self.N)]  # 每个 Agent 的xi变量向量化存储, locally
        self.gamma = np.zeros((self.N, self.K)) # 每个 Agent 的consensus dual向量化存储
        self.gamma_xi = [np.zeros((self.M[i])) for i in range(self.N)]  # 每个 Agent 的xi dual向量化存储

        self.z = np.ones(self.K)*0.1  # 全局w向量化存储
        #self.z = np.random.randn(self.K)*0.1
        self.lamb = [np.zeros((self.M[i])) for i in range(self.N)] #取决于每个agent sample的个数
        

    def solve(self, max_iter=500, rho=5000, theta=0.01, eta_c = 1):
        history = {'util': [], 'viol': [], 'consensus': []}
        z = self.z.copy()  # 全局变量向量化存储
        z_xi = [np.zeros((self.M[i])) for i in range(self.N)]  # xi 变量的辅助变量向量化存储
        gamma = self.gamma.copy()  # consensus multiplier 向量化存储
        gamma_xi = [g.copy() for g in self.gamma_xi]  # xi multiplier 向量化存储
        x = self.agents_x.copy()  # 本地变量向量化存储
        xi = [x.copy() for x in self.agent_xi]  # 本地xi变量向量化存储
        lamb = [l.copy() for l in self.lamb]
        N = self.N
        K = self.K
        M = self.M
        
        eta = 0 # need to tune 

        viol_history = []
        total_viol = []
        beta = 0
        j_loss = []
        for t in range(max_iter):
            # consensus update for z
            next_z = (beta*z+np.sum(rho* x + gamma, 0))/(self.N * rho+beta)
            #next_z = np.clip(next_z, -2.0, 2.0)
            z = next_z
            
            next_z_xi = [(beta/M[i]*z_xi[i]+rho*xi[i]+gamma_xi[i])/(rho + beta/M[i]) for i in range(N)]
            z_xi = [np.maximum(next_z_xi[i], 0.0) for i in range(N)]
            g_global = []
            v_temp = []
            obj = []
            j_temp = 0
            

            # --- Step 1: Local Update (Each Agent) ---
            
            for i in range(self.N):
                agent = self.node[i]
                g_w = agent.grad_w(z, z_xi[i])
                g_xi = agent.grad_xi(z, z_xi[i])
                constraints = agent.constraints(z, z_xi[i])
                J_w = agent.jacobian_w(z, z_xi[i])
                J_xi = agent.jacobian_xi(z, z_xi[i])
                constraints = constraints
                J_w = J_w
                J_xi = J_xi


                v_temp.append(np.mean(np.maximum(constraints, 0)**2))

                dw = g_w + lamb[i]@J_w
                dxi = g_xi + lamb[i]@J_xi

                j_temp += np.sum((dw+gamma[i])**2)#+np.sum((dxi)**2)

                latest_w = z - (dw + gamma[i]) / rho
                latest_xi = z_xi[i] - (dxi+gamma_xi[i]) / rho ###
                #latest_xi = xi[i] - (dxi) / 10000
                latest_lambda = np.minimum(np.maximum(lamb[i] + (theta * constraints)/(1+theta*eta) ,0),100) # 投影到非负正交集
                gamma[i] += rho * (latest_w - z)
                #x[i] = np.clip(latest_w, -1.0, 1.0)


                j_temp += ((lamb[i] > 0).astype(float)) @ (constraints**2)
                j_temp += np.sum(x[i] - z)**2

                gamma_xi[i] += rho * (latest_xi - z_xi[i]) ###
                #xi[i] = (beta * xi[i] +rho*latest_xi) / (rho + beta) ###
                xi[i] = latest_xi
                x[i] = latest_w
                lamb[i] = latest_lambda

            total_viol.append(np.mean(v_temp))
            j_loss.append(j_temp/N)
            

            eta=1/(t+1)**(1/4)/10
            beta =  10+ 10/eta
            
            
            print(z[:5])
        
            #if t % 50 == 0:
            #    print(f"Iter {t}: Util={u:.2f}, QoS Viol={total_qos_viol:.4f}, ConsErr={cons_err:.4f}")
 
        return j_loss, total_viol
        












if __name__ == "__main__":
     # --- 1. 配置参数 ---
    N_SAMPLES = 100   # 样本总数
    N_FEATURES = 10    # 特征维度
    N_NODES = 2      # 分布式节点数
    C_SVM = 1.0        # SVM 正则系数
    cfg = {
        'n_samples': N_SAMPLES,
        'n_features': N_FEATURES,
        'n_nodes': N_NODES,
        'C_svm': C_SVM
    }

    data_list, unc_const, meta = generate_distributed_robust_svm_data(N_SAMPLES, N_FEATURES, N_NODES, seed=123)
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
    test = False
    if test==True:
        solver = ReferenceCentralizedSolverWithSparse(
        nodes=nodes,
        n_features=N_FEATURES,
        alpha_sparse=5.0,
        tau_sparse=5.0   # 可以调小/调大观察约束是否激活
        )   
        w_star, xi_list_star, res = solver.solve(maxiter=100)
        
        print("\n" + "="*30)
        print("GROUND TRUTH SOLUTION")
        print("="*30)
        print(f"Optimal Objective Value: {res.fun}")
        print(f"Optimal w (first 5 dims): {w_star[:5]}")
        
        print("\n现在你可以用同样的接口跑 DMC，并对比 w 是否收敛到 w_star。")

    algo = DMCSolver(nodes,cfg)
    iter_num = 100
    j_1, g_1 = algo.solve(max_iter=iter_num, rho = 1, theta = 0.1, eta_c = 1)
    j_2, g_2 = algo.solve(max_iter=iter_num, rho = 10, theta = 0.1, eta_c = 3)
    j_3, g_3 = algo.solve(max_iter=iter_num, rho = 100, theta = 0.1, eta_c = 5)
    j_4, g_4 = algo.solve(max_iter=iter_num, rho = 1000, theta = 0.1, eta_c = 7)
    #g_dad = algo.solve(max_iter=1000, rho = 5000, theta = 0.1, dad = True)
    plt.figure(1)
    plt.plot(j_1, 
         label='DMC $\\alpha=1$', 
         color='#1f77b4',         # 专业的深蓝色
         linestyle='-',           # 实线
         linewidth=2)            # 【关键】每隔10个点画一个标记，防止太密
    plt.plot(j_2, 
         label='DMC $\\alpha=3$', 
         color='#ff7f0e',         # 专业的橙色
         linestyle='-',           # 实线
         linewidth=2)            # 【关键】每隔10个点画一个标记，防止太密
    plt.plot(j_3, 
         label='DMC $\\alpha=5$',          
         color='#2ca02c',         # 专业的绿色
         linestyle='-',           # 实线
         linewidth=2)            # 【关键】每隔10个点画一个标记，防止太密
    plt.plot(j_4, 
         label='DMC $\\alpha=7$',          
         color='#d62728',         # 专业的红色
         linestyle='-',           # 实线
         linewidth=2)            # 【关键】每隔10个点画一个标记

    # 2. 绘制 Gradient Ascent Descent (红色，空心方块 's')
    #plt.yscale('log')

    # --- 装饰设置 ---

    # 设置网格为虚线
    plt.grid(True, linestyle='--', alpha=0.7)

    # 设置图例 (Legend)
    plt.legend(loc='best',           # 自动寻找最佳位置
            frameon=True,         # 显示图例边框
            fancybox=True,        # 圆角边框
            framealpha=0.9,       # 边框透明度
            edgecolor='gray',     # 边框颜色
            fontsize=12)

    # 坐标轴标签
    #plt.xlabel('Iteration number', fontsize=12, fontweight='bold')
    #plt.ylabel('Infeasibility', fontsize=12, fontweight='bold')
    #plt.legend(['Proposed DMC', 'Gradient Ascent Descent'])
    plt.yscale('log')
    plt.xlabel('Iteration number', fontsize=15)
    #plt.xlim(0, 100)
    plt.ylabel('Stationary Gap', fontsize=15)
    plt.show()
    
    plt.figure(2)
    plt.plot(g_1, 
        label='DMC $\\alpha=1$', 
        color='#1f77b4',         # 专业的深蓝色
        linestyle='-',           # 实线
        linewidth=2)            # 【关键】每隔10个点画一个标记，防止太密
    plt.plot(g_2, 
        label='DMC $\\alpha=3$', 
        color='#ff7f0e',         # 专业的橙色
        linestyle='-',           # 实线
        linewidth=2)            # 【关键】每隔10个点画一个标记，防止太密
    plt.plot(g_3, 
        label='DMC $\\alpha=5$',          
        color='#2ca02c',         # 专业的绿色
        linestyle='-',           # 实线
        linewidth=2)            # 【关键】每隔10个点画一个标记，防止太密
    plt.plot(g_4, 
        label='DMC $\\alpha=7$',          
        color='#d62728',         # 专业的红色
        linestyle='-',           # 实线
        linewidth=2)            # 【关键】每隔10个点画一个标记

    # 2. 绘制 Gradient Ascent Descent (红色，空心方块 's')
    #plt.yscale('log')

    # --- 装饰设置 ---

    # 设置网格为虚线
    plt.grid(True, linestyle='--', alpha=0.7)

    # 设置图例 (Legend)
    plt.legend(loc='best',           # 自动寻找最佳位置
            frameon=True,         # 显示图例边框
            fancybox=True,        # 圆角边框
            framealpha=0.9,       # 边框透明度
            edgecolor='gray',     # 边框颜色
            fontsize=12)

    # 坐标轴标签
    #plt.xlabel('Iteration number', fontsize=12, fontweight='bold')
    #plt.ylabel('Infeasibility', fontsize=12, fontweight='bold')
    #plt.legend(['Proposed DMC', 'Gradient Ascent Descent'])
    plt.yscale('log')
    plt.xlabel('Iteration number', fontsize=15)
    #plt.xlim(0, 100)
    plt.ylabel('Infeasibility', fontsize=15)
    plt.show()
