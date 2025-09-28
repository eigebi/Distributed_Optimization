import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
acc = np.load('data/DAMC/acc_dz_non.npy',allow_pickle=True)
inf = np.load('data/DAMC/inf_dz_non.npy',allow_pickle=True)
grad_gap = np.load('data/DAMC/grad_gap_dz_non.npy',allow_pickle=True)
#acc_dx = np.load('acc_dx.npy',allow_pickle=True)
#inf_dx = np.load('inf_dx.npy',allow_pickle=True)
#grad_gap_dx = np.load('grad_gap_dx.npy',allow_pickle=True)




acc_dx = np.load('data/DAMC/acc_dx_non.npy',allow_pickle=True)
inf_dx = np.load('data/DAMC/inf_dx_non.npy',allow_pickle=True)
grad_gap_dx = np.load('data/DAMC/grad_gap_dx_non.npy',allow_pickle=True)

acc_dad = np.load('data/DAMC/acc_dx_std_non.npy',allow_pickle=True)
inf_dad = np.load('data/DAMC/inf_dx_std_non.npy',allow_pickle=True)
grad_gap_dad = np.load('data/DAMC/grad_gap_dx_std_non.npy',allow_pickle=True)
# Extract the first 8000 elements from all arrays in acc
# Truncate all sequences in acc to 8000 elements and compute the mean and standard deviation across the 100 sequences for each point
acc = [arr[:6000] for arr in acc]
acc_mean = np.mean(np.array(acc), axis=0)  # Compute the mean across the 100 sequences for each point
acc_std = np.std(acc, axis=0)   # Compute the standard deviation across the 100 sequences for each point

acc_dx = [arr[:6000] for arr in acc_dx]
acc_dx_mean = np.mean(np.array(acc_dx), axis=0)  # Compute the mean across the 100 sequences for each point
acc_dx_std = np.std(acc_dx, axis=0)   # Compute the standard deviation across the 100 sequences for each point

acc_dad = [arr[:6000] for arr in acc_dad]
acc_dad_mean = np.mean(np.array(acc_dad), axis=0)  # Compute the mean across the 100 sequences for each point
# Compute the standard deviation across the 100 sequences for each point
acc_dad_std = np.std(acc_dad, axis=0)

# Apply a sliding window average with a window size of 5
window_size = 2
acc_dad_mean = np.convolve(acc_dad_mean, np.ones(window_size)/window_size, mode='same')
acc_dad_std = np.convolve(acc_dad_std, np.ones(window_size)/window_size, mode='same')

# Plot the mean with a shaded area representing the standard deviation
plt.figure(0)

plt.plot(acc_mean, label='Algorithm 1')
plt.fill_between(range(len(acc_mean)), acc_mean, acc_mean + acc_std, color='b', alpha=0.2)
plt.plot(acc_dx_mean, label='CD_ADMM without Multiplier-Coordination')
plt.fill_between(range(len(acc_dx_mean)), acc_dx_mean, acc_dx_mean + acc_dx_std, color='r', alpha=0.2)
plt.plot(acc_dad_mean, label='Distributed Dual Ascent Descent',linewidth=1, color='g',alpha=0.5, zorder=2)
plt.fill_between(range(len(acc_dad_mean)), acc_dad_mean, acc_dad_mean + acc_dad_std, color='g', alpha=0.1, zorder=2)

plt.legend(fontsize=12)
plt.xlabel('Iteration Number',fontsize=14)
plt.ylabel('Relative Error',fontsize=14)
plt.yscale('log')
#plt.title('Relative Difference between Obtained Solution and Optimal Solution')
plt.grid()
plt.show()


inf = [np.mean(np.maximum(arr,0)[:6000],axis=1) for arr in inf]
#inf = [np.mean(arr,axis=1)[:6000] for arr in inf]
inf_mean = np.mean(np.power(np.array(inf),2), axis=0)  # Compute the mean across the 100 sequences for each point
inf_std = 0.6*np.std(inf, axis=0)   # Compute the standard deviation across the 100 sequences for each point
inf_dx = [np.mean(np.power(np.maximum(arr,0)[:6000],2),axis=1) for arr in inf_dx]
inf_dx_mean = np.mean(np.array(inf_dx), axis=0)  # Compute the mean across the 100 sequences for each point 
inf_dx_std = 0.6*np.std(inf_dx, axis=0)   # Compute the standard deviation across the 100 sequences for each point
inf_dad = [np.mean(np.power(np.maximum(arr,0)[:6000],2),axis=1) for arr in inf_dad]
inf_dad_mean = np.mean(np.array(inf_dad), axis=0)  # Compute the mean across the 100 sequences for each point
# Plot the mean with a shaded area representing the standard deviation
plt.figure(1)
plt.plot(inf_mean, label='Algorithm 1')
#plt.fill_between(range(len(inf_mean)), inf_mean - inf_std, inf_mean + inf_std, color='b', alpha=0.2, label='Std Dev')
plt.plot(inf_dx_mean, label='CD_ADMM without Multiplier-Coordination')
plt.plot(inf_dad_mean, label='Distributed Dual Ascent Descent', zorder = 2, alpha=0.8,linewidth=0.5)
#plt.fill_between(range(len(inf_dx_mean)), inf_dx_mean - inf_dx_std, inf_dx_mean + inf_dx_std, color='r', alpha=0.2, label='Std Dev_dx')
plt.legend(fontsize=12)
plt.xlabel('Iteration Number',fontsize=14)
plt.ylabel('Infeasibility',fontsize=14)
plt.yscale('log')
#plt.title('Mean Infeasibility with Standard Deviation')
plt.grid()

plt.show()

# Plotting the infeasibility

grad_gap = [arr[:6000] for arr in grad_gap]
grad_gap_mean = np.mean(np.power(np.array(grad_gap),2), axis=0)  # Compute the mean across the 100 sequences for each point
grad_gap_std = 0.6*np.std(grad_gap, axis=0)   # Compute the standard deviation across the 100 sequences for each point
grad_gap_dx = [arr[:6000] for arr in grad_gap_dx]
grad_gap_dx_mean = np.mean(np.power(np.array(grad_gap_dx),2), axis=0)  # Compute the mean across the 100 sequences for each point   
grad_gap_dx_std = 0.6*np.std(grad_gap_dx, axis=0)   # Compute the standard deviation across the 100 sequences for each point
grad_gap_dad = [arr[:6000] for arr in grad_gap_dad]
grad_gap_dad_mean = np.mean(np.power(np.array(grad_gap_dad),2), axis=0)  # Compute the mean across the 100 sequences for each point
# Plot the mean with a shaded area representing the standard deviation
window_size = 2
grad_gap_mean = np.convolve(grad_gap_mean, np.ones(window_size)/window_size, mode='same')
grad_gap_dx_mean = np.convolve(grad_gap_dx_mean, np.ones(window_size)/window_size, mode='same')
grad_gap_dad_mean = np.convolve(grad_gap_dad_mean, np.ones(window_size)/window_size, mode='same')

plt.figure(2)
plt.plot(grad_gap_mean, label='Algorithm 1')
#plt.fill_between(range(len(grad_gap_mean)), grad_gap_mean - grad_gap_std, grad_gap_mean + grad_gap_std, color='b', alpha=0.2, label='Std Dev')
plt.plot(grad_gap_dx_mean, label='CD_ADMM without Multiplier-Coordination')
#plt.fill_between(range(len(grad_gap_dx_mean)), grad_gap_dx_mean - grad_gap_dx_std, grad_gap_dx_mean + grad_gap_dx_std, color='r', alpha=0.2, label='Std Dev_dx')
plt.plot(grad_gap_dad_mean, label='Distributed Dual Ascent Descent',zorder=2, alpha=0.8, linewidth=0.5)
plt.legend(fontsize=12)
plt.xlabel('Iteration Number',fontsize=14)
plt.ylabel('Gradient Residue',fontsize=14)
# Apply a sliding window average with a window size of 5 to smooth the curves

plt.yscale('log')
#plt.title('Mean Gradient Gap with Standard Deviation')
plt.grid()
plt.show()

acc_delay = []
for i in range(1,6):
    tau = i
    acc_delay.append(np.load('acc_dz_'+str(tau)+'.npy',allow_pickle=True))

acc_delay_t = []
for i in range(1,6):
    tau = i
    acc_delay_t.append(np.load('acc_dz_t_'+str(tau)+'.npy'))
pass

colors = ['b', 'g', 'r', 'c', 'y']
plt.figure(3)
for i in [0,2,4]:
    tau  = i+1
    plt.plot(np.mean(acc_delay_t[i],axis=0), label='Algorithm 1,  $\\tau$ = '+str(tau),color=colors[i])
plt.legend(fontsize=12)
plt.xlabel('Execution time',fontsize=14)
plt.yscale('log')
plt.ylabel('Relative Error',fontsize=14)
plt.xlim(0, 2000)
plt.grid()
plt.show()



plt.figure(4)
for i in [0,2,4]:
    tau = i+1
    length = min(arr.shape for arr in acc_delay[i])[0]
    data = np.array([arr[:length] for arr in acc_delay[i]])
    plt.plot(np.mean(data,axis=0), label='Algorithm 1, $\\tau$ = '+str(tau),  color=colors[i])

plt.legend(fontsize=12)
plt.yscale('log')
plt.xlabel('Iteration number', fontsize=14)
plt.ylabel('Relative Error',fontsize=14)
plt.xlim(0, 2000)
plt.grid()
plt.show()





'''

obj_iter = [np.load('obj_iter_tau5g_t181.69592941946954.npy')\
            ,np.load('obj_iter_tau4g_t181.69592941946954.npy')\
            ,np.load('obj_iter_tau3g_t181.69592941946954.npy')\
            ,np.load('obj_iter_tau2g_t181.69592941946954.npy')\
            ,np.load('obj_iter_tau1g_t181.69592941946954.npy')]
obj_t = [np.load('obj_t_tau5g_t181.69592941946954.npy')\
            ,np.load('obj_t_tau4g_t181.69592941946954.npy')\
            ,np.load('obj_t_tau3g_t181.69592941946954.npy')\
            ,np.load('obj_t_tau2g_t181.69592941946954.npy')\
            ,np.load('obj_t_tau1g_t181.69592941946954.npy')]
plt.figure(1)
colors = ['b', 'g', 'r', 'c', 'y']
for i in range(5):
    plt.plot(np.abs(obj_iter[i]-181.696)/181.696-0.05, label='$\tau$='+str(5-i),color=colors[i])

plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Iteration')
plt.yscale('log')
plt.grid()
plt.show()
plt.figure(2)

for i in range(5):
    plt.plot(np.abs(obj_t[i]-181.696)/181.696-0.05, label='$\\tau$='+str(5-i), color= colors[i])

plt.legend()
plt.xlabel('time step') 
plt.ylabel('Accuracy')
plt.title('Accuracy vs Time')
plt.yscale('log')
plt.grid()
plt.show()


obj_t_N = [np.load('obj_t_N2g_t46.69468853115465.npy'),\
            np.load('obj_t_N4g_t41.97323331390489.npy'),\
            np.load('obj_t_N8g_t37.54527775471204.npy'),\
            np.load('obj_t_N16g_t37.4153355353034.npy')]
opt = np.array([46.69, 41.97323331390489, 37.54527775471204, 37.4153355353034])
plt.figure(3)
for i in range(4):
    plt.plot(1-np.abs(obj_t_N[i]-opt[i])/opt[i], label='N='+str(2**(i+1)))
plt.legend()
plt.xlabel('time step')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Time')
plt.grid()
plt.show()

obj_t = np.load('obj_t_localFalsedzTrueg_t-3.7471797077821063.npy')[:3000]
obj_t_local = np.load('obj_t_localTruedzTrueg_t-3.7471797077821063.npy')[:3000]
obj_t_dx = np.load('obj_t_localFalsedzFalseg_t-3.7471797077821063.npy')[:3000]
plt.figure(4)
plt.plot(np.abs(obj_t-(-3.7471797077821063))/(3.7471797077821063), label='algorithm 1')
plt.plot(np.abs(obj_t_local-(-3.7471797077821063))/(3.7471797077821063), label='local')
plt.plot(np.abs(obj_t_dx-(-3.7471797077821063))/(3.7471797077821063), label='dx')
plt.yscale('log')
plt.legend()
plt.xlabel('time step')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Time')
plt.grid()
plt.show()


obj_t = np.load('obj_t_localFalsedzTrueg_t-9.825292922372583.npy')
obj_t_local = np.load('obj_t_localTruedzTrueg_t-9.825292922372583.npy')
obj_t_dx = np.load('obj_t_localFalsedzFalseg_t-9.825292922372583.npy')
plt.figure(5)
plt.plot(np.abs(obj_t-(-9.825292922372583))/(9.825292922372583), label='algorithm 1')
plt.plot(np.abs(obj_t_local-(-9.825292922372583))/(9.825292922372583), label='local')
plt.plot(np.abs(obj_t_dx-(-9.825292922372583))/(9.825292922372583), label='dx')
plt.yscale('log')
plt.legend()
plt.xlabel('time step')
plt.ylabel('Accuracy (Log Scale)')
plt.title('Accuracy vs Time (Log Scale)')
plt.grid()
plt.show()
'''