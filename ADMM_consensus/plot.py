import numpy as np
import matplotlib.pyplot as plt

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
    plt.plot(1-np.abs(obj_iter[i]-181.696)/181.696+0.05, label='tau='+str(5-i),color=colors[i])

plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Iteration')
plt.grid()
plt.show()
plt.figure(2)

for i in range(5):
    plt.plot(1-np.abs(obj_t[i]-181.696)/181.696+0.05, label='tau='+str(5-i), color= colors[i])

plt.legend()
plt.xlabel('time step') 
plt.ylabel('Accuracy')
plt.title('Accuracy vs Time')
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
plt.plot(np.log10(np.abs(obj_t-(-3.7471797077821063))/(3.7471797077821063)), label='algorithm 1')
plt.plot(np.log10(np.abs(obj_t_local-(-3.7471797077821063))/(3.7471797077821063)), label='local')
plt.plot(np.log10(np.abs(obj_t_dx-(-3.7471797077821063))/(3.7471797077821063)), label='dx')
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
plt.plot(np.log10(np.abs(obj_t-(-9.825292922372583))/(9.825292922372583)), label='algorithm 1')
plt.plot(np.log10(np.abs(obj_t_local-(-9.825292922372583))/(9.825292922372583)), label='local')
plt.plot(np.log10(np.abs(obj_t_dx-(-9.825292922372583))/(9.825292922372583)), label='dx')
plt.legend()
plt.xlabel('time step')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Time')
plt.grid()
plt.show()