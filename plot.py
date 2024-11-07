import numpy as np
from matplotlib import pyplot as plt

L1 = np.load('L_truth_convex_-10.47_n=10.npy')
L2 = np.load('L_truth_convex_-138.16.npy')
L3 = np.load('L_truth_non_convex_-182.7.npy')

O1 = np.load('obj_train_convex_-10.47_n=10.npy')
O2 = np.load('obj_train_convex_-138.16.npy')
O3 = np.load('obj_train_non_convex_-182.7.npy')


plt.figure(1)
plt.plot(L1.reshape(-1),label='Lagrange function')
plt.plot(O1,label='Objective Function')
plt.plot(np.ones_like(O1)*-10.47,label='Optimal Value')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('Convex Function 2X scaled')
plt.show()

plt.figure(2)
plt.plot(L2.reshape(-1),label='Lagrange function')
plt.plot(O2,label='Objective Function')
plt.plot(np.ones_like(O2)*-138.16,label='Optimal Value')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('Convex Function')
plt.show()

plt.figure(3)
plt.plot(L3.reshape(-1),label='Lagrange function')
plt.plot(O3,label='Objective Function')
plt.plot(np.ones_like(O3)*-182.7,label='Optimal Value')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('Non-Convex Function')
plt.show()