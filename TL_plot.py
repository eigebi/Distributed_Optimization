import numpy as np
from matplotlib import pyplot as plt

acc_source = np.load('acc_source.npy')
acc_source_10000 = np.load('acc_source_10000.npy')
acc_target = np.load('acc_target.npy')
acc_source_TL = np.load('acc_source_TL.npy')
acc_target_TL = np.load('acc_target_TL.npy')

plt.figure(1)
plt.plot(acc_source[:100],label='source_1000')
#plt.plot(acc_target[:100],label='target')
plt.plot(acc_source_10000[:100],label='source_10000')
plt.plot(acc_source_TL[:100],label='source_TL_10000')
plt.plot(acc_target_TL[:100],label='target_TL')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Mean Accuracy')
plt.show()