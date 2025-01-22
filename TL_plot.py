import numpy as np
from matplotlib import pyplot as plt

acc_source = np.load('acc_source.npy')
acc_target = np.load('acc_target.npy')

plt.figure(1)
plt.plot(acc_source,label='source')
plt.plot(acc_target,label='target')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Mean Accuracy')
plt.show()