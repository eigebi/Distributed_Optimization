import numpy as np
from matplotlib import pyplot as plt

acc_source = np.load('acc_source.npy')[:100]
acc_source_10000 = np.load('acc_source_10000.npy')[:100]
acc_target = np.load('acc_target.npy')[:100]
acc_source_TL = np.load('acc_source_TL.npy')[:100]
acc_target_TL = np.load('acc_target_TL.npy')[:100]

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

plt.figure(2)
plt.plot(acc_source,label='data set size 5000')
plt.plot(acc_source_10000,label='dataset size 1000')
plt.grid()
plt.ylim(0.85,1)
plt.xlabel('Iteration')
plt.ylabel('Mean Accuracy')
plt.legend()
plt.title('Source: small data set vs large data set')
plt.show()

plt.figure(3)
plt.plot(acc_source,label='source')
plt.plot(acc_target,label='target')
plt.grid()
plt.ylim(0.85,1)
plt.xlabel('Iteration')
plt.ylabel('Mean Accuracy')
plt.legend()
plt.title('Source model in target domain')
plt.show()

plt.figure(4)
plt.plot(acc_source_TL,label='target model in source domain')
plt.plot(acc_target_TL,label='target model in target domain')
plt.plot(acc_source,label='baseline')
plt.grid()
plt.ylim(0.85,1)
plt.xlabel('Iteration')
plt.ylabel('Mean Accuracy')
plt.legend()
plt.title('Transfer learning')
plt.show()