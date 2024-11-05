import numpy as np
from matplotlib import pyplot as plt

a = np.load("obj_train.npy")
b = np.load("obj_train_no_bound.npy")

plt.figure()
plt.plot(a)
plt.plot(b)
plt.show()