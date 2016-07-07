# @author:akash
# package:kthasrdnn


import numpy as np
from matplotlib import pyplot as plt


data = np.array([
    [1, 10688, 57.46, 57.93, 59.65, 59.84],
    [3, 32065, 61.71, 61.31, 64.12, 64.20],
    [5, 53441, 63.20, 63.30, 65.44, 65.71],
    [10, 106881, 65.78, 65.82, 66.96, 67.03],
    [20, 213763, 68.02, 67.80, 69.31,  69.18],
    [30, 320644, 69.08, 68.83, np.inf, np.inf],
    [50, 534408, 70.34, 69.71, np.inf, np.inf]
    # [100, 71.89, 71.54, np.inf, np.inf]
], dtype='float32')

test_set_delta = data[:,5] - data[:,3]

plt.figure(1)
plt.plot(data[:,0], data[:,2], '-o', label='NN valid')
plt.plot(data[:,0], data[:,3], '-^', label='NN test')
plt.plot(data[:,0], data[:,4], '-s', label='SSEAE valid')
plt.plot(data[:,0], data[:,5], '-v', label='SSEAE test')
plt.legend(loc='best')
plt.xlabel('% labeled examples')
plt.ylabel('accuracy')
plt.title('TIMIT')
plt.savefig('timit.pdf')

plt.figure(2)
plt.plot(data[:,0], test_set_delta, '-o', label='% difference in accuracy')
plt.legend(loc='best')
plt.xlim((0, 15))
plt.ylim((0, 4))
plt.xlabel('% labelled examples')
plt.ylabel('accuracy')
plt.title('TIMIT-difference')
plt.savefig('timit-diff.pdf')
