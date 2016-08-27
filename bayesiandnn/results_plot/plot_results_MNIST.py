# @author:akash
# package:kthasrdnn


import numpy as np
import argparse
from matplotlib import pyplot as plt



data = np.array([
    [0.2, 100, 75.8, 74.04, 77.6, 76.2],
    [1, 500, 87.01, 86.39, 89.21, 88.1],
    [1.2, 600, 87.56, 86.64, 89.53, 89.35],
    [2, 1000, 89.47, 89.14, 91.43, 91.07],
    [6, 3000, 92.61, 92.41, 94.24, 93.8],
    # [100, 50000, 97.46, 97.54, np.inf, np.inf]
])

test_set_del = data[:,5] - data[:,3]

plt.figure(1)
plt.plot(data[:,1], data[:,2], '-o', label='NN valid')
plt.plot(data[:,1], data[:,3], '-^', label='NN test')
plt.plot(data[:,1], data[:,4], '-s', label='SSSAE valid')
plt.plot(data[:,1], data[:,5], '-v', label='SSSAE test')
plt.xlim(0,4000)
plt.legend(loc='best')
plt.xlabel('# labeled examples')
plt.ylabel('accuracy')
plt.title('MNIST')

plt.savefig('mnist-2.pdf')

plt.figure(2)
plt.plot(data[:,1], test_set_del, '-o', label="difference between NN and SSSAE")
plt.legend(loc='best')
plt.xlabel('# labeled examples')
plt.ylabel('difference')
plt.title('mnist-difference')
plt.savefig('mnist-diff.pdf')

