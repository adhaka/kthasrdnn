# @author:akash
# package:kthasrdnn

# plot alpha for mnist
import numpy as np 
import argparse
from matplotlib import pyplot as plt

x = np.asarray([100, 500, 600, 1000, 3000])
y = np.asarray([50, 300, 300, 600, 1400])

# data = np.array([[100, 500, 600, 1000, 3000],
	# [50, 300, 300, 600, 1400]
	# ])


# print data 
# plt.plot(data[0:], data[1:])
plt.plot(x, y, '-o', label='alpha')
plt.xlim((0, 5000))
plt.ylim((0, 2000))
plt.legend(loc='best')
plt.xlabel('# labeled examples')
plt.ylabel('alpha optimised')
plt.title('MNIST-alpha')
plt.savefig('mnist-alpha.pdf')