# @author:akash
# package:kthasrdnn

# plot alpha values for timit
import numpy as np 
import argparse
from matplotlib import pyplot as plt

x = np.asarray([1, 3, 5, 10, 20])
y = np.asarray([90, 150, 160, 400, 600])

# data = np.array([[100, 500, 600, 1000, 3000],
	# [50, 300, 300, 600, 1400]
	# ])


# print data 
# plt.plot(data[0:], data[1:])
plt.plot(x, y, '-o', label='alpha')
plt.xlim((0, 50))
plt.ylim((0, 1000))
plt.legend(loc='best')
plt.xlabel('% labeled examples')
plt.ylabel('alpha optimised')
plt.title('TIMIT-alpha')
plt.savefig('timit-alpha.pdf')