import numpy as np
from neuralnetwork.learning.sgd import *
from neuralnetwork.DNN import DNN
from datasets import mnist
from datasets import timit




mnist = mnist.load_mnist_theano('mnist.pkl.gz')
print mnist

rng = np.random.RandomState(1111)
nn = DNN(rng, [3096, 3096], 784, 10)
# bsgd(nn, mnist)




# neural network for triphones
# xnn = DNN(rng, [1024, 1024], 440, 1953)
# nn = DNN(rng, [500, 500], x.shape[1], 10)


# neural network for monophones 

nn = DNN(rng, [3096, 3096, 3096], 429, 144)



train_x, train_y = timit.readTIMIT('timit-mono-mfcc-train.pfile.gz', shared=False, listify=True)
valid_x, valid_y = timit.readTIMIT('timit-mono-mfcc-valid.pfile.gz', shared=True, listify=False)
test_x, test_y = timit.readTIMIT('timit-mono-mfcc-test.pfile.gz', shared=True, listify=False)

train_x, train_y  = timit.make_shared_partitions(train_x, train_y)
num_partitions = len(train_x)
print num_partitions


for i in xrange(num_partitions):
	train_set_x = train_x[i]
	train_set_y = train_y[i]
	train_set_xy = (train_set_x, train_set_y)
	timit = [train_set_xy, (valid_x, valid_y), (test_x, test_y)]
	bsgd(nn, timit)





# valid_x, valid_y = timit.make_shared_partitions(valid_x, valid_y)
# test_x, test_y = timit.make_shared_partitions(test_x, test_y)


#  neural network training with sets made from the main training set on TIMIT.
