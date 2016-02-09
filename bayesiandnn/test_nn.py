import numpy as np
from neuralnetwork.learning.sgd import *
from neuralnetwork.DNN import DNN
from datasets import mnist
from datasets import timit




mnist = mnist.load_mnist_theano('mnist.pkl.gz')
print mnist

rng = np.random.RandomState(1111)
# nn = DNN(rng, [1024, 1024], 784, 10)
# bsgd(nn, mnist)

# xnn = DNN(rng, [1024, 1024], 440, 1953)
# nn = DNN(rng, [500, 500], x.shape[1], 10)
nn = DNN(rng, [2048, 2048, 2048], 429, 144)


train_x, train_y = timit.readTIMIT('timit-mono-mfcc-train.pfile.gz')
valid_x, valid_y = timit.readTIMIT('timit-mono-mfcc-valid.pfile.gz', shared=True)
test_x, test_y = timit.readTIMIT('timit-mono-mfcc-test.pfile.gz', shared=True)



# train_x, train_y  = timit.make_shared_partitions(train_x, train_y)
# valid_x, valid_y = timit.make_shared_partitions(valid_x, valid_y)
# test_x, test_y = timit.make_shared_partitions(test_x, test_y)


timit = timit.make_shared_sets(train_x, train_y)
# train_x, train_y  = timit.make_shared_sets(train_x, train_y)
# valid_x, valid_y = timit.make_shared_partitions(valid_x, valid_y)
# test_x, test_y = timit.make_shared_partitions(test_x, test_y)

# timit = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
# timit = [train_xy, valid_xy, test_xy]

bsgd(nn, timit)
# bsgd_partition(nn, timit)

