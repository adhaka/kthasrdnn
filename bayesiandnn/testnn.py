import numpy as np
from neuralnetwork.learning.sgd import *
from neuralnetwork.DNN import DNN
from datasets import mnist
from datasets import timit




mnist = mnist.load_mnist_theano('mnist.pkl.gz')
print mnist

rng = np.random.RandomState(1111)
nn = DNN(rng, [1024, 1024], 440, 1953)
# bsgd(nn, mnist)

#xnn = DNN(rng, [1024, 1024, 1024], 440, 1953)
# nn = DNN(rng, [500, 500], x.shape[1], 10)
nn = DNN(rng, [1024, 1024], 429, 144)


train_x, train_y = timit.readTIMIT('timit-mono-mfcc-train.pfile.gz')
valid_x, valid_y = timit.readTIMIT('timit-mono-mfcc-valid.pfile.gz')
test_x, test_y = timit.readTIMIT('timit-mono-mfcc-test.pfile.gz')



train_x, train_y  = timit.make_shared_partitions(train_x, train_y)
valid_x, valid_y = timit.make_shared_partitions(valid_x, valid_y)
test_x, test_y = timit.make_shared_partitions(test_x, test_y)

timit = [(train_x, train_y), (valid_x[0], valid_y[0]), (test_x[0], test_y[0])]

# bsgd(nn, timit)
bsgd_partition(nn, timit)

