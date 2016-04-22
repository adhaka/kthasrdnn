import numpy as np
from neuralnetwork.learning.sgd import *
from neuralnetwork.DNN import DNN
from neuralnetwork.LadderAE import LadderAE
from theano.tensor.shared_randomstreams import RandomStreams
from datasets import mnist
from datasets import timit


mnist = mnist.load_mnist_theano('mnist.pkl.gz')
print mnist

train_set_x, train_set_y = mnist[0]
valid_set_x, valid_set_y = mnist[1]
test_set_x, test_set_y = mnist[2]

numpy_rng = np.random.RandomState(1111)
theano_rng = RandomStreams(numpy_rng.randint( 2**30 ))

train_x_label = train_set_x[:20000,:]
train_y_label = train_set_y[:20000]
train_x_unlabel = train_set_x[20000:30000,:] 

LadderNetwork = LadderAE(rng, train_x_label, train_x_unlabel, train_y_label, [784, 500, 500, 100, 50, 10], )

# nn = DNN(rng, [3096, 3096], 784, 10)
bsgd(nn, mnist, epochs=40)

