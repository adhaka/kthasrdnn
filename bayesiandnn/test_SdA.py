import numpy as np 
from neuralnetwork.learning.sgd import *
from neuralnetwork.DNN import DNN 
from neuralnetwork.AutoEncoder import SdA
from datasets import mnist
from datasets import timit
from theano.tensor.shared_randomstreams import RandomStreams 



mnist = mnist.load_mnist_theano('mnist.pkl.gz')
print mnist
train_set_x, train_set_y = mnist[0]
valid_set_x, valid_set_y = mnist[1]
test_set_x, test_set_y = mnist[2]

numpy_rng = np.random.RandomState(1111)

theano_rng = RandomStreams(numpy_rng.randint( 2**30 ))

# nn_ae = DNN(numpy_rng, [1024, 1024], 429, 144)
# configuration for mnist

nn_ae = DNN(numpy_rng, [1024, 1024], 784, 10)
ae1 = SdA(train_set_x, numpy_rng, theano_rng, [1024, 1024], nn_ae)

pretrain_fns = ae1.pretraining_functions()

