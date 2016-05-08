# @author:Akash
# @package:bayesiandnn

from os import sys, path
sys.path.append(path.dirname(path.dirname( path.abspath(__file__) ) ) )


import numpy as np
import theano
import theano.tensor as T 
from theano.tensor.shared_randomstreams import RandomStreams
import math
from utils.utils import one_of_K_encoding
from neuralnetwork.SSDAE import SSDAE
from datasets import mnist



NUM_EPOCHS = 100
NUM_CLASSES = 10
mnist = mnist.load_mnist_numpy('mnist.pkl.gz', binarize=True)

# print mnist

train_set_x, train_set_y = mnist[0]
valid_set_x, valid_set_y = mnist[1]
test_set_x, test_set_y = mnist[2]

numpy_rng = np.random.RandomState(1111)

theano_rng = RandomStreams(numpy_rng.randint( 2**30 ))

# print train_set_x[1000]
train_x_label = train_set_x[:2000,:]
train_y_label = train_set_y[:2000]
train_x_unlabel = train_set_x[2000:30000,:] 

# train_y_label = one_of_K_encoding(train_y_label, NUM_CLASSES)

# train_x_unlabel = np.zeros((1000, train_x_label.shape[1]), dtype='float32')
network = SSDAE(numpy_rng, [7000, 7000, 7000], train_x_label, train_y_label, train_x_unlabel)
# train_fns = network.get_training_functions()
network.trainSGD()
network.trainSGDSupervised()





