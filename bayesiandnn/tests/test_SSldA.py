# @author:Akash
# @package:tmhasrdnn

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


parser = argparse.ArgumentParser(description='setting up hyperparams by command line.')
parser.add_argument('--percent', '-p', type=float, default=0.999)
parser.add_argument('--beta', '-b', type=int, default=200)
parser.add_argument('--alpha', '-a', type=int, default=3)

args == parser.parse_args()
alpha = args.alpha
beta = args.beta
percent = args.percent

# print train_set_x[1000]
train_x_label = train_set_x[:3000,:]
train_y_label = train_set_y[:3000]
train_x_unlabel = train_set_x[3000:50000,:] 

# train_y_label = one_of_K_encoding(train_y_label, NUM_CLASSES)

# train_x_unlabel = np.zeros((1000, train_x_label.shape[1]), dtype='float32')
network = SSDAE(numpy_rng, [3000, 3000], train_x_label, train_y_label, train_x_unlabel)
# train_fns = network.get_training_functions()
network.trainSGD(epochs=[100, 1])
network.trainSGDSupervised(train_x_label, train_y_label, valid_set_x, valid_set_y, test_set_x, test_set_y)





