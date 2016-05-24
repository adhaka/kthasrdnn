# @author:Akash
# @package:bayesiandnn

from os import sys, path
sys.path.append(path.dirname(path.dirname( path.abspath(__file__) ) ) )

import numpy as np
from neuralnetwork.learning.sgd import *
from dataio.pfileio import PfileIO
from neuralnetwork.DNN import DNN
from datasets import mnist
from datasets import timit




mnist = mnist.load_mnist_theano('mnist.pkl.gz', percent_data=0.01)
print mnist

rng = np.random.RandomState(1111)
nn = DNN(rng, [100], 784, 10)
bsgd(nn, mnist, epochs=90, batch_size=20, lr=0.08)

