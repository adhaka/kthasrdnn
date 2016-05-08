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




mnist = mnist.load_mnist_theano('mnist.pkl.gz')
print mnist

rng = np.random.RandomState(1111)
nn = DNN(rng, [6000, 6000], 784, 10)
bsgd(nn, mnist, epochs=60, percent_data=0.04)

