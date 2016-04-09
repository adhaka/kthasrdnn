import numpy as np
from neuralnetwork.learning.sgd import *
from neuralnetwork.DNN import DNN
from datasets import mnist
from datasets import timit




mnist = mnist.load_mnist_theano('mnist.pkl.gz')
print mnist

rng = np.random.RandomState(1111)
nn = DNN(rng, [3096, 3096], 784, 10)
bsgd(nn, mnist, epochs=40)

