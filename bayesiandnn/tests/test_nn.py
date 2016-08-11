# @author:Akash
# @package:bayesiandnn

from os import sys, path
sys.path.append(path.dirname(path.dirname( path.abspath(__file__) ) ) )

import numpy as np
import argparse
from neuralnetwork.learning.sgd import *
from dataio.pfileio import PfileIO
from neuralnetwork.DNN import DNN
from datasets import mnist
from datasets import timit


parser = argparse.ArgumentParser(description='setting up hyperparameters by command line ...')
parser.add_argument('--percent', '-p', type=float, default=0.9999)
parser.add_argument('--lr', '-l', type=float, default=0.08)
args = parser.parse_args()
percent = args.percent
lr = args.lr


mnist = mnist.load_mnist_theano('mnist.pkl.gz', percent_data=percent)
# print mnist

rng = np.random.RandomState(1111)
nn = DNN(rng, [100], 784, 10)
bsgd(nn, mnist, epochs=15, batch_size=20, lr=lr)

