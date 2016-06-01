# @author:Akash
# @package:tmhasrdnn

from os import sys, path
sys.path.append(path.dirname(path.dirname( path.abspath(__file__))))


import numpy as np 
import argparse
import theano
import theano.tensor as T 
from theano.tensor.shared_randomstreams import RandomStreams
import math 
from utils.utils import one_of_K_encoding
from neuralnetwork.SSDAE import SSDAE
from datasets import timit


NUM_EPOCHS = 16
NUM_CLASSES = 48
NUM_CLASSES = 39
BATCH_SIZE = 400

numpy_rng = np.random.RandomState(11111)
theano_rng = RandomStreams(numpy_rng.randint( 2**30 ))

parser = argparse.ArgumentParser(description='setting up hyperparams by command line.')
parser.add_argument('--percent', '-p', type=float, default=0.999)
parser.add_argument('--beta', '-b', type=int, default=200)
parser.add_argument('--alpha', '-a', type=int, default=3)

args = parser.parse_args()
alpha = args.alpha
beta = args.beta
percent = args.percent

# parser.add_argument('')

print "fraction of training data used is:", percent
train_x_lab, train_y_lab, train_x_unlab = timit.readTIMITSSL('timit-mono-mfcc-train.pfile.gz', shared=False, listify=True, mapping=48, randomise=True, percent_data=percent)
# train_x_lab, train_y_lab, train_x_unlab = timit.readTIMITSSL('timit-mono-mfcc-train.pfile.gz', shared=False, listify=True, mapping=48, randomise=True, percent_data=0.9999)

valid_x, valid_y = timit.readTIMIT('timit-mono-mfcc-valid.pfile.gz', shared=False, listify=False, mapping=48)
test_x, test_y = timit.readTIMIT('timit-mono-mfcc-test.pfile.gz', shared=False, listify=False, mapping=48)

# train_y_all = reduce(lambda x,y:x+y, train_y)

# train_x, train_y  = timit.make_shared_partitions(train_x, train_y)
# train_set_x = train_x[0]
# print train_x_all.get_value().shape[0]

# train_x_label = train_x_all[:30000, :]
# train_y_label = train_x_all[:30000, :]
# train_x_unlabel = train_y_all[30000:90000,:]
print valid_x.shape

network = SSDAE(numpy_rng, [7000, 7000], train_x_lab, train_y_lab, train_x_unlab, alpha=alpha, beta=beta)
network.trainSGD(epochs = [70, 2])
network.trainSGDSupervised(train_x_lab, train_y_lab, valid_x, valid_y, test_x, test_y)
