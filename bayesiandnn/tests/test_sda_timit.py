# @author:Akash
# @package:bayesiandnn

from os import sys, path
sys.path.append(path.dirname(path.dirname( path.abspath(__file__) ) ) )

import numpy as np 
from neuralnetwork.learning.sgd import *
from neuralnetwork.DNN import DNN 
from neuralnetwork.SdA import SdA
from datasets import mnist
from datasets import timit
from theano.tensor.shared_randomstreams import RandomStreams 


def map_y_48(y_inp):
	y_out = map(lambda x: int(x/3.), y_inp)
	return y_out

def map_y_39(y):
	y_48 = map(lambda x: int(x/3.), y)
	y_39 = y_48
	y_39[y_48 == 14] = 27
	y_39[y_48 == 5] = 2
	y_39[y_48 == 3] = 0
	y_39[y_48 == 15] = 29
	y_39[y_48 == 47] = 36
	y_39[y_48 == 23] = 22
	y_39[y_48 == 43] = 37
	y_39[y_48 == 16] = 37
	y_39[y_48 == 9] = 37
	return y_39


BATCH_SIZE = 400
NUM_EPOCHS = 200
# mnist = mnist.load_mnist_theano('mnist.pkl.gz')

numpy_rng = np.random.RandomState(11111)

theano_rng = RandomStreams(numpy_rng.randint( 2**30 ))

# configuration for timit


train_x, train_y = timit.readTIMIT('timit-mono-mfcc-train.pfile.gz', shared=False, listify=True)
valid_x, valid_y = timit.readTIMIT('timit-mono-mfcc-valid.pfile.gz', shared=False, listify=False)
test_x, test_y = timit.readTIMIT('timit-mono-mfcc-test.pfile.gz', shared=False, listify=False)

train_x_all = np.vstack(train_x)
train_y_all = np.hstack(train_y)
# train_y_all = reduce(lambda x,y:x+y, train_y)
train_x_all, train_y_all = timit.shared_dataset((train_x_all, train_y_all))

train_y = map(lambda x: map_y_48(x), train_y)
valid_y, test_y = map_y_48(valid_y), map_y_48(test_y)


train_x, train_y  = timit.make_shared_partitions(train_x, train_y)
valid_x, valid_y = timit.shared_dataset((valid_x, valid_y))
test_x, test_y = timit.shared_dataset((test_x, test_y))

train_set_x = train_x[0]
print train_x_all.get_value().shape[0]
print train_set_x.get_value().shape[0]

# nn_ae = DNN(numpy_rng, [5096, 5096], 429, 144)
nn_ae = DNN(numpy_rng, [6000, 6000], 429, 39)
# nn_ae = DNN(numpy_rng, [6096, 6096], 429, 39)

ae1 = SdA(train_x_all, numpy_rng, theano_rng, [6000, 6000], nn_ae, mode='contractive', activations_layers=['tanh', 'tanh', 'tanh'])

pretrain_fns = ae1.pretraining_functions(train_x_all, BATCH_SIZE)
num_samples_part = train_set_x.get_value(borrow=True).shape[1]
num_samples = train_x_all.get_value(borrow=True).shape[1]

num_batches = num_samples / BATCH_SIZE
indices = np.arange(num_samples, dtype=np.dtype('int32'))


# layer-wise pretraining
for i in xrange(len(ae1.da_layers)):
	for epoch in xrange(NUM_EPOCHS):
		c = []
		for j in xrange(num_batches):
			index = indices[j*BATCH_SIZE:(j+1)*BATCH_SIZE] 
			c.append(pretrain_fns[i](index=index))

		print "pretraining reconstruction error:",i, epoch, np.mean(c)



num_partitions = len(train_x)
print num_partitions


for i in xrange(num_partitions):
	train_set_x = train_x[i]
	train_set_y = train_y[i]
	train_set_xy = (train_set_x, train_set_y)
	timit = [train_set_xy, (valid_x, valid_y), (test_x, test_y)]
	bsgd(nn_ae, timit, epochs=8, lr=0.006)



