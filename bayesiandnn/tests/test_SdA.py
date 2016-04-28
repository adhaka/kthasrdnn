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



BATCH_SIZE = 400
NUM_EPOCHS = 200
mnist = mnist.load_mnist_theano('mnist.pkl.gz')


print mnist
train_set_x, train_set_y = mnist[0]
valid_set_x, valid_set_y = mnist[1]
test_set_x, test_set_y = mnist[2]

numpy_rng = np.random.RandomState(1111)

theano_rng = RandomStreams(numpy_rng.randint( 2**30 ))

# nn_ae = DNN(numpy_rng, [1024, 1024], 429, 144)
# configuration for mnist

nn_ae = DNN(numpy_rng, [4096, 4096], 784, 10)
ae1 = SdA(train_set_x, numpy_rng, theano_rng, [4096, 4096], nn_ae)

pretrain_fns = ae1.pretraining_functions(train_set_x, BATCH_SIZE)

num_samples = train_set_x.get_value(borrow=True).shape[1]
num_batches = num_samples / BATCH_SIZE
indices = np.arange(num_samples, dtype=np.dtype('int32'))

# layer-wise pretraining

for i in xrange(len(ae1.da_layers)):
	for epoch in xrange(NUM_EPOCHS):
		c = []
		for i in xrange(num_batches):
			index = indices[i*BATCH_SIZE:(i+1)*BATCH_SIZE] 
			c.append(pretrain_fns[i](index=index))

		print "pretraining reconstruction error:",i, epoch, np.mean(c)


bsgd(nn_ae, mnist)


