# @author:Akash
# @package:bayesiandnn

from os import sys, path
sys.path.append(path.dirname(path.dirname( path.abspath(__file__) ) ) )

import numpy as np
from neuralnetwork.learning.sgd import *
from neuralnetwork.DNN import DNN 
from neuralnetwork.GdA import GdA
from datasets import mnist
from datasets import timit
from theano.tensor.shared_randomstreams import RandomStreams 



BATCH_SIZE = 300
NUM_EPOCHS = 40
mnist_data = mnist.load_mnist_numpy('mnist.pkl.gz', percent_data=0.1)
mnist_full = mnist.load_mnist_numpy('mnist.pkl.gz', percent_data=1.0)

print mnist
# train_set_x, train_set_y = mnist_data[0]
valid_set_x, valid_set_y = mnist_data[1]
test_set_x, test_set_y = mnist_data[2]

train_set_x, train_set_y = mnist_full[0]

numpy_rng = np.random.RandomState(1111)

theano_rng = RandomStreams(numpy_rng.randint( 2**30 ))

# nn_ae = DNN(numpy_rng, [1024, 1024], 429, 144)
# configuration for mnist

train_x_label = train_set_x[:3000,:]
train_y_label = train_set_y[:3000]
train_x_unlabel = train_set_x[3000:50000,:] 

nn_ae = DNN(numpy_rng, [1000, 1000], 784, 10)
ae1 = GdA(train_x_label, train_x_unlabel, train_y_label, numpy_rng, theano_rng, [500, 500], nn_ae, mode='contractive', activations_layers=['tanh', 'tanh', 'tanh'])

pretrain_fns = ae1.pretraining_functions(BATCH_SIZE)

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


bsgd(nn_ae, mnist_data, epochs=100, batch_size=100, lr=0.04)


