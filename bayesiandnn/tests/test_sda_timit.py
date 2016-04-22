import numpy as np 
from neuralnetwork.learning.sgd import *
from neuralnetwork.DNN import DNN 
from neuralnetwork.SdA import SdA
from datasets import mnist
from datasets import timit
from theano.tensor.shared_randomstreams import RandomStreams 



BATCH_SIZE = 400
NUM_EPOCHS = 100
# mnist = mnist.load_mnist_theano('mnist.pkl.gz')

numpy_rng = np.random.RandomState(11111)

theano_rng = RandomStreams(numpy_rng.randint( 2**30 ))

# configuration for timit
nn_ae = DNN(numpy_rng, [2096, 2096, 2096], 429, 144)


train_x, train_y = timit.readTIMIT('timit-mono-mfcc-train.pfile.gz', shared=False, listify=True)
valid_x, valid_y = timit.readTIMIT('timit-mono-mfcc-valid.pfile.gz', shared=True, listify=False)
test_x, test_y = timit.readTIMIT('timit-mono-mfcc-test.pfile.gz', shared=True, listify=False)

train_x_all = np.vstack(train_x)
train_y_all = np.hstack(train_y)
# train_y_all = reduce(lambda x,y:x+y, train_y)
train_x_all, train_y_all = timit.shared_dataset((train_x_all, train_y_all))

train_x, train_y  = timit.make_shared_partitions(train_x, train_y)
train_set_x = train_x[0]
print train_x_all.get_value().shape[0]
print train_set_x.get_value().shape[0]

ae1 = SdA(train_x_all, numpy_rng, theano_rng, [500, 500, 500], nn_ae, ['tanh', 'tanh', 'tanh'])

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
	bsgd(nn_ae, timit)



