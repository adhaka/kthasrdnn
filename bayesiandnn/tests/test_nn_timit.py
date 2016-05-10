# @author:Akash
# @package:bayesiandnn

from os import sys, path
sys.path.append(path.dirname(path.dirname( path.abspath(__file__) ) ) )

import numpy as np
from neuralnetwork.learning.sgd import *
from neuralnetwork.DNN import DNN
from datasets import mnist
from datasets import timit
from theano.tensor.shared_randomstreams import RandomStreams 
import theano
import theano.tensor as T 



def map_y_48(y_inp):
	# y = T.matrix('y')
	# z = T.scalar('z')
	# y_out = y / z
	# f = theano.function(inputs=[y, z], outputs=y_out)
	# out =  f(y_inp, 3)

	y_out = map(lambda x: int(x/3.), y_inp)
	return y_out



numpy_rng = np.random.RandomState(1111)
theano_rng = RandomStreams(numpy_rng.randint( 2**30 ))

# neural network for monophones 
nn = DNN(numpy_rng, [5096, 5096, 5096], 429, 144)
# nn = DNN(numpy_rng, [4096, 4096], 429, 48)
MODE = 'usevalid'

train_x, train_y = timit.readTIMIT('timit-mono-mfcc-train.pfile.gz', shared=False, listify=True)
valid_x, valid_y = timit.readTIMIT('timit-mono-mfcc-valid.pfile.gz', shared=False, listify=False)
test_x, test_y = timit.readTIMIT('timit-mono-mfcc-test.pfile.gz', shared=False, listify=False)

train_y = map(lambda x: map_y_48(x), train_y)
valid_y, test_y = map_y_48(valid_y), map_y_48(test_y)

# this mode uses the standard validation set 
if MODE == 'usevalid':
	train_x, train_y  = timit.make_shared_partitions(train_x, train_y)
	valid_x, valid_y = timit.shared_dataset((valid_x, valid_y))
	test_x, test_y = timit.shared_dataset((test_x, test_y))
	num_partitions = len(train_x)
	print num_partitions


	for i in xrange(num_partitions):
		train_set_x = train_x[i]
		train_set_y = train_y[i]
		train_set_xy = (train_set_x, train_set_y)
		timit = [train_set_xy, (valid_x, valid_y), (test_x, test_y)]
		bsgd(nn, timit)

else:
	print len(train_x)
	num_partitions = len(train_x)
	train_x_mat = np.vstack(train_x)
	train_y_mat = np.hstack(train_y)	
	# train_y_mat = np.concatenate()
	# train_x_mat = np.concatenate((train_x[0], train_x[1], train_x[2]), axis=0)
	# train_y_mat = np.concatenate((train_y[0], train_y[1], train_y[2]), axis=0)


	# train_y_mat = np.vstack(train_y)
	print train_x_mat.shape

	timit_mat = timit.make_sets(train_x_mat, train_y_mat)
	train_xy_mat, valid_xy, test_xy = timit_mat[0], timit_mat[1], timit_mat[2]
	train_x_mat, train_y_mat = train_xy_mat 

	valid_x_np, valid_y_np = valid_xy
	test_x_np, test_y_np = test_xy

	valid_x, valid_y = timit.shared_dataset(valid_xy)
	test_x, test_y = timit.shared_dataset(test_xy)
	num_rows = train_x_mat.shape[0]
	# print train_x_mat[1000,:]

	# split the train_x_mat into three equal parts
	train_x_list = []
	train_y_list = []

	partition_index = int(num_rows / num_partitions)

	for i in xrange(num_partitions):
		x = train_x_mat[i*partition_index:(i+1)*partition_index, :]
		y = train_y_mat[i*partition_index:(i+1)*partition_index]
		train_x_list.append(x)
		train_y_list.append(y)

		xy = (x,y)
		shared_x, shared_y = timit.shared_dataset(xy)
		td = [(shared_x, shared_y), (valid_x, valid_y), (test_x, test_y)]
		bsgd(nn, td, [valid_y_np, test_y_np], epochs=9)


