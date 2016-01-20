# author:akash
# package:bayesiandnn


import cPickle, gzip
import os
import numpy as np
import theano
from theano import tensor as T

BATCH_SIZE = 300

# load dataset mnist

#load data with path as dataset, could be absolute and relative at the same time. 
def load_mnist(dataset):

	if not os.path.exists(dataset):
		raise Exception('file path error')

	dirpath, filename = os.path.split(dataset)

	if dirpath == '' and not os.path.isfile(dataset):
		new_dirpath = os.path.join(
			os.path.split(__file__)[0],
			'../datasets/',
			filename
			)

		if os.path.isfile(new_dirpath) or dataset == 'mnist.pkl.gz':
			dataset = new_dirpath



	f = gzip.open(dataset, 'rb')
	train_set, valid_set, test_set = cPickle.load(f)


	def shared_dataset(data_xy, borrow = True):
		data_x, data_y = data_xy
		shared_x = theano.shared(np.asarray(data_x, dtype= theano.config.floatX), borrow=borrow)
		shared_y = theano.shared(np.asarray(data_y, dtype= theano.config.floatX), borrow=borrow)

		return shared_x, T.cast(shared_y, 'int32')


	test_set_x, test_set_y = shared_dataset(train_set)
	valid_set_x, valid_set_y = shared_dataset(valid_set)
	test_set_x, test_set_y = shared_dataset(test_set)


	batch_size = BATCH_SIZE

	return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]




def load_tidigits():
	td = gzip.open('tidigits_examples.npz')['tidigits']




def load_timit():
	pass

