# author:akash
# package:bayesiandnn


import cPickle, gzip
import os
import numpy as np
import theano
import math
import random
from Collections import Counter
from theano import tensor as T

BATCH_SIZE = 300

# load dataset mnist

#load data with path as dataset, could be absolute and relative at the same time. 

def _load_data(dataset):
	dirpath, filename = os.path.split(dataset)

	if dirpath == '' and not os.path.isfile(dataset):
		new_dirpath = os.path.join(
			os.path.split(__file__)[0],
			'rawdata/',
			filename
			)

		if os.path.isfile(new_dirpath) or dataset == 'mnist.pkl.gz':
			dataset = new_dirpath


	f = gzip.open(dataset, 'rb')
	train_set, valid_set, test_set = cPickle.load(f)
	return train_set, valid_set, test_set



def load_mnist_theano(dataset):
	# if not os.path.exists(dataset):
	# 	raise Exception('file path error')
	train_set, valid_set, test_set = _load_data(dataset)

	def shared_dataset(data_xy, borrow = True):
		data_x, data_y = data_xy
		shared_x = theano.shared(np.asarray(data_x, dtype= theano.config.floatX), borrow=borrow)
		shared_y = theano.shared(np.asarray(data_y, dtype= theano.config.floatX), borrow=borrow)

		return shared_x, T.cast(shared_y, 'int32')


	train_set_x, train_set_y = shared_dataset(train_set)
	valid_set_x, valid_set_y = shared_dataset(valid_set)
	test_set_x, test_set_y = shared_dataset(test_set)


	batch_size = BATCH_SIZE

	return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]



# @TODO: very crude way of normalising the dataset, needs better logic

#  create semi-supervised sets of labelled and unlabelled data 
#  controlled by the percent parameter.



def load_mnist_ssl(dataset, percent = 0.50):
	train_set, valid_set, test_set = _load_data(dataset)	
	ts_x, ts_y = train_set_x

	# balance the dataset such that each class has the same no of input
	# num_cls = np.max(ts_y, axis = 0)
	num_cls = ts_y[0].shape[0]

	# number of data points per class.

	num_points = ts_x[0].shape[0]
	num_labelled = math.ceil(percent * num_points)
	num_unlabelled = num_points - num_labelled

	ts_lab_x = np.zeros(num_cls)
	ts_lab_y = np.zeros(num_cls)
	ts_unlab_x = np.zeros(num_cls)
	ts_unlab_y = np.zeros(num_cls)


	for i in range(num_cls):
		idx = range(ts_lab_x[i].shape[1])
		np.random.shuffle(idx)
		ts_lab_x[i] = ts_lab_x[i][:, idx[:, num_labelled]]
		ts_lab_y[i] = ts_lab_y[i][:, ix[:, num_labelled]]
		ts_unlab_x[i] = ts_lab_x[i][:, idx[num_labelled:]]
		ts_unlab_y[i] = ts_lab_y[i][:, idx[num_labelled:]]

	return np.hstack(ts_lab_x), np.hstack(ts_lab_y), np.hstack(ts_unlab_x), np.hstack(ts_unlab_y)


# def load_mnist_ssl(dataset, percent = 0.50):
# 	train_set, valid_set, test_set = _load_data(dataset)
	
# 	ts_x, ts_y = train_set_x

# 	# balance the dataset such that each class has the same no of input
# 	num_cls = np.max(ts_y, axis = 0)
# 	num_points = ts_x.shape[0]
# 	num_labelled = math.ceil(percent * num_points)
# 	num_unlabelled = num_points - num_labelled

# 	xy_comb = np.hstack((ts_x, ts_y), dtype=ts_x.dtype)
# 	np.random.shuffle(xy_comb)
# 	ts_x, ts_y = xy_comb[:xy_comb.shape[1] -1], xy_comb[xy_comb.shape[1]-1,:]
# 	idxcnt_cls = math.ceil(num_labelled / num_cls)
# 	ts_lab_x, ts_lab_y = ts_x[:num_labelled,:], ts_y[:num_labelled, :]
# 	ts_unlab_x, ts_unlab_y = ts_x[num_labelled:,:], ts_y[num_labelled, :]
# 	return ts_unlab_x, ts_lab_y, ts_unlab_x, ts_unlab_y


	def balancedataset(x, y):
		min_dp_cl = min(Counter(y).values())
		max_dp_cl = max(Counter(y).values())
		diff = abs(max_dp_cl - min_dp_cl)
		threshold = math.ceil(0.15 * max_dp_cl)
		if diff >= threshold:
			raise Exception("datasets are unbalanced.")






