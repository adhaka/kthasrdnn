# author:akash
# package:bayesiandnn


import cPickle, gzip
import os
import numpy as np
import theano
import math
import random
from collections import Counter
from utils.utils import *
from theano import tensor as T

BATCH_SIZE = 300
NUM_CLASSES = 10

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



def load_mnist_numpy(dataset, binarize=True):
	train_set, valid_set, test_set = _load_data(dataset)
	ts_x, ts_y = train_set
	va_x, va_y = valid_set
	te_x, te_y = test_set
	# ts_y_bin = one_of_K_encoding(ts_y)
	# va_y_bin = one_of_K_encoding(va_y)
	# te_y_bin = one_of_K_encoding(te_y)

	# if binarize == True:
		# return [(ts_x, ts_y_bin), (va_x, va_y_bin), (te_y_bin)]

	return [(ts_x, ts_y), (va_x, va_y), (te_x, te_y)] 



def load_mnist_theano(dataset):
	# if not os.path.exists(dataset):
	# 	raise Exception('file path error')
	train_set, valid_set, test_set = _load_data(dataset)

	def shared_dataset(data_xy, borrow = True):
		data_x, data_y = data_xy
		shared_x = theano.shared(np.asarray(data_x, dtype= np.float32), borrow=borrow)
		shared_y = theano.shared(np.asarray(data_y, dtype= np.float32), borrow=borrow)

		return shared_x, T.cast(shared_y, 'int32')


	train_set_x, train_set_y = shared_dataset(train_set)
	valid_set_x, valid_set_y = shared_dataset(valid_set)
	test_set_x, test_set_y = shared_dataset(test_set)
	print train_set_x.get_value(borrow=True).shape[0]

	batch_size = BATCH_SIZE

	return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]



# @TODO: very crude way of normalising the dataset, needs better logic

#  create semi-supervised sets of labelled and unlabelled data 
#  controlled by the percent parameter.



# def load_mnist_ssl1(dataset, percent = 0.70):
# 	train_set, valid_set, test_set = _load_data(dataset)	

# 	x,y = train_set
#     n_x = x[0].shape[0]
#     n_classes = y[0].shape[0]
#     if n_labeled%n_classes != 0: raise("n_labeled (wished number of labeled samples) not divisible by n_classes (number of classes)")
#     n_labels_per_class = n_labeled/n_classes
#     x_labeled = [0]*n_classes
#     x_unlabeled = [0]*n_classes
#     y_labeled = [0]*n_classes
#     y_unlabeled = [0]*n_classes
#     for i in range(n_classes):
#         idx = range(x[i].shape[1])
#         random.shuffle(idx)
#         x_labeled[i] = x[i][:,idx[:n_labels_per_class]]
#         y_labeled[i] = y[i][:,idx[:n_labels_per_class]]
#         x_unlabeled[i] = x[i][:,idx[n_labels_per_class:]]
#         y_unlabeled[i] = y[i][:,idx[n_labels_per_class:]]
#     return np.hstack(x_labeled), np.hstack(y_labeled), np.hstack(x_unlabeled), np.hstack(y_unlabeled)



def load_mnist_ssl(dataset, percent = 0.50):
	train_set, valid_set, test_set = _load_data(dataset)
	
	ts_x, ts_y = train_set
	# balance the dataset such that each class has the same no of input
	num_cls = np.max(ts_y, axis = 0)
	num_points = ts_x.shape[0]
	num_labelled = int(math.ceil(percent * num_points))
	num_unlabelled = num_points - num_labelled
	ts_y = ts_y[:, np.newaxis]

	# xy_comb = np.hstack((ts_x, ts_y))
	# np.random.shuffle(xy_comb)
	# ts_x, ts_y = xy_comb[:,:xy_comb.shape[1] -1], xy_comb[:, xy_comb.shape[1]-1:]
	# print ts_y
	# idxcnt_cls = math.ceil(num_labelled / num_cls)

	ts_lab_x = ts_x[:num_labelled,:]
	ts_lab_y = ts_y[:num_labelled,:]
	ts_unlab_x, ts_unlab_y = ts_x[num_labelled:,:], ts_y[num_labelled, :]
	return [ts_unlab_x, ts_lab_y, ts_unlab_x, ts_unlab_y], valid_set, test_set 


	def balancedataset(x, y):
		min_dp_cl = min(Counter(y).values())
		max_dp_cl = max(Counter(y).values())
		diff = abs(max_dp_cl - min_dp_cl)
		threshold = math.ceil(0.15 * max_dp_cl)
		if diff >= threshold:
			raise Exception("datasets are unbalanced.")






