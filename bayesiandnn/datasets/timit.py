# @author:Akash
# @package:tmhasrdnn

import numpy as np 
import cPickle, gzip
import os
import theano
import math
# import random
from collections import Counter
from theano import tensor as T
from os import sys, path

sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from dataio.pfileio import PfileIO
# from .. import dataio.PickleIO as PickleIO
# 

sys.path.append('~/masters-thesis/bayesiandnn/bayesiandnn/datasets/rawdata')
SEED = 139589

def _load_raw_data(datapath):
	dirpath, filename = os.path.split(datapath)

	if dirpath == '' and not os.path.isfile(datapath):
		new_dirpath = os.path.join(
			os.path.split(__file__)[0],
			'datasets/rawdata/',
			filename
			)

		if os.path.isfile(new_dirpath) or datapath == '.pickle.gz':
			datapath = new_dirpath

	f = gzip.open(datapath, 'rb')




def readTIMIT(datapath='timit-mfcc-mono-tr.pfile.gz', format='pfile', shared=False, listify=False, mapping=48, percent_data=1., randomise=False):
	file_reader = PfileIO(datapath)
	file_reader.readpfileInfo()
	file_reader.readPfile(randomise=randomise)
	x, y = file_reader.generate_features(listify)

	if percent_data < 1. :
		x, y = partition_data(x, y, percent_data)

	# stats = Counter(y)
	if mapping == 48:
	# if y is a list, then iterate otherwise apply function once.	
		if isinstance(y, (list, tuple)) :
			y = map(lambda x: map_y_48(x), y)
		else:	
			y = map_y_48(y)
	elif mapping == 39:
		if isinstance(y, (list, tuple)):
			y = map(lambda x: map_y_39(x), y)
		else:	
			y = map_y_39(y)

	# print stats.most_common(100)
	if shared == True:
		x, y = shared_dataset((x, y))
	
	return x, y




def readTIMITSSL(datapath='timit-mfcc-mono-tr.pfile.gz', format='pfile', shared=False, listify=False, mapping=48, percent_data=0.99, randomise=True):
	file_reader = PfileIO(datapath)
	file_reader.readpfileInfo()
	file_reader.readPfile(randomise=randomise)
	x, y = file_reader.generate_features(listify)
	if isinstance(x, (list, tuple)):
		xmat = np.vstack(x)
		ymat = np.concatenate(y)
		x = xmat
		y = ymat

	total_samples = xmat.shape[0]
	total_labels = int(percent_data*total_samples)
	x_lab = x[:total_labels]
	if mapping == 48:
		if isinstance(y, (list, tuple)):
			y = map(lambda x:map_y_48(x), y)
		else:
			y = map_y_48(y)
	elif mapping == 39:
		if isinstance(y, (list, tuple)):
			y = map(lambda x:map_y_39(x), y)
		else:
			y = map_y_39(y)

	y_lab = y[:total_labels]
	x_unlab = x[total_labels:]
	return x_lab, y_lab, x_unlab  


# to seperate the data into training, test and validation sets.
# divide into a ratio of 70,15,15
# use it only for numpy
def make_sets(x, y):
	indices = x.shape[0]
	seed = 1111
	np.random.seed(seed)
	np.random.shuffle(x)
	np.random.seed(seed)
	np.random.shuffle(y)

	n_train_idx = abs(0.85 * indices)
	n_valid_idx = abs(0.11 * indices)
	n_test_idx = abs(0.04 * indices)

	train_set_x, train_set_y = x[:n_train_idx, :], y[:n_train_idx]
	valid_set_x, valid_set_y = x[n_train_idx:n_train_idx + n_valid_idx, :], y[n_train_idx:n_train_idx + n_valid_idx]
	test_set_x, test_set_y = x[n_train_idx + n_valid_idx:, :], y[n_train_idx + n_valid_idx:]
	return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]




def make_shared_sets(x, y):
	[train_set, valid_set, test_set] = make_sets(x, y)
	train_set_x, train_set_y = make_shared_partitions(train_set)
	valid_set_x, valid_set_y = shared_dataset(valid_set)
	test_set_x, test_set_y = shared_dataset(test_set)
	# print train_set_x.get_value().shape[0]
	return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]



def shared_dataset(data_xy, borrow=True):
	data_x, data_y = data_xy
	shared_x = theano.shared(np.asarray(data_x, dtype=np.float32), borrow=borrow)
	shared_y = theano.shared(np.asarray(data_y, dtype=np.float32), borrow=borrow)
	return shared_x, T.cast(shared_y, 'int32')



def make_shared_partitions(x, y):
	assert len(x) == len(y)

	shared_x = map(lambda x: theano.shared(np.asarray(x, dtype=np.float32), borrow=True), x)
	shared_y = map(lambda x: theano.shared(np.asarray(x, dtype=np.float32), borrow=True), y)
	shared_y = map(lambda x: T.cast(x, 'int32'), shared_y)

	return shared_x, shared_y


def map_y_48(y_inp):
        y_out = map(lambda x: int(x/3.), y_inp)
        return y_out


def map_y_39(y):
        # convert list of values to quotients of 3
        y_48 = map(lambda x: int(x/3.), y)
        y_39 = np.asarray(y_48)
        y_39[y_39 == 14] = 27
        y_39[y_39 == 5] = 2
        y_39[y_39 == 3] = 0
        y_39[y_39 == 15] = 29
        y_39[y_39 == 47] = 36
        y_39[y_39 == 23] = 22
        y_39[y_39 == 43] = 37
        y_39[y_39 == 16] = 37
        y_39[y_39 == 9] = 37

        print min(y_39), max(y_39)
        print min(y_48), max(y_48)
        phoneme_count = Counter(y_39)
        ph = phoneme_count.keys()
        ph.sort()
        p_keys = range(39)
        p_map = zip(ph, p_keys)
        # print p_map
        for x,y in p_map:
                y_39[y_39 == x] = y

        print min(y_39), max(y_39)
        y_39 = y_39.tolist()
        return y_39


def partition_data(x, y, percent_data, max_partitions=1):
	if isinstance(x, (list, tuple)):
		xmat = np.vstack(x)
		ymat = np.concatenate(y)
		x = xmat
		y = ymat

	num_samples = xmat.shape[0]
	num_labels = int(percent_data * num_samples)
	print "number of frames used in training:", num_labels
	# x_lab	
	x_lab = x[:num_labels]
	y_lab = y[:num_labels]
	x_unlab = x[num_labels:]
	x_lab = [x_lab]
	y_lab = [y_lab]
	return [x_lab, y_lab]




def partition_data_ssl(x, y, percent_data, max_partitions=1):
	if isinstance(x, (list, tuple)):
		xmat = np.vstack(x)
		ymat = np.concatenate(y)
		x = xmat
		y = ymat

	num_samples = xmat.shape[0]
	num_labels = int(percent_data * num_samples)
	# x_lab	
	x_lab = x[:num_labels]
	y_lab = y[:num_labels]
	x_unlab = x[num_labels:]

	
	return [x_lab, y_lab, x_unlab]	

	# x_f = x[:]




