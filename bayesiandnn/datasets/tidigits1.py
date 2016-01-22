
import numpy as np 
import cPickle, gzip
import os
import theano
import math
from collections import Counter
from theano import tensor as T 



def _load_raw_data(datapath):
	dirpath, filename = os.path.split(datapath)

	if dirpath == '' and not os.path.isfile(datapath):
		new_dirpath = os.path.join(
			os.path.split(__file__)[0],
			'rawdata/',
			filename
			)

		if os.path.isfile(new_dirpath) or datapath == 'train_not_isolated_mfcc.pickle.gz':
			datapath = new_dirpath

	f = gzip.open(datapath, 'rb')
	feats, labels = cPickle.load(f)

	return feats, labels



def load_data_ssl(datapath, percent = 0.70):
	feats, labels = _load_raw_data(datapath)
	print feats.shape, labels.shape

	print Counter(labels).most_common()
	# Counter(labels).most_common()
	num_cls = np.max(labels, axis=0)
	num_points = feats.shape[0]

	num_labelled = int(math.ceil(percent * num_points))
	num_unlabelled = num_points - num_labelled
	labels = labels[:, np.newaxis]

	xy_comb = np.hstack((feats, labels))
	np.random.shuffle(xy_comb)
	feats, labels = xy_comb[:,:xy_comb.shape[1] -1], xy_comb[:, xy_comb.shape[1]-1:]

	x_lab, y_lab = feats[:num_labelled, :], labels[:num_labelled, :]
	x_unlab, y_unlab = feats[:num_labelled, :], labels[:num_labelled,:]

	return [x_lab, y_lab, x_unlab, y_unlab]

