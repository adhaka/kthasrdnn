
import cPickle, gzip
import numpy as np
import theano
from theano import tensor as T

BATCH_SIZE = 300

# load dataset mnist

def load_datasets():
	f = gzip.open('mnist.pkl.gz', 'rb')
	train_set, valid_set, test_set = cPickle.load(f)


	def shared_dataset(data_xy):
		data_x, data_y = data_xy
		shared_x = theano.shared(np.asarray(data_x, dtype= theano.config.floatX))
		shared_y = theano.shared(np.asarray(data_y, dtype= theano.config.floatX))

		return shared_x, T.cast(shared_y, 'int32')


	test_set_x, test_set_y = shared_dataset(train_set)
	valid_set_x, valid_set_y = shared_dataset(valid_set)
	test_set_x, test_set_y = shared_dataset(test_set)


	batch_size = BATCH_SIZE
	


