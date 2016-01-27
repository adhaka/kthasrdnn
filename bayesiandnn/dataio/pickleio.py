import numpy as np 
import os
import cPickle, gzip
import theano 
from theano import tensor as T 
from theano import sharedstreams


# pickle io - only for theano and for using it in parallel programming
class PickleIO(object):
	def __init__(self):
		self.feats = []
		self.labels = []
		self.partCounter = 0


	def loadData(self, datapath):
		dirpath, filename = os.path.split(datapath)

		if dirpath == '' and not os.path.isfile(datapath):
			new_dirpath = os.path.join(
			os.path.split(__file__)[0],
			'rawdata/',
			filename
			)

		if os.path.isfile(new_dirpath) or datapath == 'speechtrain1.pickle.gz':
			datapath = new_dirpath


		f = gzip.open(datapath, 'rb')
		self.feats, self.labels = cPickle.load(f)


	def loadDataTheano(self, datapath):
		x, y = self.loadData(datapath)
		x, y = self._shared_dataset([x, y])
		return x, y


	def _shared_dataset(self, data_xy, borrow=True):
		data_x, data_y = data_xy
		shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
		shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
		return shared_x, shared_y


