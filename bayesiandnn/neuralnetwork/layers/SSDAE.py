import numpy as np
import theano
import theano.tensor as T 
import math
from layers.HiddenLayer import HiddenLayer, LogisticRegression
from theano.tensor.shared_randomstreams import RandomStreams
from collections import OrderedDict


# single point class for implementing semi-supervised training of DNN with autoencoder penalty.

class SSDAE(object):
	def __init__(self, numpy_rng, config, batch_size=400, theano_rng=None, inp=None):
		self.numpy_rng = numpy_rng
		self.theano_rng = theano_rng
		self.batch_size = batch_size

		if not theano_rng:
			self.theano_rng = RandomStreams(numpy_rng.randint( 2 ** 30))

		self.num_layers = len(config)
		self.params_layers = []


	def train():
		pass





# move it to layers folder later .. 
class SSLayer(object):
	def __init__(self, rng, n_inputs, n_outputs, n_targets, corruption=0.40, batch_size=400, activation='sigmoid'):
		self.rng = rng
		self.n_inputs = n_inputs
		self.n_outputs = n_outputs 
		self.encoder = HiddenLayer(self.rng, self.n_inputs, self.n_outputs, activation=activation)
		self.decoder = HiddenLayer(self.rng, self.n_outputs, self.n_inputs, activation=activation)

		self.softmaxLayer = LogisticRegression(self.rng, n_outputs, n_targets)
		self.params = self.encoder.params + self.decoder.params + self.softmaxLayer.params 
		self.delta_params = self.encoder.delta_params + self.decoder.delta_params + self.softmaxLayer		


	@staticmethod
	def _shared_dataset(x, borrow=True):
		return theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=borrow)



	def get_cost_updates(self, x_lab, x_unlab, y_lab):
		self.x_lab = x_lab 
		self.x_unlab = x_unlab 
		self.y_lab = y_lab
		out_unlab = self.encoder.output(self.x_unlab)
		z_unlab = self.decoder.output(out_unlab)
		preds_lab = self.softmaxLayer.predict(x_lab)


		accuracy = self.softmaxLayer.calcAccuracy(x_lab y_lab)
		cost_reconstruction_unlab = T.mean((z_unlab-x_unlab)*(z_unlab-x_unlab))
		cost_reconstruction_lab = T.mean()  
		cost_classification = self.softmaxLayer.cost(x_lab, y_lab)
		cost = cost_reconstruction + cost_classification

		updates = OrderedDict()
		gparams = T.grad(cots, wrt=self.params)
		for p, gp in zip(params, gparams):
			updates[p] = p - gp*learning_rate

		return (cost, accuracy)

	# for a better control, this fn will take numpy arrays. 
	# make batches such that they have some respresentation from labelled data as well and if possible with the same amount of points per class.
	def train(self, xlab_numpy, ylab_numpy, xunlab_numpy):

		assert xlab.shape[0] == len(ylab)
		xlab = self._shared_dataset(xlab_numpy)
		ylab = self._shared_dataset(ylab_numpy)
		xunlab = self._shared_dataset(xunlab_numpy)

		self.alpha = float(xlab.shape[0] / xunlab.shape[0])
		index_unlab = T.ivector('index_unlab')
		index_lab = T.ivector('index_lab')
		momentum = T.scalar('momentum')
		learning_rate = T.scalar('learning_rate')
		cost, updates = self.get_cost_updates(xlab, xunlab, ylab)

		batch_size_lab = self.batch_size * alpha
		batch_size_unlab = self.batch_size * (1-alpha)
		xlab = T.matrix('xlab')
		xunlab = T.matrix('xunlab')
		ylab = T.ivector('ylab')


		
		batch_sgd_train = theano.function(inputs=[index_unlab, index_lab], outputs=[cost, accuracy], givens={xlab:xlab[]})

















		




