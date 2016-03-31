import numpy as np
import theano
import theano.tensor as T 
import math
from layers.HiddenLayer import HiddenLayer, LogisticRegression
from theano.tensor.shared_randomstreams import RandomStreams
from collections import OrderedDict


# single point class for implementing semi-supervised training of DNN with autoencoder penalty.
class SSDAE(object):
	def __init__(self, numpy_rng, hidden_layers, x_lab_np, y_lab_np, x_unlab_np, batch_size=400, theano_rng=None, activation='sigmoid'):
		self.numpy_rng = numpy_rng
		self.theano_rng = theano_rng
		self.batch_size = batch_size
		self.hidden_layers = hidden_layers

		self.theano_rng = RandomStreams(numpy_rng.randint( 2 ** 30))
		self.num_layers = len(hidden_layers)
		self.params_layers = []
		self.x_lab_np = x_lab_np
		self.x_unlab_np = x_unlab_np
		self.y_lab_np = y_lab_np
		self.num_samples = self.x_lab_np.shape[0] + self.x_unlab_np.shape[0]
		input_size = self.x_lab_np.shape[1]
		target_size = len(list(set(y_lab_np)))
		output_size = hidden_layers[0]

		self.layers = []
		self.params = []
		for i, hl in enumerate(hidden_layers):
			if i > 0:
				input_size = hidden_layers[i-1]
				output_size = l
			ssda = SSLayer(numpy_rng, self.theano_rng, input_size, output_size, target_size, activation=activation)
			self.layers.append(ssda)
			self.params = self.params + ssda.params 



	@staticmethod
	def _shared_dataset(x, borrow=True):
		return theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=borrow)


	def get_training_functions(self, x_lab_np=None, y_lab_np=None, x_unlab_np=None):
		# assert xlab.shape[0] == len(y_lab) 
		assert self.x_lab_np.shape[0] == len(y_lab)
		self.x_lab = self._shared_dataset(self.x_lab_np)
		self.y_lab = self._shared_dataset(self.y_lab_np)
		self.x_unlab = self._shared_dataset(self.x_unlab_np)
		self.alpha = float(xlab.shape[0] / xunlab.shape[0])
		index_unlab = T.ivector('index_unlab')
		index_lab = T.ivector('index_lab')
		momentum = T.scalar('momentum')
		learning_rate = T.scalar('learning_rate')
		cost, updates = self.get_cost_updates(self.x_lab, self.x_unlab, self.y_lab)

		self.batch_size_lab = self.batch_size * self.alpha
		self.batch_size_unlab = self.batch_size * (1-self.alpha)
		x_lab = T.matrix('x_lab')
		x_unlab = T.matrix('x_unlab')
		y_lab = T.ivector('y_lab')

		self.num_labels = self.x_lab_np.shape[0]
		self.num_unlabels = self.x_unlab_np[0]
		self.num_samples = num_labels + num_unlabels

		num_batches = num_samples / float(self.batch_size)
		pretraining_fns = []
		for i in xrange(len(hidden_layers)):
			ssda = self.layers[i]
			cost, updates = ssda.get_cost_updates(self.x_lab, self.x_unlab, self.y_lab)
			train_fn = theano.function(inputs=[index_lab, index_unlab], updates=updates, outputs=[cost], givens={self.x_lab:self.x_lab[index_lab], self.x_unlab:self.x_unlab[index_unlab], self.y_lab:self.y_lab[index_lab]})
			pretraining_fns.append(train_fn)

		return  pretraining_fns



# this function does the complete training for the network. single point function.
	def trainSGD(self):
		self.num_batches = self.num_samples / float(self.batch_size)
		NUM_EPOCHS = 25
		indices_lab = np.arange(self.num_labels, dtype=np.dtype('int32'))
		indices_unlab = np.arange(self.num_unlabels, dtype=np.dtype('int32'))
		pretrain_fns = self.get_training_functions()


		for i in xrange(len(self.hidden_layers)):
			for epoch in xrange(NUM_EPOCHS):
				for j in xrange(self.num_batches):
					index_lab = indices_lab[j*self.batch_size_lab:(j+1)*self.batch_size_lab]
					index_unlab = indices_unlab[j*self.batch_size_unlab:(j+1)*self.batch_size_unlab]
					c,a = pretrain_fns[i](index_lab=index_lab, index_unlab, index_unlab=index_unlab)


		# pass





# move it to layers folder later .. 
class SSLayer(object):
	def __init__(self, numpy_rng, theano_rng, n_inputs, n_outputs, n_targets, corruption=0.40, batch_size=400, activation='sigmoid'):
		self.rng = rng
		self.n_inputs = n_inputs
		self.n_outputs = n_outputs 
		self.encoder = HiddenLayer(self.rng, self.n_inputs, self.n_outputs, activation=activation)
		self.decoder = HiddenLayer(self.rng, self.n_outputs, self.n_inputs, activation=activation)

		self.numpy_rng = numpy_rng
		self.theano_rng = theano_rng
		self.x_lab = None
		self.x_unlab = None
		self.y_lab = None
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

		accuracy = self.softmaxLayer.calcAccuracy(x_lab, y_lab)
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
	def train(self, x_lab_numpy, y_lab_numpy, xunlab_numpy):
		pass
		# batch_sgd_train = theano.function(inputs=[index_unlab, index_lab], outputs=[cost, accuracy], givens={xlab:xlab[]})

















		




