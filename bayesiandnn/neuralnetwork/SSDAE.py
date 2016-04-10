import numpy as np
import theano
import theano.tensor as T
from theano.printing import debugprint, Print
from theano import pp
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

		self.x_lab = T.matrix('x_lab')
		self.x_unlab = T.matrix('x_unlab')
		self.y_lab = T.ivector('y_lab')
		# self.x_total = 
		self.num_samples = self.x_lab_np.shape[0] + self.x_unlab_np.shape[0]
		input_size = self.x_lab_np.shape[1]
		target_size = len(list(set(y_lab_np)))
		output_size = hidden_layers[0]

		self.layers = []
		self.params = []
		for i, hl in enumerate(hidden_layers):
			if i ==0:
				input_lab = self.x_lab
				input_unlab = self.x_unlab
				out_lab = self.y_lab
			if i > 0:
				out_lab = self.y_lab
				input_lab = self.layers[-1].output(input_lab)
				input_unlab = self.layers[-1].output(input_unlab)
				input_size = hidden_layers[i-1]
				output_size = hl
			ssda = SSLayer(numpy_rng, self.theano_rng, input_size, output_size, target_size, x_lab=input_lab, y_lab=out_lab, x_unlab=input_unlab, activation=activation)
			self.layers.append(ssda)
			self.params = self.params + ssda.params 



	@staticmethod
	def _shared_dataset(x, borrow=True):
		return theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=borrow)


	@staticmethod
	def _shared_dataset_y(y, borrow=True):
		rval = theano.shared(np.asarray(y, dtype=theano.config.floatX), borrow=borrow)
		return T.cast(rval, 'int32')


	def get_training_functions(self, x_lab=None, y_lab=None, x_unlab=None):
		# assert xlab.shape[0] == len(y_lab) 
		assert self.x_lab_np.shape[0] == len(self.y_lab_np)
		# self.x_lab = self._shared_dataset(self.x_lab_np)
		# self.y_lab = self._shared_dataset(self.y_lab_np)
		# self.x_unlab = self._shared_dataset(self.x_unlab_np)

		self.alpha = self.x_lab_np.shape[0] / float(self.x_lab_np.shape[0] + self.x_unlab_np.shape[0])
		index_unlab = T.ivector('index_unlab')
		index_lab = T.ivector('index_lab')
		momentum = T.scalar('momentum')
		learning_rate = T.scalar('learning_rate')
		# cost, updates = self.get_cost_updates(self.x_lab, self.x_unlab, self.y_lab)

		self.batch_size_lab = int(self.batch_size * self.alpha)
		self.batch_size_unlab = int(self.batch_size * (1-self.alpha))

		self.num_labels = self.x_lab_np.shape[0]
		self.num_unlabels = self.x_unlab_np.shape[0]
		self.num_samples = self.num_labels + self.num_unlabels

		num_batches = self.num_samples / float(self.batch_size)
		pretraining_fns = []
		for i in xrange(len(self.hidden_layers)):
			l = self.layers[i]
			# if i ==0:
			# 	x_l = self.x_lab
			# 	x_ul = self.x_unlab
			# 	y_l = self.y_lab

			# if i > 0:
			# 	x_l = self.layers[i-1].out_lab
			# 	x_ul = self.layers[i-1].out_unlab
			# 	y_l = self.y_lab
			
			# result = l.get_cost_updates(x_l, x_ul, y_l)
			result = l.get_cost_updates()
			cost, updates = result[0], result[1]

			train_fn = theano.function(inputs=[index_lab, index_unlab], updates=updates, outputs=[cost], givens={self.x_lab:x_lab[index_lab], self.x_unlab:x_unlab[index_unlab], self.y_lab:y_lab[index_lab]}, on_unused_input='warn')
			pretraining_fns.append(train_fn)

		return  pretraining_fns




# this function does the complete training for the network. single point function.
	def trainSGD(self):
		self.num_batches = int(self.num_samples / self.batch_size)
		NUM_EPOCHS = 15
		x_lab_shared = self._shared_dataset(self.x_lab_np)
		x_unlab_shared = self._shared_dataset(self.x_unlab_np)
		y_lab_shared = self._shared_dataset_y(self.y_lab_np)
		pretrain_fns = self.get_training_functions(x_lab=x_lab_shared, x_unlab=x_unlab_shared, y_lab=y_lab_shared)
		print self.num_labels, self.num_unlabels
		indices_lab = np.arange(self.num_labels, dtype=np.dtype('int32'))
		indices_unlab = np.arange(self.num_unlabels, dtype=np.dtype('int32'))
		c = []

		print "............ Pretrainining ..............."
		# pretraining loop for all the hidden layers .....
		for i in xrange(len(self.hidden_layers)):
			print "yay"
			for epoch in xrange(NUM_EPOCHS):
				for j in xrange(self.num_batches - 1):
					index_lab = indices_lab[j*self.batch_size_lab:(j+1)*self.batch_size_lab]
					index_unlab = indices_unlab[j*self.batch_size_unlab:(j+1)*self.batch_size_unlab]
					res = pretrain_fns[i](index_lab=index_lab, index_unlab=index_unlab)
					# cost = 
					c.append(res[0])

				print "cost is:", np.nanmean(c) 


		# pass



# move it to layers folder later .. 
class SSLayer(object):
	def __init__(self, numpy_rng, theano_rng, n_inputs, n_outputs, n_targets, x_lab=None, x_unlab=None, y_lab=None, learning_rate = 0.04, corruption=0.20, batch_size=400, activation='tanh'):
		self.numpy_rng = numpy_rng
		self.theano_rng = theano_rng
		self.n_inputs = n_inputs
		self.n_outputs = n_outputs 
		self.encoder = HiddenLayer(self.numpy_rng, self.n_inputs, self.n_outputs, activation=activation)
		self.decoder = HiddenLayer(self.numpy_rng, self.n_outputs, self.n_inputs, activation=activation)
		self.learning_rate = learning_rate
		self.activation = activation
		self.out_lab = self.encoder.output(x_lab)
		self.out_unlab = self.encoder.output(x_unlab)
		# self.inp_lab 

		if x_lab == None:
			self.x_lab = T.matrix('inp_lab')
		else:
			self.x_lab = x_lab

		if x_unlab == None:
			self.x_unlab = T.matrix('inp_unlab')
		else:
			self.x_unlab = x_unlab

		if y_lab == None:
			self.y_lab = T.matrix('y_lab')
		else:
			self.y_lab = y_lab

		self.softmaxLayer = LogisticRegression(self.numpy_rng, n_outputs, n_targets)
		self.params = self.encoder.params + self.decoder.params
		self.paramsAll = self.encoder.params + self.decoder.params + self.softmaxLayer.params
		# self.params = self.encoder.params + self.decoder.params
		# self.delta_params = self.encoder.delta_params + self.decoder.delta_params + self.softmaxLayer.delta_params		


	@staticmethod
	def _shared_dataset(x, borrow=True):
		return theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=borrow)


	def output(self, x):
		out = self.encoder.output(x)
		# out_unlab = self.encoder.output(x_unlab)
		return out

	def get_cost_updates(self):
		# self.x_lab_layer = x_lab 
		# self.x_unlab_layer = x_unlab 
		# self.y_lab_layer = y_lab
		out_unlab = self.encoder.output(self.x_unlab)
		out_lab = self.encoder.output(self.x_lab)
		z_unlab = self.decoder.output(out_unlab)
		z_lab = self.decoder.output(out_lab)
		preds_lab = self.softmaxLayer.predict(out_lab)

		# accuracy = self.softmaxLayer.calcAccuracy(out_lab, y_lab)
		# cost_reconstruction_unlab = T.mean((z_unlab-x_unlab)*(z_unlab-x_unlab))
		# cost_reconstruction_lab = T.mean((z_lab - x_lab)*(z_lab - x_lab))  
		if self.activation == 'sigmoid':
			cost_reconstruction_lab = -T.sum(self.x_lab * T.log(z_lab) + (1 - self.x_lab) * T.log(1-z_lab), axis=1)
			cost_reconstruction_lab = T.mean(cost_reconstruction_lab)
			cost_reconstruction_unlab = T.mean(-T.sum(self.x_unlab * T.log(z_unlab) + (1 - self.x_unlab) * T.log(1-z_unlab), axis=1))
		elif self.activation == 'tanh':
			cost_reconstruction_lab = T.mean(T.sum((self.x_lab - z_lab)*(self.x_lab - z_lab), axis=1))
			cost_reconstruction_unlab = T.mean(T.sum(self.x_unlab - z_unlab)*(self.x_unlab - z_unlab), axis=1)
		cost_classification = self.softmaxLayer.calcAccuracy(out_lab, self.y_lab) * 10
		cost = cost_reconstruction_lab + cost_reconstruction_unlab + cost_classification  
		# pp(cost)
		updates = OrderedDict()
		gparams = T.grad(cost, wrt=self.params)
		for p, gp in zip(self.params, gparams):
			updates[p] = p - gp*self.learning_rate

		# debugprint(cost)
		# exit()
		return [cost, updates]


	# for a better control, this fn will take numpy arrays. 
	# make batches such that they have some respresentation from labelled data as well and if possible with the same amount of points per class.
	def train(self, x_lab_numpy, y_lab_numpy, xunlab_numpy):
		pass
		# batch_sgd_train = theano.function(inputs=[index_unlab, index_lab], outputs=[cost, accuracy], givens={xlab:xlab[]})

















		




