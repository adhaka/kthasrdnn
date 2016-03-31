import numpy as np
import cPickle, gzip
import theano
import theano.tensor as T 
import math
from layers.HiddenLayer import HiddenLayer, LogisticRegression
from layers.DAE import DAE 
from theano.tensor.shared_randomstreams import RandomStreams


#  if there is no neural network, then sda should have the configuration for it by itself.
#  sda model can be used as a pretraining for a MLP or the configuration can also be defined seperatly, 
#  but if we have both of them, then the dnn configuration will take precedence.

class SdA(object):
	def __init__(self, inp, numpy_rng, theano_rng=None, hidden_layer=None, dnn=None, activations_layers=None):
		self.numpy_rng = numpy_rng
		self.hidden_layer = hidden_layer
		self.theano_rng = theano_rng
		self.hidden_layer_size = len(self.hidden_layer)

		if not theano_rng:
			theano_rng = RandomStreams(numpy_rng.randint( 2 ** 30))
		self.da_layers = []
		self.params = []

		self.x = T.matrix('x')
		self.x = inp

		if dnn:
			if dnn.x:
				self.x = dnn.x
			
			self.n_in = dnn.n_in
			self.n_out = dnn.n_out
			self.hidden_layer = dnn.hidden_layer

		self.hidden_layer_size = len(self.hidden_layer)
		layer_x = self.x
		n_input = layer_x.get_value().shape[1]

		for i, l in enumerate(hidden_layer):
			if i > 0:
				layer_x = dnn.layers[i-1].output(layer_x)
				n_input = self.hidden_layer[i-1]
			
			activation_fn = activations_layers[i]

			w = dnn.layers[i].w
			bhid = dnn.layers[i].b
			da = DAE(numpy_rng=numpy_rng, theano_rng=theano_rng, inp=layer_x, n_inputs=n_input, n_hiddens=l, w=w, bhid=bhid, activation=activation_fn)
			self.da_layers.append(da)
			self.params = self.params + da.params



	def pretraining_functions(self, train_set_x, batch_size):
		index = T.ivector('index')
		learning_rate = T.scalar('learning_rate')
		# corruption_level = T.scalar('corruption_level')
		momentum = T.scalar('momentum')	
		

		num_samples = train_set_x.get_value().shape[1]
		num_batches = num_samples / batch_size
		pretrain_fns = []

		for i in xrange(self.hidden_layer_size):
			da = self.da_layers[i]
			# cost, updates = da.get_cost_updates(corruption_level, learning_rate, momentum)
			# pt_fn = theano.function(inputs=[index, theano.Param(learning_rate, default=0.02), theano.Param(momentum, default=0.3), theano.Param(corruption_level, default=0.35)], updates=updates, outputs=[cost], givens={self.x:train_set_x[index]})
			cost, updates = da.get_cost_updates()
			pt_fn = theano.function(inputs=[index], updates=updates, outputs=[cost], givens={self.x:train_set_x[index]})
			pretrain_fns.append(pt_fn)


		return pretrain_fns