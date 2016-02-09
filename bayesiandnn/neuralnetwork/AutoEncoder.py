import numpy as np
import cPickle, gzip
import theano
import theano.tensor as T 
import math
from layers.HiddenLayer import HiddenLayer, LogisticRegression
from layers.DAe import DAE 
from theano.tensor.shared_randomstreams import RandomStreams




class SdA(object):
	def __init__(self, inp, numpy_rng, theano_rng=None, hidden_layer=None, dnn=None):
		self.numpy_rng = numpy_rng
		self.hidden_layer = hidden_layer
		self.theano_rng = theano_rng
		self.hidden_layer_size = len(self.hidden_layer)


		if not theano_rng:
			theano_rng = RandomStreams(numpy_rng.randint( 2 ** 30))
		self.da_layers = []

		self.x = inp		
		if dnn:
			self.x = dnn.x

		layer_x = self.x
		n_input = layer_x.shape[1]

		for i, l in enumerate(hidden_layer):
			if i > 0:
				layer_x = dnn.layers[i-1].output
				n_input = self.hidden_layer[i-1]

			w = dnn.layers[i].w
			bhid = dnn.layers[i].b
			da = DAE(numpy_rng, theano_rng, layer_x, n_input, l, w, bhid)
			self.da_layers.append(da)





	def pretraining_functions(self, train_set_x, batch_size):

		index = T.ivector('index')
		learning_rate = T.dscalar('learning_rate')
		corruption_level = T.dscalar('corruption_level')
		momentum = T.dscalar('momentum')
		num_samples = train_set_x.get_value().shape[1]
		num_batches = num_samples / batch_size

		pretrain_fns = []

		for i in xrange(self.hidden_layer_size):
			da = self.da_layers[i]
			cost, updates = da.get_cost_updates(corruption_level, learning_rate, momentum)
			pt_fn = theano.function(inputs=[index, theano.Param(learning_rate, default=0.02), theano.Param(momentum, default=0.4), theano.Param(corruption_level, default=0.35)], updates=updates, outputs=[cost], givens={ x:train_set_x[index] })
			pretrain_fns.append(pt_fn)

		return pretrain_fns