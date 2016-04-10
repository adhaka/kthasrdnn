
# this file implements a ladder network ..... 
import logging
import numpy as np
from collection import OrderedDict 
import os, cPickle, gzip 
import math
import theano
import theano.tensor as T 
from theano.tensor.shared_randomstreams import RandomStreams 


class LadderAE(object):
	def __init__(self, x_labelled=None, x_unlabelled=None, y_labelled=None, hidden_layer_config, corruption_layer_config, activation_layer_config, learning_rate_config=[0.02], random_seed=1111):
		self.hidden_layer_config = hidden_layer_config
		self.corruption_layer_config = corruption_layer_config
		self.activation_layer_config = activation_layer_config
		self.learning_rate_config = learning_rate_config
		self.num_layers = len(hidden_layer_config)
		self.x_labelled = x_labelled
		self.y_labelled = y_labelled
		self.x_unlabelled = x_unlabelled
		self.numpy_rng = np.random.RandomState(random_seed)
		self.theano_rng = RandomStreams(random_seed)


		if len(self.activation_layer_config) == 1:
			self.activation_layer_config = self.activation_layer_config * self.num_layers 

		if len(self.corruption_layer_config) == 1:
			self.corruption_layer_config = self.corruption_layer_config * self.num_layers


		self.activation_layer_config += ['sigmoid']
		self.acts = [None] * self.num_layers


	def encoder_pass(self):
		for i in xrange(self.num_layers):
			l = HiddenLayer(hidden_layer_config[i], corruption_layer_config[i], activation=activation_layer_config[i])
			
		# pass


	def generate_noise(self, x):
		noise_np = np.random.randn(x.shape)
		noise = T.cast(noise_np, theano.config.floatX)
		return noise

	def add_gaussian_noise(self, x):
		noise = np.random.normal(size=x.shape, loc=0.0, scale=1.0)
		noise = T.cast(noise, dtype=theano.config.floatX)
		return x + noise


	def normalise_numpy(self, x):
		return  (x - np.mean(x, axis=0)) / np.std(x, axis=0)


	def normalise_theano(self, x):
		x_normalised = (x - T.mean(x, axis=0)) / np.std(x, axis=0)
		return x_normalised

#  initlaise W matrix in numpy format
	def initialise_weights(self, n_input, n_output):
		W =  np.random.randn(size=(n_input, n_output), dtype=theano.config.floatX) / np.sqrt(n_input)
		return W 



	def decoder_pass(self):
		pass




		# if len()



	# pass


