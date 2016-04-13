
# this file implements a ladder network ..... 
import logging
import numpy as np
from collections import OrderedDict 
from os import sys, path
import os, cPickle, gzip 
import theano
import theano.tensor as T 
from theano.tensor.shared_randomstreams import RandomStreams 

sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from utils import ConfusionMatrix, LearningRate
from utils.LearningRate import *
from utils import *


class LadderAE(object):
	def __init__(self, x_labelled, x_unlabelled, y_labelled, num_output, hidden_layer_config, noise_layers_std=[0.2], gamma_layers=[1.0], beta_layers=[1.0], activation_layer_config=['tanh'], learning_rate_config=[0.02], BATCH_SIZE=400, random_seed=1111):
		self.hidden_layer_config = hidden_layer_config
		self.noise_layers_std = noise_layers_std
		self.activation_layer_config = activation_layer_config
		self.learning_rate_config = learning_rate_config
		self.num_layers = len(hidden_layer_config)
		self.x_labelled = x_labelled
		self.y_labelled = y_labelled
		self.x_unlabelled = x_unlabelled
		self.noise_layers_std = noise_layers_std
		self.gamma_layers = gamma_layers
		self.beta_layers = beta_layers
		# dimension of the last softmax output layer ..

		self.num_output = num_output
		self.numpy_rng = np.random.RandomState(random_seed)
		self.theano_rng = RandomStreams(random_seed)
		self.denoising_cost = []
		self.encoders = []
		# self.rstream = 


		if len(self.activation_layer_config) == 1:
			self.activation_layer_config = self.activation_layer_config * self.num_layers 

		if not isinstance(corruption_layer_config, ['list', 'tuple']):
			raise TypeError()

		if len(self.corruption_layer_config) == 1:
			self.corruption_layer_config = self.corruption_layer_config * self.num_layers

		# last layer will always have a sigmoid non-linearity 
		self.activation_layer_config += ['sigmoid']
		self.acts = [None] * self.num_layers
		self.num_labelled = self.x_labelled.shape[0]
		self.num_unlabelled = self.x_unlabelled.shape[0]
		self.num_samples = self.num_labelled + self.num_unlabelled
		self.num_dims = x_labelled.shape[1]
		self.x_joined = T.concatenate([x_unlabelled, x_unlabelled], axis=0)
		self.join = lambda x,y: T.concatenate([x,y], axis=0)
		self.labelled = lambda x: x[:self.num_labelled,:]
		self.unlabelled = lambda x: x[self.num_labelled:,:]
		self.seperate = lambda x: [self.labelled(x), self.unlabelled(x)]


	#  apply enocder fist and then decoder functionality here ...
	def apply(self):
		self.encoder_pass()
		self.decoder_pass()



	def encoder_pass(self, inputs, weight):
		# for i in xrange(self.num_layers):
			# l = HiddenLayer(hidden_layer_config[i], corruption_layer_config[i], activation=activation_layer_config[i])
		# h = self.add_gaussian_noise(inputs)
		# inp_dim = self.num_dims
		# layer_values = {}
		# layer_values[1] = {}
		# layer_values[1]['inp'] = h
		# # calculate preactivation here ....
		# for i in 1:self.num_layers:
		# 	layer_values[i]['W'] = init_shared_weight() 
		encoders = []
		corrupt_encoders = []
		input_labelled = self.x_labelled
		input_unlabelled = self.x_unlabelled
		h_dict = {}


		noise_std = 1.0
		# clean encoder pass ...
		for i in xrange(self.num_layers):
			le = EncoderLadder(input_labelled, input_unlabelled, self.hidden_layer_config[i], gamma=gamma_layers[i], beta=beta_layers[i])
			le_corrupt = EncoderLadder(input_labelled_corrupt, input_unlabelled_corrupt, self.hidden_layer_config[i], noise_std=self.noise_layers_std[i], gamma =gamma_layers[i], beta=beta_layers[i])
			encoders.append(le)
			corrupt_encoders.append(le_corrupt)
			input_labelled = self.labelled(le.h)
			input_unlabelled = self.unlabelled(le.h)
			input_labelled_corrupt = self.labelled(le_corrupt.h)
			input_unlabelled_corrupt = self.unlabelled(le_corrupt.h)



	def decoder_pass()


	def generate_noise(self, x):
		noise_np = np.random.randn(x.shape)
		noise = T.cast(noise_np, theano.config.floatX)
		return noise

	def add_gaussian_noise(self, x, sigma=1.0):
		noise = np.random.normal(size=x.shape, loc=0.0, scale=1.0)
		noise = T.cast(noise, dtype=x.dtype)
		return x + sigma*noise


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


