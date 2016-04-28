
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
	def __init__(self, x_labelled_np, x_unlabelled_np, y_labelled_np, hidden_layer_config, noise_layers_std=[0.2], gamma_layers=[1.0], beta_layers=[1.0], activation_layer_config=['tanh'], learning_rate_config=[0.02], BATCH_SIZE=300, random_seed=1111):
		self.hidden_layer_config = hidden_layer_config
		self.noise_layers_std = noise_layers_std
		self.activation_layer_config = activation_layer_config
		self.learning_rate_config = learning_rate_config
		self.num_layers = len(hidden_layer_config)
		self.x_labelled_np = x_labelled_np
		self.y_labelled_np = y_labelled_np
		self.x_unlabelled_np = x_unlabelled_np
		self.BATCH_SIZE = BATCH_SIZE

		self.noise_layers_std = noise_layers_std
		self.gamma_layers = gamma_layers
		self.beta_layers = beta_layers
		# dimension of the last softmax output layer ..

		self.x_labelled = self._shared_dataset(self.x_labelled_np)
		self.y_labelled = self._shared_dataset(self.y_labelled_np)
		self.x_unlabelled = self._shared_dataset(self.x_unlabelled_np)
		self.num_output = hidden_layer_config[-1]
		self.N = self.x_labelled_np.shape[0]
		self.num_samples = self.N + self.x_unlabelled_np.shape[0]
		self.numpy_rng = np.random.RandomState(random_seed)
		self.theano_rng = RandomStreams(random_seed)
		self.denoising_cost = []
		self.encoders = []

		if len(self.activation_layer_config) == 1:
			self.activation_layer_config = self.activation_layer_config * self.num_layers 

		if not isinstance(corruption_layer_config, ['list', 'tuple']):
			raise TypeError()

		if len(self.corruption_layer_config) == 1:
			self.corruption_layer_config = self.corruption_layer_config * self.num_layers

		# last layer will always have a sigmoid non-linearity 
		if self.activation_layer_config[-1] != 'sigmoid':
			self.activation_layer_config += ['sigmoid']


		self.acts = [None] * self.num_layers
		self.num_labelled = self.x_labelled_np.shape[0]
		self.num_unlabelled = self.x_unlabelled_np.shape[0]

		self.num_samples = self.num_labelled + self.num_unlabelled
		self.num_dims = x_labelled_np.shape[1]
		self.x_joined = T.concatenate([x_labelled, x_unlabelled], axis=0)
		self.join = lambda x,y: T.concatenate([x,y], axis=0)
		self.labelled = lambda x: x[:self.num_labelled,:]
		self.unlabelled = lambda x: x[self.num_labelled:,:]
		self.seperate = lambda x: [self.labelled(x), self.unlabelled(x)]




	#  apply enocder fist and then decoder functionality here ...
	def apply(self):
		clean_encoders, corrupt_encoders = self.encoder_pass()
		cost_reconstruction = self.decoder_pass(clean_encoders, corrupt_encoders)
		outLayer = corrupt_encoders[-1]
		outLayer_clean = encoders[-1]
		y_c_l = outLayer.d['labelled']['h']
		y, d = outLayer_clean.get_layer_params()

		# y_labelled showld be one of K encdoing for this to work properly ...
		supervised_cost = -T.mean(T.sum(y_labelled*y_c_l, axis=1), axis=0)
		cost = cost_reconstruction + supervised_cost

		pred_cost = -T.mean(T.sum(y_labelled*T.log(y), axis=1), axis=0)
		correct_predictions = T.equal(T.argmax(y_labelled, axis=1), T.argmax(y, axis=1))
		accuracy = T.mean(T.cast(correct_predictions, 'float32'), axis=0) * T.scalar(value=100.0)
		return cost, cost_reconstruction, pred_cost


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

		self.clean_encoders = encoders
		self.corrupt_encoders = corrupt_encoders



	def decoder_pass(self, clean_encoders, corrupt_encoders):
		decoders = []
		input_labelled = self.x_labelled
		input_unlabelled = self.x_unlabelled
		L = self.L

		#  decalring the list of hyper paramsters for each layer here .. to be optimised later ..
		l1_params = []
		u = clean_encoders[-1].h
		u_c = corrupt_encoders[-1].h
		u_c_ul, u_c_l = self.seperate(u_c)
		z_c_ul, z_c_l =u_c_ul, u_c_l
		hyper_params = [0, 1, 0, 0, 0, 0, 1, 0, 0, 0]
		r_cost =[]

		for l in L:1:-1:
			n_outputs = self.hidden_layer_config[l]
			d = DecoderLadder(z_denoised_top=z_c_ul, encoder=clean_encoders[l], encoder_corrupt=corrupt_encoders[l], n_outputs=n_outputs, hyper_params=hyper_params)
			z_c_ul = self.split(d.z_est)
			r_c = d.getCost()
			r_cost = r_c + r_cost
			decoders = d + decoders

		self.decoders = decoders
		total_reconstruct_cost = reduce(lambda x,y:x+y, r_cost)
		return total_reconstruct_cost


	@staticmethod
	def _shared_dataset(x, borrow=True):
		return theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=borrow)


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


# training by mini-batch stochastic gradient ....
	def get_train_fns(self):
		learning_rate = T.scalar('learning_rate')
		self.x_labelled = T.matrix('x_labelled')
		self.y_labelled = T.matrix('y_labelled')
		self.x_unlabelled = T.matrix('x_unlabelled')

		index1 = T.ivector('index1')
		index2 = T.ivector('index2')
		momentum = T.scalar('momentum')

		num_batches = num_samples / self.BATCH_SIZE
		NUM_EPOCHS = 25

		cost, reconstruction_cost, accuracy = self.apply()
		params = []
		for l in self.encoders:
			params += l.get_layer_params()

		for d in self.decoders:
			params += d.get_layer_params()


		p_grads = T.grad(cost=cost, wrt=params)
		updates = OrderedDict()

		for p, gp in zip(params, p_grads):
			updates[p] = p - lr*gp


		batch_sgd_train = theano.function(inputs=[index1, index2], outputs=[cost, accuracy], updates=updates, givens={self.x_labelled:self.x_labelled_np[index1], self.y_labelled:self.y_labelled_np[index1], self.x_unlabelled:self.x_unlabelled_np[index2])})
		
		batch_sgd_validate = theano.function(inputs=[index_valid], outputs=accuracy)

		batch_sgd_train = theano.function(inputs=[index_test], outputs=accuracy)

		return [batch_sgd_train, batch_sgd_validate, batch_sgd_train]


	def trainmbSGD(self):
		self.num_unlabelled_batches = (self.num_samples - self.N) / float(self.BATCH_SIZE)
		self.num_labelled_batches = self.N/self.BATCH_SIZE
		NUM_EPOCHS= 25
		indices_lab = np.arange(self.N, dtype=np.dtype('int32'))
		indices_unlab = np.arange(self.N, dtype=np.dtype('int32'))
		fns = self.get_train_fns() 
		train_fn, valid_fn, test_fn = fns[0], fns[1], fns[2]

		for epoch in xrange(NUM_EPOCHS):
			for i in xrange(self.num_batches):
				index_lab = indices_lab[i*self.BATCH_SIZE:(i+1)*self.BATCH_SIZE]
				index_unlab = indices_unlab[]
				c,a = train_fn(index1=index_lab, index2=index_unlab)

		# pass
