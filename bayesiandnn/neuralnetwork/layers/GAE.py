import numpy as np
import os
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from sklearn import NearestNeighbors
from collections import OrderedDict 


# in autoencoders, each layer is greedily trained with respect to the final output.

class GAE(object):
	'''
	This class is implemented as a basic underlying layer to make a stacked denoising auto-encoder...
	This is a denoising auto-encoder single layer. A stacked denoising auto-encoder can be made by stacking together
	multiple such single layers.
	'''
	def __init__(self, numpy_rng, theano_rng, x_lab, x_unlab, y_lab,  n_inputs, n_hiddens, w, bhid=None, bvis=None, corruption=0.1, learning_rate=0.01, activation='sigmoid', kl_div=True):
		self.numpy_rng = numpy_rng
		self.n_inputs = n_inputs
		self.n_hiddens = n_hiddens
		self.corruption = corruption
		self.w = w
		self.bhid = bhid
		self.bvis = bvis
		self.activation = activation
		self.kl_div = kl_div
		self.learning_rate = learning_rate
		print n_inputs, n_hiddens

		if not theano_rng:
			theano_rng = RandomStreams(numpy_rng.randint( 2 ** 30))

		self.theano_rng = theano_rng

		if not w:
			self.w = theano.shared(
			value=np.asarray(
				numpy_rng.uniform(
					low =-2*np.sqrt(2. / (n_inputs + n_hiddens)),
					high= 2*np.sqrt(2. / (n_inputs + n_hiddens)),
					size=(n_inputs, n_hiddens) 
					),
					dtype=theano.config.floatX
				),
					name='w',
					borrow=True
			)

		print self.w.get_value().shape
		if not bhid:
			self.bhid = theano.shared(value=np.zeros(n_hiddens, dtype=theano.config.floatX), name='bhid', borrow=True)
		
		if not bvis:
			self.bvis = theano.shared(value=np.zeros(n_inputs, dtype=theano.config.floatX), name='bvis', borrow=True)

		self.w_prime = self.w.T

		self.delta_w = theano.shared(value=np.zeros_like(self.w.get_value()), name='delta_w', borrow=True)
		self.delta_bhid = theano.shared(value=np.zeros_like(self.bhid.get_value()), name='delta_bhid', borrow=True)
		self.delta_bvis = theano.shared(value=np.zeros_like(self.bhid.get_value()), name='delta_bvis', borrow=True)
		self.params = [self.w, self.bhid, self.bvis]
		self.delta_params = [self.delta_w, self.delta_bhid, self.delta_bvis]	

		self.x_unlab = T.matrix(name='x_unlab')
		self.x_lab = T.matrix(name='x_lab')
		self.y_lab = T.matrix(name='x_unlab')


		self.x_lab_np = x_lab
		self.x_unlab_np = x_unlab
		self.y_lab_np = y_lab





	def make_corruption(self, X):
		pass


	def get_hidden_output(self, X):
		self.x = X 
		if self.activation == 'sigmoid':
			return T.nnet.sigmoid(T.dot(self.x, self.w) + self.bhid)
		elif self.activation == 'tanh':
			return T.tanh(T.dot(self.x, self.w) + self.bhid)
		elif self.activation == 'relu':
			return T.max(0, T.dot(self.x, self.w) + self.bhid)


	def get_reconstructed_output(self, hidden):
		if self.activation == 'sigmoid':
			return T.nnet.sigmoid(T.dot(hidden, self.w_prime) + self.bvis)
		elif self.activation == 'tanh':
			return T.tanh(T.dot(hidden, self.w_prime) + self.bvis)
		elif self.activation == 'relu':
			return T.max(0, T.dot(hidden, self.w_prime) + self.bvis)

	@staticmethod
	# wrapper method for sklearn method of nearest neighbours.
	def get_neighbour(x, x_full, num_neighbours=10):
		n_samples, n_dims = x_full.shape[0], x_full.shape[1]
		indices = np.arange(n_samples)
		indices_neighbours = np.random.random_integers(0, n_samples-1, num_neighbours)
		nbs = x_full[indices_neighbours]
		return nbs


	def get_all_neighbours(xvec, X, num_neighbours=10, mode='batch'):
		n_samples = xvec.shape[0]
		n_dims = xvec.shape[1]
		nbs_list = []
		nbs_list = np.array((num_neighbours*n_samples, 1))
		for i in xrange(n_samples):
			x = xvec[i]
			if mode == 'batch':
				nbs = self.get_neighbour(x, xvec, num_neighbours)
			elif mode == 'full':
				nbs = self.get_neighbour(x, X, num_neighbours)

			nbs_list[]




	def get_cost_updates(self, corruption_level=0.2, lr =0.05, momentum=0.3):
		""" This function computes the cost update after one epoch of training."""
		corrupted_x = self.get_corrupted_input(self.x_unlab, corruption_level)
		num_neighbours = 10
		loss_tensor = T.fmatrix('loss')

		neighbourhood_x = self.get_neighbours(self.x_unlab_np, num_neighbours)
		y = self.get_hidden_output(corrupted_x)
		z = self.get_reconstructed_output(y)

		#  if activation is tanh, then use squared difference as the objective ...
		# still not suer why people do like that, have to study more about this ...
		if self.activation == 'sigmoid':
			cost = -T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1-z), axis=1)
		elif self.activation == 'tanh':
			cost = T.sum((self.x - z)*(self.x - z), axis=1)

		full_neighbour = np.zeros((self.x.shape[0], num_neighbours))
		for i in xrange(len(x)):
			neighbourhood_x = self.get_neighbourhood(self.x[i], num_neighbours)
			full_neighbour[i*num_neighbours:(i+1)*num_neighbours -1] = neighbourhood_x
			for j in xrange(num_neighbours):
				loss_tensor[i,j] = calcSimilarity(self.x[i], neighbourhood_x[j])



		full_x_unlab = np.tile(self.x_unlab_np, (num_neighbours,1))

		cost_neighbour = T.sum((full_x_unlab - full_neighbour)*(full_x_unlab - full_neighbour), axis=1)
		cost_neighbour = T.mean(cost_neighbour)

		# if self.kl_div:
		# 	cost = cost + self.kl_divergence(x, z)
		cost = T.mean(cost + cost_neighbour)

		#@TODO: add regularisation term to the cost variable. 
		# regularise_penalty = l2penalty(self.w)

		gparams = T.grad(cost, wrt=self.params)
		updates = [(p, p - self.learning_rate*gp) for p,gp in zip(self.params, gparams)]
		return (cost, updates)



	def get_corrupted_input(self, X, corruption_level):
		return self.theano_rng.binomial(size=X.shape, n=1, p=1 - corruption_level, dtype=theano.config.floatX) * X 



	def compute_cost(self, X):
		pass


	@staticmethod	
	def kl_divergence(x, x_hat):
		return x*T.log(x/x_hat) + (1-x) * T.log((1-x) / (1-x_hat))
