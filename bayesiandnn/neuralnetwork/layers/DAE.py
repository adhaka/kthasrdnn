import numpy as np
import os
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from collections import OrderedDict 


# in autoencoders, each layer is greedily trained with respect to the final output.

class DAE(object):
	'''
	This class is implemented as a basic underlying layer to make a stacked denoising auto-encoder...
	This is a denoising auto-encoder single layer. A stacked denoising auto-encoder can be made by stacking together
	multiple such single layers.
	'''
	def __init__(self, numpy_rng, theano_rng, inp, n_inputs, n_hiddens, w, bhid=None, bvis=None, corruption=0.2, learning_rate=0.08, activation='sigmoid', kl_div=True):
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
					low =-6*np.sqrt(6. / (n_inputs + n_hiddens)),
					high= 6*np.sqrt(6. / (n_inputs + n_hiddens)),
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
		if inp == None:
			x = T.matrix(name='x')
			self.x = x
		else:
			self.x = inp



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



	def get_cost_updates(self, corruption_level=0.15, lr =0.05, momentum=0.3):
		""" This function computes the cost update after one epoch of training."""
		corrupted_x = self.get_corrupted_input(self.x, corruption_level)
		y = self.get_hidden_output(corrupted_x)
		z = self.get_reconstructed_output(y)
		#  if activation is tanh, then use squared difference as the objective ...
		# still not sure why people do like that, have to study more about this ...
		if self.activation == 'sigmoid':
			cost = -T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1-z), axis=1)
		elif self.activation == 'tanh':
			cost = T.sum((self.x - z)*(self.x - z), axis=1)

		# if self.kl_div:
		# 	cost = cost + self.kl_divergence(x, z)
		cost = T.mean(cost)

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




# contractive auto-encoders
class CAE(DAE):
	'''class to implement contractive auto encoders
		which include frobenius norm of the derivative of the 
		hidden layer output with respect to input to the cost term.
		serves as a single contractive auto-encoder layer.
	'''
	def __init__(self, numpy_rng, theano_rng, inp, n_inputs, n_hiddens, w, bhid=None, bvis=None, lambda_constant=0.0010, activation='tanh', kl_div=True):
		super(CAE, self).__init__(numpy_rng, theano_rng, inp, n_inputs, n_hiddens, w, bhid, bvis, corruption=0.0, activation=activation, kl_div=kl_div)
		self.lambda_constant = lambda_constant


	def get_cost_updates(self, lr =0.004, momentum=0.3):
		y = self.get_hidden_output(self.x)
		z = self.get_reconstructed_output(y)
		dy = y * (1 - y)
		dy_sq = dy ** 2
		w_t_sq = self.w.T ** 2
		dydx = T.dot(dy_sq, w_t_sq)
		frobenius_cost = T.sum(dydx, axis=1)

		if self.activation == 'sigmoid':
			cost = -T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1-z), axis=1)
		elif self.activation == 'tanh':
			cost = -T.sum((self.x - z)*(self.x - z), axis=1)

		print "lambda constant=", self.lambda_constant
		cost = cost +  self.lambda_constant * frobenius_cost 

		cost_mean = T.mean(cost)

		gparams = T.grad(cost_mean, wrt=self.params)
		updates = [(p, p - lr*gp) for p,gp in zip(self.params, gparams)]
		return (cost, updates)






	