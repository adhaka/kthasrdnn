import numpy as np
import os
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams 


# in autoencoders, each layer is greedily trained with respect to the final output.

class DAE(object):
	def __init__(self, numpy_rng, theano_rng, inp, n_inputs, n_hiddens, w, bhid=None, bvis=None, corruption=0.30, config=1):
		self.numpy_rng = numpy_rng
		self.n_inputs = n_inputs
		self.n_hiddens = n_hiddens
		self.corruption = corruption
		self.w = w
		self.bhid = bhid
		self.bvis = bvis
		print n_inputs, n_hiddens
		# exit()

		if not theano_rng:
			theano_rng = RandomStreams(numpy_rng.randint( 2 ** 30))

		self.theano_rng = theano_rng

		if not w:
			self.w = theano.shared(
			value=np.asarray(
				numpy_rng.uniform(
					low =-4*np.sqrt(6. / (n_inputs + n_hiddens)),
					high=-4*np.sqrt(6. / (n_inputs + n_hiddens)),
					size=(n_inputs, n_hiddens) 
					),
					dtype=theano.config.floatX
				),
					name='w',
					borrow=True
			)

		print self.w.get_value().shape
		# exit()

		if not bhid:
			self.bhid = theano.shared(value=np.zeros(n_hiddens, dtype=theano.config.floatX), name='bhid', borrow=True)
		
		if not bvis:
			self.bvis = theano.shared(value=np.zeros(n_inputs, dtype=theano.config.floatX), name='bvis', borrow=True)

		self.w_prime = self.w.T

		self.delta_w = theano.shared(value=np.zeros_like(self.w.get_value()), name='delta_w', borrow=True)
		self.delta_bhid = theano.shared(value=np.zeros_like(self.bhid.get_value()), name='delta_b', borrow=True)
		self.params = [self.w, self.bhid, self.bvis]
		# self.delta_params = [self.delta_w, self.delta_bhid]
		
		if inp == None:
			x = T.matrix(name='x')
			self.x = x
		else:
			self.x = inp




	def make_corruption(self, X):
		pass


	def get_hidden_output(self, X):
		self.x = X 
		return T.nnet.sigmoid(T.dot(self.x, self.w) + self.bhid)


	def get_reconstructed_output(self, hidden):
		return T.nnet.sigmoid(T.dot(hidden, self.w_prime) + self.bvis)


	def get_cost_updates(self, corruption_level=0.35, lr =0.02, momentum=0.4):
		""" This function computes the cost update after one epoch of training."""
		corrupted_x = self.get_corrupted_input(self.x, corruption_level)
		y = self.get_hidden_output(corrupted_x)
		z = self.get_reconstructed_output(y)
		cost = -T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1-z), axis=1)
		cost = T.mean(cost)

		gparams = T.grad(cost, wrt=self.params)
		updates = [(p, p-lr*gp) for p,gp in zip(self.params, gparams)]
		return (cost, updates)



	def get_corrupted_input(self, X, corruption_level):
		return self.theano_rng.binomial(size=X.shape, n=1, p=1 - corruption_level, dtype=theano.config.floatX) * X 



	def compute_cost(self, X):
		pass



