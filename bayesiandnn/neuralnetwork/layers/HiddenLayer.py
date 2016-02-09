import numpy as np 
import cPickle, gzip
import theano
import math
import collections as C 
import theano.tensor as T 


class HiddenLayer(object):

	def __init__(self, rng, n_inputs, n_outputs, activation='tanh'):
		# self.w = theano.shared(value=np.asarray(np.uniform((n_inputs, n_outputs), dtype=theano.config.floatX)), name='w', borrow=True)
		self.w = theano.shared(
            value=np.asarray(
                rng.uniform(
                    low=-4*np.sqrt(6. / (n_inputs + n_outputs)),
                    high=4*np.sqrt(6. / (n_inputs + n_outputs)),
                    size=(n_inputs, n_outputs)
                ),
                dtype=theano.config.floatX),
            name='w',
            borrow=True
        )
		# self.w = theano.shared(value=np.zeros((n_inputs, n_outputs), dtype=theano.config.floatX), name='w', borrow=True)
		# self.w = theano.shared(value=0.01 * np.ones((n_inputs, n_outputs), dtype= theano.config.floatX), name='W')

		self.b = theano.shared(value=np.zeros((n_outputs,), dtype=theano.config.floatX), name='b', borrow=True)

		self.delta_w = theano.shared(value = np.zeros_like(self.w.get_value(borrow=True)), name = 'delta_w')
		self.delta_b = theano.shared(value = np.zeros_like(self.b.get_value(borrow=True)), name = 'delta_b')
		
		self.activation = activation
		self.params = [self.w, self.b]
		self.delta_params = [self.delta_w, self.delta_b]



	def  output(self, X):
		# TODO: activation for ReLu.

		if self.activation == 'sigmoid':
			return 1 / (1 + T.exp(-T.dot(X, self.w) - self.b)) 
		elif self.activation == 'tanh':
			return T.tanh(T.dot(X, self.w) + self.b)
		elif self.activation == 'relu':
			val = T.dot(X, self.w) + self.b
			return val*(val > 0)

		# return self.output




class LogisticRegression(object):
	def __init__(self, rng, n_inputs, n_outputs, activation='sigmoid'):
		# super(LogisticRegression, self).__init__(rng, n_inputs, n_outputs, activation='sigmoid')
		self.w = theano.shared(value=np.zeros((n_inputs, n_outputs), dtype=theano.config.floatX), name='W', borrow=True)
		self.b = theano.shared(value=np.zeros((n_outputs, ), dtype=theano.config.floatX), name='b', borrow=True)
		self.delta_w = theano.shared(value = np.zeros_like(self.w.get_value(borrow=True)), name = 'delta_w')
		self.delta_b = theano.shared(value = np.zeros_like(self.b.get_value(borrow=True)), name = 'delta_b')

		self.activation = activation
		self.activation = 'tanh'
		self.params = [self.w, self.b]
		self.delta_params = [self.delta_w, self.delta_b]


	def output(self, X):
		out = super(LogisticRegression, self).output()


	def calcProb(self, X):
		return T.nnet.softmax(T.dot(X, self.w) + self.b)


	def predict(self, X):
		return T.argmax(self.calcProb(X), axis=1)


	# two ways of calculating the cost function, basically this is the objective function, and its value has to be minimised
	def cost(self, X, y):
		p1 = self.calcProb(X)
		c = -T.mean(T.log(p1)[ T.arange(y.shape[0]), y])
		return c 


	def calcAccuracy(self, X, y):
		estimates = self.predict(X)
		return T.mean(T.eq(estimates, y))




