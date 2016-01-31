import numpy as np 
import cPickle, gzip
import theano
import math
import collections as C 
import theano.tensor as T 


class HiddenLayer(object):
	def __init__(self, rng, n_inputs, n_outputs, activation='tanh', *args, **kwargs):
		self.w = theano.shared(
			value=np.asarray(
				rng.uniform(
					low=-2*np.sqrt(6. / (n_inputs + n_outputs)),
					high=2*np.sqrt(6. / (n_inputs + n_outputs)),
					size=(n_inputs*n_outputs)
					)),
					dtype=theano.config.floatX,
					name='w',
					borrow=True
					)
				
		
		self.b = theano.shared(value=np.zeros(1, n_outputs), dtype=theano.config.floatX, name='b', borrow=True)
		self.activation = activation
		self.params = [self.w, self.b]



	def output(self, X):
		out = T.dot(X, self.w) + self.b 
		if activation=='sigmoid':
			out = 1. / (1 + T.exp(-out))
		if activation == 'tanh':
			out = T.tanh(out)
		if activation == 'relu':
			out = T.tanh(out)


		return out 
		# return out



class LogisticRegression(HiddenLayer):
	def __init__(self, rng, n_inputs, n_outputs, activation='sigmoid', *args, **kwargs):
		super(LogisticRegression, self).__init__(rng, n_inputs, n_outputs, activation='tanh', *args, **kwargs)


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


	def calcAccuracy(self, X, y):
		estimates = self.predict(X)
		return T.mean(T.eq(estimates, y))




