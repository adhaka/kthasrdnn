# @author:Akash
# @package:bayesiandnn


import numpy as np 
import cPickle, gzip
import theano
import math
import collections as C 
import theano.tensor as T 



class HiddenLayer(object):
	'''
	Our basic minimal and functional hidden layer class ...
	'''	
	def __init__(self, rng, n_inputs, n_outputs, init_w=None, init_b=None, activation='tanh'):
		# self.w = theano.shared(value=np.asarray(np.uniform((n_inputs, n_outputs), dtype=theano.config.floatX)), name='w', borrow=True)
		if init_w :
			self.w = init_w
		else:
			self.w = theano.shared(
            	value=np.asarray(
                	rng.uniform(
                    	low=-6*np.sqrt(6. / (n_inputs + n_outputs)),
                    	high=6*np.sqrt(6. / (n_inputs + n_outputs)),
                    	size=(n_inputs, n_outputs)
                	),
                	dtype=theano.config.floatX),
            	name='w',
            	borrow=True
        	)
		# self.w = theano.shared(value=np.zeros((n_inputs, n_outputs), dtype=theano.config.floatX), name='w', borrow=True)

		if init_b:
			self.b = init_b
		else:
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
		elif self.activation == 'linear':
			return T.dot(X, self.w) + self.b

		# return self.output


# getter functions start here .........

	def get_weight(self):
		w_np = self.w.get_value()
		return w_np


	def get_params(self):
		return self.params


	def get_delta_params(self):
		return self.delta_params


	def get_bias(self):
		return self.b.get_value()




class LogisticRegression(object):
	'''
	This is the implementation of a softmax output layer for multi class target kind of output.
	
	'''
	def __init__(self, rng, n_inputs, n_outputs, activation='sigmoid', init_zero=True):
		np.random.seed(seed=21222)
		# super(LogisticRegression, self).__init__(rng, n_inputs, n_outputs, activation='sigmoid')
		if init_zero == True:
			self.w = theano.shared(value=np.zeros((n_inputs, n_outputs), dtype=theano.config.floatX), name='W', borrow=True)
		else:
			self.w = theano.shared(value=np.random.randn(n_inputs, n_outputs).astype(theano.config.floatX),  name='W', borrow=True)
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


	# two ways of calculating the cost function, basically this is the objective function, and its value has to be minimised,
	# categorical cross-entropy error implementation where the classes are from: 0 to C-1.


	def cost(self, X, y):
		p1 = self.calcProb(X)
		c = -T.mean(T.log(p1)[ T.arange(y.shape[0]), y])
		return c 


	def calcAccuracy(self, X, y):
		estimates = self.predict(X)
		return T.mean(T.eq(y, estimates))


# hacky way of doing it, but this function collapses the set of 48 phonemes into 39 phonemes.
	def calcAccuracyTimitMono(self, X, y):
		# estimates = self.predict(X)s
		resultdict = {}
		num_phonemes = 48
		num_eval_phonemes = 39
		estimates = self.predict(X)
		t1 = estimates.eval()
		# print t1


	def calcAccuracyTimitTri(self,X, y):
		pass 


	# getter functions ..

	def get_weight(self):
		return self.w.get_value()

	def get_bias(self):
		return self.b.get_value()

	#  get params in tensor format ....
	def get_params(self):
		return self.params


	def set_weight(self):
		pass




# class which implements Logistic Regression 
class LogisticRegression1(object):
	def __init__(self, rng, n_inputs, n_outputs, activation='sigmoid', init_zero=True):
		numpy.random.seed(seed=1111)
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
		# return T.mean(T.eq(estimates, y))
		return T.mean(T.eq(y, estimates))


	
# hacky way of doing it, but this function collapses the set of 48 phonemes into 39 phonemes.
	def calcAccuracyTimitMono(self, X, y):
		# estimates = self.predict(X)s
		resultdict = {}
		num_phonemes = 48
		num_eval_phonemes = 39
		estimates = self.predict(X)
		t1 = estimates.eval()
		print t1


	def calcAccuracyTimitTri(self,X, y):
		pass 