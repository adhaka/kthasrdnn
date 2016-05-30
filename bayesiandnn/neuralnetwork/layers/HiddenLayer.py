# @author:Akash
# @package:tmhasrdnn


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
			# self.w = theano.shared(value=np.asarray(np.random.randn(n_inputs, n_outputs),dtype=theano.config.floatX),name='w',borrow=True)
			self.w = theano.shared(
            	value=np.asarray(
                	rng.uniform(
                    	low=-2*np.sqrt(2. / (n_inputs + n_outputs)),
                    	high=2*np.sqrt(2. / (n_inputs + n_outputs)),
                    	size=(n_inputs, n_outputs)
                	),
                	dtype=theano.config.floatX),
            	name='w',
            	borrow=True
        	)
# 
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


# setter functions ...
	def set_weight(self, w_np):
		self.w.set_value(w_np)


	def set_bias(self, b_np):
		if type(b_np) in ['list', 'tuple']:
			b_np = np.asarray(b_np)
		self.b.set_value(b_np)





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
		return self.calcProb(X)


	def calcProb(self, X):
		return T.nnet.softmax(T.dot(X, self.w) + self.b)


	def predict(self, X):
		return T.argmax(self.calcProb(X), axis=1)

	# def predict_np(self, x_np):
	# 	mult = 

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
# first of all- collapse 144 phonemes hmm states to monophones ...
#  so 0,1,2 creespond to 0; 3,4,5 correspond to 1, and so on ...

	def calcAccuracyTimitMono(self, X, y):
		estimates = T.argmax(self.calcProb(X), axis=1)
		estimates_phones = estimates / 3
		y_phones = y / 3
		return T.mean(T.eq(estimates_phones, y_phones))

		# resultdict = {}
		# num_phonemes = 48
		# num_eval_phonemes = 39
		# estimates = self.predict(X)
		# t1 = estimates.eval()
		# print t1

# this is a hacky solution ... and just for first iteration ..
	def calcAccuracyTimitMono39(self, X, y):
		estimates = T.argmax(self.calcProb(X), axis=1)
		estimates_phones = estimates / 3
		y_phones = y / 3
		direct_accuracy = T.mean(T.eq(estimates, y))
		reduced_set_accuracy= T.mean(T.eq(estimates_phones, y_phones))
		# 15 == 28
		val1 = (T.eq(y_phones, 14) & T.eq(estimates_phones, 27) or (T.eq(y_phones, 27) & T.eq(estimates_phones, 14))).mean(axis=0)
		# 3 == 6
		val2 = (T.eq(y_phones, 2) & T.eq(estimates_phones, 5) or (T.eq(y_phones, 5) & T.eq(estimates_phones, 2))).mean(axis=0)
		val3 = (T.eq(y_phones, 0) & T.eq(estimates_phones, 3) or (T.eq(y_phones, 3) & T.eq(estimates_phones, 0))).mean(axis=0)
		val4 = (T.eq(y_phones, 22) & T.eq(estimates_phones, 23) or (T.eq(y_phones, 23) & T.eq(estimates_phones, 22))).mean(axis=0)
		val5 = (T.eq(y_phones, 36) & T.eq(estimates_phones, 47) or (T.eq(y_phones, 47) & T.eq(estimates_phones, 36))).mean(axis=0)
		val6 = (T.eq(y_phones, 15) & T.eq(estimates_phones, 29) or (T.eq(y_phones, 29) & T.eq(estimates_phones, 15))).mean(axis=0)
		val7 = (T.eq(y_phones, 37) & T.eq(estimates_phones, 43) or (T.eq(y_phones, 43) & T.eq(estimates_phones, 37))).mean(axis=0)
		val8 = (T.eq(y_phones, 37) & T.eq(estimates_phones, 16) or (T.eq(y_phones, 16) & T.eq(estimates_phones, 37))).mean(axis=0)
		val9 = (T.eq(y_phones, 37) & T.eq(estimates_phones, 9) or (T.eq(y_phones, 9) & T.eq(estimates_phones, 37))).mean(axis=0)
		return reduced_set_accuracy + val1 + val2 + val3 + val4 + val5 + val6 + val7 + val8 + val9



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


	# setter functions ...
	def set_weight(self, w_np):
		self.w.set_value(w_np)


	def set_bias(self, b_np):
		if type(b_np) in ['list', 'tuple']:
			b_np = np.asarray(b_np)
		self.b.set_value(b_np)



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