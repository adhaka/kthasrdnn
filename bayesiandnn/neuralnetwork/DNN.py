# @author:Akash
# @package:bayesiandnn

import numpy as np 
import cPickle, gzip
import theano
import math
import theano.tensor as T
from layers.HiddenLayer import HiddenLayer, LogisticRegression


# class model for implementing a deep neural network with multpile hidden layers.
# rng represents a random state generated either by numpy or theano

class DNN(object):
	def __init__(self, rng, hidden_layer, n_in=None, n_out=None, w_layers=None, config=1):
		self.hidden_layer = hidden_layer
		self.rng = rng
		self.params = []
		self.n_in = n_in
		self.n_out = n_out
		self.rng = rng
		self.layers = []
		self.x = None

		self.opLayer = LogisticRegression(rng, hidden_layer[-1], n_out)
		# self.X = X 
		prev_out = self.n_in
		self.params = []
		self.delta_params = []

#  construct a neural network with provided weights for each layer and then add a log layer ..

		if w_layers and len(hidden_layer) == len(w_layers):
			for i in range(len(w_layers)):
				w_np = w_layers[i]
				print w_np.shape
				print w_np
				# exit()
				# w = theano.shared(value=np.asarray(w_np, dtype=theano.config.floatX), name='We', borrow=True)
				w = theano.shared(
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
				HL = HiddenLayer(self.rng, prev_out, self.n_out, init_w=w)
				# prev_out = 
				self.params += HL.params
				self.delta_params = self.delta_params + HL.delta_params 
		else:
			for ind, h in enumerate(hidden_layer):
				HL = HiddenLayer(self.rng, prev_out, h)
				self.layers.append(HL)
				prev_out = h
				self.params += HL.params
				self.delta_params = self.delta_params + HL.delta_params 

		self.params += self.opLayer.params
		self.delta_params = self.delta_params + self.opLayer.delta_params



	def forward(self, X):
		# self.activations = []
		inp = X
		# self.x = X
		activations = [X]
		for i, l in enumerate(self.layers):
			act = l.output(inp)
			activations.append(act)
			inp = act
		return activations


	def cost(self, X, y):
		act = self.forward(X)
		estimate = act[-1]
		return self.opLayer.cost(estimate, y)



	def calcAccuracy(self, X, y):
		act = self.forward(X)
		estimate = act[-1]
		return self.opLayer.calcAccuracy(estimate, y)


	def calcAccuracyTimit(self, X, y):
		act = self.forward(X)
		estimate = act[-1]
		return self.opLayer.calcAccuracyTimitMono(estimate, y)
		

	def prettyprint(self):
		pass
		# print self.w.get_value()
		# print self.b.get_value()


	def get_weight(self):
		w_list = []
		for l in self.layers:
			print l.get_weight().shape
			w_list.append(l.get_weight())

		w_list.append(self.opLayer.get_weight())
		return w_list
