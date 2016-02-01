import numpy as np 
import cPickle, gzip
import theano
import math
import theano.tensor as T
from layers.HiddenLayer import HiddenLayer, LogisticRegression



# class model for implementing a deep neural network with multpile hidden layers.
# rng represents a random state generated either by numpy or theano

class DNN(object):
	def __init__(self, rng, hidden_layer, n_in=None, n_out=None, config=1):
		self.w = []
		self.b = []
		self.hidden_layer = hidden_layer
		self.rng = rng
		self.params = []
		self.n_in = n_in

		self.opLayer = LogisticRegression(hidden_layer[-1], n_out)




	def forward(self, X):
		self.X = X 
		self.layers = []
		self.n_in = X.shape[1]
		prev_out = self.n_in

		for ind, x in enumerate(hidden_layer):
			hl = HiddenLayer(rng, prev_out, hidden_layer[ind])
			self.layers.append(hl)
			prev_out = hidden_layer[ind]

		self.activations = []
		self.params = []
		inp = X
		for i in xrange(len(hidden_layer)):
			act = self.layers[i].output(inp)
			self.activations.append(act)
			self.params = self.params + self.layers[i].params
			inp = act

		self.params = self.params + self.opLayer.params		


	def cost(self, X, y):
		self.forward(X)
		estimate = self.activations[-1]
		return self.opLayer.cost(X, y)


	def calcAccuracy(self, X, y):
		self.forward(X)
		estimate = self.activations[-1]
		return self.opLayer.calcAccuracy(X,y)
		




