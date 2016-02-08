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
		self.hidden_layer = hidden_layer
		self.rng = rng
		self.params = []
		self.n_in = n_in
		self.rng = rng
		self.layers = []
		self.x = None

		self.opLayer = LogisticRegression(rng, hidden_layer[-1], n_out)
		# self.X = X 
		prev_out = self.n_in
		self.params = []
		self.delta_params = []

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
		self.x = X
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
		

	def prettyprint():
		print self.w.get_value()
		print self.b.get_value()


