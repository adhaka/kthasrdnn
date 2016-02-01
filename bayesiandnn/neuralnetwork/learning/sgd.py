
import numpy as np 
import os
import math
import theano
import random
from theano import tensor as T 


#  function to perform stochastic gradient descent

def sgd(nn, data, name='', lr=0.01, momentum=0.9, epochs = 200, batch_size=500):
	train_set_x, train_set_y = data[0]
	valid_set_x, valid_set_y = data[1]
	test_set_x, test_set_y = data[2]

	num_samples = train_set_x.get_value(borrow=True).shape[0]
	num_batches = num_samples / batch_size + 1

	layers = nn.layers
	params = nn.params

	x = T.dmatrix('x')
	y = T.ivector('y')

	cost = nn.cost(X, y)
	accuracy = nn.opLayer.calcAccuracy(X, y)
	





