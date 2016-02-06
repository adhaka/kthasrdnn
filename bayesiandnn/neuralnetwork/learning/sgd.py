
import numpy as np 
import os
import math
import theano
import random
import theano.tensor as T
from theano import function
from collections import OrderedDict


#  function to perform stochastic gradient descent

def bsgd(nn, data, name='sgd', lr=0.025, alpha=0.3, batch_size=500, epochs = 400):
	# train_set is a list of trainingsets divided into partitions

	train_set_x, train_set_y = data[0]
	valid_set_x, valid_set_y = data[1]
	test_set_x, test_set_y = data[2]


	num_samples = train_set_x.get_value(borrow=True).shape[0] 
	num_batches = num_samples / batch_size 

	layers = nn.layers
	x = T.matrix('x')
	y = T.ivector('y')

	cost = nn.cost(x, y)
	accuracy = nn.calcAccuracy(x, y)
	params = nn.params
	delta_params = nn.delta_params

	print theano.pp(cost)
	# theano.pp(accuracy)

	p_grads = [T.grad(cost=cost, wrt = p) for p in params]  
	# implementing gradient descent with momentum 
	print p_grads
	updates = OrderedDict()
	for dp, gp in zip(delta_params, p_grads):
		updates[dp] = dp*alpha - gp*lr
	for p, dp in zip(params, delta_params):
		updates[p] = p + updates[dp]

	# updates = [(p, p - lr*gp) for p, gp in zip(params, p_grads)]
	index = T.ivector('index')
	batch_sgd_train = theano.function(inputs=[index], outputs=[cost, accuracy], updates=updates, givens={x: train_set_x[index], y:train_set_y[index]})

	batch_sgd_valid = theano.function(inputs=[], outputs=nn.calcAccuracy(x, y), givens={x: valid_set_x, y:valid_set_y})

	batch_sgd_test = theano.function(inputs=[], outputs=nn.calcAccuracy(x, y), givens={x: test_set_x, y:test_set_y})

	indices = np.arange(num_samples,  dtype=np.dtype('int32'))
	np.random.shuffle(indices)

	for n in xrange(epochs):
		np.random.shuffle(indices)
		for i in xrange(num_batches):
			batch = indices[i*num_batches: (i+1)*batch_size]
			batch_sgd_train(batch)

		print "validation accuracy:",  batch_sgd_valid()


	print batch_sgd_test()

	# params = nn.




def bsgd_partition(nn, data, name='sgd', lr=0.025, alpha=0.3, batch_size=500, epochs = 400):
	# train_set is a list of trainingsets divided into partitions

	train_set_x, train_set_y = data[0]
	valid_set_x, valid_set_y = data[1]
	test_set_x, test_set_y = data[2]


	num_partitions = len(train_set_x)
	print "number of partitions:", num_partitions
	train_set_x = np.asarray(train_set_x)
	num_samples = train_set_x[0].get_value(borrow=True).shape[0] 
	num_batches = num_samples / batch_size 

	layers = nn.layers
	x = T.matrix('x')
	y = T.ivector('y')

	cost = nn.cost(x, y)
	accuracy = nn.calcAccuracy(x, y)
	params = nn.params
	delta_params = nn.delta_params

	print theano.pp(cost)
	# theano.pp(accuracy)

	p_grads = [T.grad(cost=cost, wrt = p) for p in params]  
	# implementing gradient descent with momentum 
	print p_grads
	updates = OrderedDict()
	for dp, gp in zip(delta_params, p_grads):
		updates[dp] = dp*alpha - gp*lr
	for p, dp in zip(params, delta_params):
		updates[p] = p + updates[dp]

	# updates = [(p, p - lr*gp) for p, gp in zip(params, p_grads)]
	index = T.ivector('index')
	ii = T.ivector('ii')


	batch_sgd_train = theano.function(inputs=[ii, index], outputs=[cost, accuracy], updates=updates, givens={x: train_set_x[ii][index], y:train_set_y[ii][index]})

	batch_sgd_valid = theano.function(inputs=[], outputs=nn.calcAccuracy(x, y), givens={x: valid_set_x, y:valid_set_y})

	batch_sgd_test = theano.function(inputs=[], outputs=nn.calcAccuracy(x, y), givens={x: test_set_x, y:test_set_y})

	indices = np.arange(num_samples,  dtype=np.dtype('int32'))
	np.random.shuffle(indices)

	for n in xrange(epochs):
		np.random.shuffle(indices)
		sup_indices = random.randrange(0, num_partitions)
		sup_indices = np.arange(num_partitions, dtype=np.dtype('int32'))
		for j in xrange(num_partitions):
			sup_index = sup_indices[j]
			for i in xrange(num_batches):
				# batch = [sup_index]
				batch = indices[i*num_batches: (i+1)*batch_size]
				batch_sgd_train([sup_index, batch])

		print "validation accuracy:",  batch_sgd_valid()


	print batch_sgd_test()


	





