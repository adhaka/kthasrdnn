
import numpy as np 
import math
import csv
import theano
import random
import theano.tensor as T
from theano import function
from collections import OrderedDict
from os import sys, path
# sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from utils.LearningRate import Learning_Rate_Linear_Decay

# from utils import Learning_Rate.Learning_Rate_Linear_Decay
# from utils import Learning_Rate_Linear_Decay


#  function to perform stochastic gradient descent

def bsgd(nn, data, name='sgd', lr=0.06, alpha=0.3, batch_size=300, epochs=20, percent_data=1.):

	train_set_x, train_set_y = data[0]
	valid_set_x, valid_set_y = data[1]
	test_set_x, test_set_y = data[2]

	num_samples = train_set_x.get_value(borrow=True).shape[0] 
	num_batches = int((num_samples / batch_size)  * percent_data)

	layers = nn.layers
	x = T.matrix('x')
	y = T.ivector('y')
	y_eval = T.ivector('y_eval')

	cost = nn.cost(x, y)
	accuracy = nn.calcAccuracy(x, y)
	accuracy_phonemes = nn.calcAccuracyTimit(x, y)
	params = nn.params
	delta_params = nn.delta_params

	print theano.pp(cost)
	LR = Learning_Rate_Linear_Decay(start_rate=lr)
	# theano.pp(accuracy)
	index = T.ivector('index')
	learning_rate = T.scalar('learning_rate')
	momentum = T.scalar('momentum')

	p_grads = [T.grad(cost=cost, wrt = p) for p in params]  
	alpha = 0.2
	lr = 0.02
	# implementing gradient descent with momentum 
	
	print p_grads
	updates = OrderedDict()
	for dp, gp in zip(delta_params, p_grads):
		updates[dp] = dp*momentum - gp*learning_rate
	for p, dp in zip(params, delta_params):
		updates[p] = p + updates[dp]

	# updates = [(p, p - lr*gp) for p, gp in zip(params, p_grads)]



	batch_sgd_train = theano.function(inputs=[index, theano.Param(learning_rate, default=0.045), theano.Param(momentum, default=0.3)], outputs=[cost, accuracy, accuracy_phonemes], updates=updates, givens={x: train_set_x[index], y:train_set_y[index]})

	batch_sgd_valid = theano.function(inputs=[], outputs=[nn.calcAccuracy(x, y)], givens={x: valid_set_x, y:valid_set_y})

	batch_sgd_test = theano.function(inputs=[], outputs=[nn.calcAccuracy(x, y), nn.calcAccuracyTimit(x, y), nn.calcAccuracyTimitMono39(x, y)], givens={x: test_set_x, y:test_set_y})

	indices = np.arange(num_samples,  dtype=np.dtype('int32'))
	np.random.shuffle(indices)
	train_error_epochs = []


	# this function takes a list as input and computes a new list which is basically a diff list .
	def get_diff_list(li):
		li = [0] + li
		lidiff = []
		for i in xrange(len(li)-1):
			lidiff.append(abs(li[i+1] - li[i]))

		return lidiff


	ofile  = open('train_log.csv', "wb")
	train_log_w = csv.writer(ofile, delimiter=' ')

	for n in xrange(epochs):
		np.random.shuffle(indices)
		train_accuracy = []
		for i in xrange(num_batches):
			batch = indices[i*batch_size: (i+1)*batch_size]
			c,a1,a2 = batch_sgd_train(index=batch, learning_rate=LR.getRate(), momentum=alpha)
			train_accuracy.append(a1)
		print LR.getRate()
		if LR.getRate() == 0:
			break
		wt = nn.get_weight()
		# print np.mean(wt[0].flatten()), np.mean(wt[1].flatten()), np.mean(wt[2].flatten())
		valid_accuracy = batch_sgd_valid()

		
		log_n = ["epoch:", str(n), "train_accuracy:", str(np.mean(a1)), " train_accuracy_phonemes:", str(np.mean(a2)) , " validation_accuracy:", str(valid_accuracy[0])]
		train_log_w.writerow(log_n)

		print "epoch:", str(n), "train_accuracy:", str(np.mean(a1)), " train_accuracy_phonemes:", str(np.mean(a2)) , " validation_accuracy:", str(valid_accuracy[0])
		# print "epoch:", n, "  train accuracy", np.mean(a1)
		train_error_current = 1.0 - np.mean(a1)
		train_error_epochs.append(np.mean(a1))
		LR.updateError(error=(1.0 - valid_accuracy[0])*100.0)
		LR.updateRate()

	test_accuracy = batch_sgd_test()
	print test_accuracy[0], test_accuracy[1], test_accuracy[2]




def bsgd1(nn, data, name='sgd', lr=0.022, alpha=0.3, batch_size=500, epochs = 10):
	train_set_x, train_set_y = data[0]
	valid_set_x, valid_set_y = data[1]
	test_set_x, test_set_y = data[2]

	# valid_y_numpy = y_numpy[0]
	# test_y_numpy = y_numpy[1]
	test_y_numpy = map_48_to_39(test_y_numpy)
	valid_y_numpy = map_48_to_39(valid_y_numpy)
	print test_y_numpy

	num_samples = train_set_x.get_value(borrow=True).shape[0] 
	num_batches = num_samples / batch_size 

	layers = nn.layers
	x = T.matrix('x')
	y = T.ivector('y')
	y_eval = T.ivector('y_eval')

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

	batch_sgd_valid = theano.function(inputs=[], outputs=[nn.calcAccuracy(x, y), nn.calcAccuracyTimit(x,y)], givens={x: valid_set_x, y:valid_set_y})

	batch_sgd_test = theano.function(inputs=[], outputs=nn.calcAccuracy(x, y), givens={x: test_set_x, y:test_set_y})

	indices = np.arange(num_samples,  dtype=np.dtype('int32'))
	np.random.shuffle(indices)

	for n in xrange(epochs):
		np.random.shuffle(indices)
		for i in xrange(num_batches):
			batch = indices[i*batch_size: (i+1)*batch_size]
			batch_sgd_train(batch)

		# y_np = y.get_value()
		# print y.eval()

		print "epoch:", n,  "	validation accuracy:",  batch_sgd_valid()


	print batch_sgd_test()

	# params = nn.




def bsgd_partition(nn, data, name='sgd', lr=0.025, alpha=0.3, batch_size=500, epochs = 10):
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
	y_eval = T.ivector('y_eval')

	batch_sgd_train = theano.function(inputs=[ii, index], outputs=[cost, accuracy], updates=updates, givens={x: train_set_x[index], y:train_set_y[index]})

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
				batch = indices[i*batch_size: (i+1)*batch_size]
				batch_sgd_train([sup_index, batch])

		print "validation accuracy:",  batch_sgd_valid()


	print batch_sgd_test()



def map_1_to_1(y, a1, a1_to):
	a1 = 14
	a1_to = 27
	ind = [np.where(y == a1)]
	for i in ind:
		y[i] = a1_to
	return y



def map_48_to_39(y):
	a1 = [0, 2, 22, 27, 29, 36, 37, 37, 37]
	a2 = [3, 5, 23, 14, 15, 47, 49, 16, 43]

	for m,n in zip(a2, a1):
		y = map_1_to_1(y, m,n)
	return y


