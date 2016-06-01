# @author:Akash
# @package:bayesiandnn

import numpy as np
import math
import theano
import theano.tensor as T
from theano.printing import debugprint, pprint
from theano import pp
from theano.tensor.shared_randomstreams import RandomStreams
from collections import OrderedDict
from itertools import count 

from DNN import DNN
from layers.HiddenLayer import HiddenLayer, LogisticRegression
from learning.sgd import *
from datasets import mnist
from utils import utils



# single point class for implementing semi-supervised training of DNN with autoencoder penalty.
class SSDAE(object):
	def __init__(self, numpy_rng, hidden_layers, x_lab_np, y_lab_np, x_unlab_np, alpha=100, beta=3, batch_size=500, theano_rng=None, activation='tanh'):
		self.numpy_rng = numpy_rng
		self.theano_rng = theano_rng
		self.batch_size = batch_size
		self.hidden_layers = hidden_layers
		self.alpha = alpha
		self.beta = beta

		# initializing the alpha values for all the layers ..... 
		# if not isinstance(alpha, list):
		# 	self.alpha = [alpha] * len(hidden_layers) 

		self.theano_rng = RandomStreams(numpy_rng.randint( 2 ** 30))
		self.num_layers = len(hidden_layers)
		self.params_layers = []
		self.x_lab_np = x_lab_np
		self.x_unlab_np = x_unlab_np
		self.y_lab_np = y_lab_np

		# y with one of K encoding ....
		# self.y_lab_np_1_K = utils.one_of_K_encoding(y_lab_np, num_classes=10)

		self.x_lab = T.matrix('x_lab')
		self.x_unlab = T.matrix('x_unlab')
		self.y_lab = T.ivector('y_lab')
		# self.x_total = 
		self.num_samples = self.x_lab_np.shape[0] + self.x_unlab_np.shape[0]
		input_size = self.x_lab_np.shape[1]
		self.input_size = input_size

		target_size = len(list(set(utils.reduce_encoding(y_lab_np))))
		self.target_size = target_size
		output_size = hidden_layers[0]

		self.layers = []
		self.params = []
		for i, hl in enumerate(hidden_layers):
			if i == 0:
				input_lab = self.x_lab
				input_unlab = self.x_unlab
				out_lab = self.y_lab
			if i > 0:
				out_lab = self.y_lab
				input_lab = self.layers[-1].output(input_lab)
				input_unlab = self.layers[-1].output(input_unlab)
				input_size = hidden_layers[i-1]
				output_size = hl
			ssda = SSDAELayer(numpy_rng, self.theano_rng, input_size, output_size, target_size, x_lab=input_lab, y_lab=out_lab, x_unlab=input_unlab, activation=activation, alpha=self.alpha, beta=self.beta)
			# ssda = SSCAELayer(numpy_rng, self.theano_rng, input_size, output_size, target_size, x_lab=input_lab, y_lab=out_lab, x_unlab=input_unlab, activation=activation)
			self.layers.append(ssda)
			self.params = self.params + ssda.params 

		self.logLayer = LogisticRegression(self.numpy_rng, hidden_layers[-1], self.target_size, init_zero=True)


	@staticmethod
	def _shared_dataset(x, borrow=True):
		return theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=borrow)


	@staticmethod
	def _shared_dataset_y(y, borrow=True):
		rval = theano.shared(np.asarray(y, dtype=theano.config.floatX), borrow=borrow)
		return T.cast(rval, 'int32')


	def get_training_functions(self, x_lab=None, y_lab=None, x_unlab=None):
		# assert xlab.shape[0] == len(y_lab) 
		assert self.x_lab_np.shape[0] == len(self.y_lab_np)
		# self.x_lab = self._shared_dataset(self.x_lab_np)
		# self.y_lab = self._shared_dataset(self.y_lab_np)
		# self.x_unlab = self._shared_dataset(self.x_unlab_np)

		self.alpha = self.x_lab_np.shape[0] / float(self.x_lab_np.shape[0] + self.x_unlab_np.shape[0])
		index_unlab = T.ivector('index_unlab')
		index_lab = T.ivector('index_lab')
		momentum = T.scalar('momentum')
		learning_rate = T.scalar('learning_rate')

		# cost, updates = self.get_cost_updates(self.x_lab, self.x_unlab, self.y_lab)

		self.batch_size_lab = int(self.batch_size * self.alpha)
		self.batch_size_unlab = int(self.batch_size * (1-self.alpha))

		self.num_labels = self.x_lab_np.shape[0]
		self.num_unlabels = self.x_unlab_np.shape[0]
		self.num_samples = self.num_labels + self.num_unlabels

		num_batches = self.num_samples / float(self.batch_size)
		pretraining_fns = []
		for i in xrange(len(self.hidden_layers)):
			l = self.layers[i]
			# if i ==0:
			# 	x_l = self.x_lab
			# 	x_ul = self.x_unlab
			# 	y_l = self.y_lab

			# if i > 0:
			# 	x_l = self.layers[i-1].out_lab
			# 	x_ul = self.layers[i-1].out_unlab
			# 	y_l = self.y_lab
			
			# result = l.get_cost_updates(x_l, x_ul, y_l)
			result = l.get_cost_updates()
			cost1, cost2, cost3, preds, updates = result[0], result[1], result[2], result[3], result[4]

			train_fn = theano.function(inputs=[index_lab, index_unlab], updates=updates, outputs=[cost1, cost2, cost3, preds], givens={self.x_lab:x_lab[index_lab], self.x_unlab:x_unlab[index_unlab], self.y_lab:y_lab[index_lab]}, on_unused_input='warn')
			pretraining_fns.append(train_fn)

		return  pretraining_fns




# this function does the complete training for the network. single point function.
	def trainSGD(self, epochs=3):

		# if epochs is just a single value, use it for all the layers combined ....
		if not isinstance(epochs, (list, tuple)):
			epochs = epochs * len(self.hidden_layers)		
		self.num_batches = int(self.num_samples / self.batch_size)
		NUM_EPOCHS = epochs
		x_lab_shared = self._shared_dataset(self.x_lab_np)
		x_unlab_shared = self._shared_dataset(self.x_unlab_np)
		y_lab_shared = self._shared_dataset_y(self.y_lab_np)

		# get pretrining fns for all the layers ....
		pretrain_fns = self.get_training_functions(x_lab=x_lab_shared, x_unlab=x_unlab_shared, y_lab=y_lab_shared)
		print "number of labelled samples is:", self.num_labels
		print "number of unlabelled samples is:", self.num_unlabels

		indices_lab = np.arange(self.num_labels, dtype=np.dtype('int32'))
		indices_unlab = np.arange(self.num_unlabels, dtype=np.dtype('int32'))
		c = []
		c1= []
		c2 = []
		c3 = []
		c4 = []


		print "............ Pretrainining ..............."
		# pretraining loop for all the hidden layers .....
		for i in xrange(len(self.hidden_layers)):
			total_epochs = NUM_EPOCHS[i]
			la = self.layers[i]
			wc_np = la.softmaxLayer.w.get_value()

			for epoch in xrange(total_epochs):
				for j in xrange(self.num_batches - 1):
					index_lab = indices_lab[j*self.batch_size_lab:(j+1)*self.batch_size_lab]
					index_unlab = indices_unlab[j*self.batch_size_unlab:(j+1)*self.batch_size_unlab]
					res = pretrain_fns[i](index_lab=index_lab, index_unlab=index_unlab)
						# cost = 
					c.append(res[0])
					c1.append(res[1])
					c2.append(res[2])
					c3.append(res[3])
					# c4.append(res[4])
					x_full_np = np.vstack((self.x_lab_np[j*self.batch_size_lab:(j+1)*self.batch_size_lab,:], self.x_unlab_np[j*self.batch_size_unlab:(j+1)*self.batch_size_unlab,:]))

					wt_np = self.layers[i].encoder.w.get_value()
					b_np = self.layers[i].encoder.b.get_value()
					wc_np = self.layers[i].softmaxLayer.w.get_value()
					wd_np = self.layers[i].decoder.w.get_value()
					bc_np = self.layers[i].softmaxLayer.b.get_value()

				# out = np.dot(self.x_lab_np , wt_np) + b_np
				# out2 = np.dot(out, wc_np) + bc_np
				# preds_np = np.argmax(out2, axis=1)
				# preds = la.predict(out)
				# preds_np = utils.one_of_K_encoding(preds_np, 10) 

				# # print wt_np[100,1], np.mean(b_np), out[100,1], out2[100, 1]
				# print "wc  we    out    wd:", np.mean(wc_np.flatten()),  np.mean(wt_np.flatten()), np.mean(out[100,:]), np.mean(wd_np[100,:])
				print "epoch is:", epoch 
				print "cost is: %d, %d, %d", np.nanmean(c), np.nanmean(c1), np.nanmean(c2) , np.mean(c3)
				# oldWe = wt_np
				# oldWc = la.getWc()
				# la.update_Wc(self.y_lab_np_1_K, preds_np, out)
				# new_Wc = la.getWc()
				# new_We = la.update_Wc_We(x_full, oldWe, oldWc, new_Wc)
				# print np.mean(wc_np[110,:])

		# pass



	def trainSGDSupervised(self, train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y):
		# dnn = DNN(self.numpy_rng, [self.hidden_layers[-1]], self.hidden_layers[-1], 10, w_layers=[self.layers[0].encoder.get_weight()], b_layers=[self.layers[0].encoder.get_bias()])
		dnn = DNN(self.numpy_rng, [self.hidden_layers[0]], self.hidden_layers[0], 10, w_layers=[self.layers[0].encoder.get_weight()], b_layers=[self.layers[0].encoder.get_bias()])
		# mnist_data = mnist.load_mnist_theano('mnist.pkl.gz')
		# Hl = HiddenLayer(self.numpy_rng, self.input_size, self.hidden_layers[0], init_w=self.layers[0].get_weight(), init_b=self.layers[0].get_bias(), activation='tanh')
		# mnist_data = mnist.load_mnist_numpy('mnist.pkl.gz')
		print "............... Final training starts now ........."
		# bsgd(dnn, mnist_data, epochs=40)

		# train_set_x, train_set_y = mnist_data[0]
		# valid_set_x, valid_set_y = mnist_data[1]
		# test_set_x, test_set_y = mnist_data[2]

		# train_set_x, train_set_y = train_set_x[:600,:], train_set_y[:600]
		batch_size = 300
		epochs = 140

		x_final = T.matrix('x_final')
		y_final = T.ivector('y_final')
		y_eval = T.ivector('y_eval')

		bsgd(dnn, mnist_data, epochs=25, lr=0.008)

		print train_set_x.shape, self.layers[0].get_weight().shape
		z1_np = np.tanh(np.dot(train_set_x, self.layers[0].get_weight()) + self.layers[0].get_bias())
		z2_np = np.tanh(np.dot(z1_np, self.layers[1].get_weight()) + self.layers[1].get_bias())
		# z3_np = np.tanh(np.dot(z2_np, self.layers[2].get_weight()) + self.layers[2].get_bias())
		z1_valid_np = np.tanh(np.dot(valid_set_x, self.layers[0].get_weight()) + self.layers[0].get_bias())
		z2_valid_np = np.tanh(np.dot(z1_valid_np, self.layers[1].get_weight()) + self.layers[1].get_bias())
		# z3_valid_np = np.tanh(np.dot(z2_valid_np, self.layers[2].get_weight()) + self.layers[2].get_bias())
		z1_test_np = np.tanh(np.dot(test_set_x, self.layers[0].get_weight()) + self.layers[0].get_bias())
		z2_test_np = np.tanh(np.dot(z1_test_np, self.layers[1].get_weight()) + self.layers[1].get_bias())
		# z3_test_np = np.tanh(np.dot(z2_test_np, self.layers[2].get_weight()) + self.layers[2].get_bias())


		z1 = self.layers[0].encoder.output(x_final)
		z2 = self.layers[1].encoder.output(z1)
		# z3 = self.layers[2].encoder.output(z2)

		def get_shared(x, borrow=True):
			x_shared = theano.shared(np.asarray(x, dtype=x.dtype), borrow=borrow)
			return x_shared


		def get_shared_int(y, borrow=True):
			y_shared = theano.shared(np.asarray(y), borrow=borrow)
			return T.cast(y_shared, 'int32')

		# print z3_np.shape
		train_set_x_shared = get_shared(train_set_x)
		train_set_z_shared = get_shared(z1_np)
		valid_set_x_shared = get_shared(valid_set_x)
		valid_set_z_shared = get_shared(z1_valid_np)
		test_set_x_shared = get_shared(test_set_x)
		test_set_z_shared = get_shared(z1_test_np)
		train_set_y_shared = get_shared_int(train_set_y)
		valid_set_y_shared = get_shared_int(valid_set_y)
		test_set_y_shared = get_shared_int(test_set_y)


		num_samples = train_set_x.shape[0] 
		indices = np.arange(num_samples,  dtype=np.dtype('int32'))
		num_batches = num_samples / batch_size

		supervised_cost = self.logLayer.cost(x_final, y_final) 
		supervised_accuracy = self.logLayer.calcAccuracy(x_final, y_final)
		# params_supervised = self.layers[-1].encoderParams + self.logLayer.params
		# params_supervised = Hl.params + self.logLayer.params

		# exit() 
		params_supervised =  self.logLayer.params

		updates = OrderedDict()
		p_final_grads = [T.grad(cost=supervised_cost, wrt = p) for p in params_supervised] 
		lr = 0.06

		for p, gp in zip(params_supervised, p_final_grads):
			updates[p] = p - lr*gp


		index = T.ivector('index')

		batch_sgd_train_final = theano.function(inputs=[index], outputs=[supervised_cost, supervised_accuracy], updates=updates, givens={x_final: train_set_z_shared[index], y_final:train_set_y_shared[index]})

		batch_sgd_valid_final = theano.function(inputs=[], outputs=[self.logLayer.calcAccuracy(x_final, y_final)], givens={x_final: valid_set_z_shared, y_final:valid_set_y_shared})
		batch_sgd_test_final = theano.function(inputs=[], outputs=[self.logLayer.calcAccuracy(x_final, y_final)], givens={x_final: test_set_z_shared, y_final:test_set_y_shared})
		train_accuracy = []

		for n in xrange(epochs):
			for i in xrange(num_batches):
				batch = indices[i*batch_size: (i+1)*batch_size]
				c,a = batch_sgd_train_final(index=batch)
				train_accuracy.append(a)

			print "epoch:", n, "  train accuracy", np.mean(a)
			print "epoch:", n,  " validation accuracy:",  batch_sgd_valid_final()
			# self.finalLogLayer = LogisticRegression(n_inputs=self.hidden_layers[-1], n_outputs=self.n_outputs, activation='tanh', init_zero=True)

		print "test accuracy:", batch_sgd_test_final()




# move it to layers folder later .. 
class SSDAELayer(object):

	# class variable to keep track of layers created 
	__layer_nums = count(0)
	def __init__(self, numpy_rng, theano_rng, n_inputs, n_outputs, n_targets, x_lab=None, x_unlab=None, y_lab=None, learning_rate = 0.020, corruption=0.20, batch_size=400, alpha=700, beta=3, tied=False, activation='tanh'):
		self.numpy_rng = numpy_rng
		self.theano_rng = theano_rng
		self.n_inputs = n_inputs
		self.n_outputs = n_outputs 
		self.alpha = alpha
		self.beta = beta
		self.encoder = HiddenLayer(self.numpy_rng, self.n_inputs, self.n_outputs, activation=activation)

		if tied == True:
			self.decoder = HiddenLayer(self.numpy_rng, self.n_outputs, self.n_inputs, init_w=self.encoder.w.T, activation=activation)
		else:
			self.decoder = HiddenLayer(self.numpy_rng, self.n_outputs, self.n_inputs, activation=activation)
		
		self.learning_rate = learning_rate
		self.activation = activation
		self.out_lab = self.encoder.output(x_lab)
		self.out_unlab = self.encoder.output(x_unlab)
		self.layer_num = self.__layer_nums.next()
		# self.inp_lab 

		if x_lab == None:
			self.x_lab = T.matrix('inp_lab')
		else:
			self.x_lab = x_lab

		if x_unlab == None:
			self.x_unlab = T.matrix('inp_unlab')
		else:
			self.x_unlab = x_unlab

		if y_lab == None:
			self.y_lab = T.matrix('y_lab')
		else:
			self.y_lab = y_lab

		self.softmaxLayer = LogisticRegression(self.numpy_rng, n_outputs, n_targets, init_zero=False)
		self.encoderParams = self.encoder.params
		self.params = self.encoder.params + self.decoder.params
		self.paramsAll = self.encoder.params + self.decoder.params + self.softmaxLayer.params
		# self.params = self.encoder.params + self.decoder.params
		# self.delta_params = self.encoder.delta_params + self.decoder.delta_params + self.softmaxLayer.delta_params		


	@staticmethod
	def _shared_dataset(x, borrow=True):
		return theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=borrow)


	@classmethod
	def count_instances(cls):
		cls.layer_num += 1


	def output(self, x):
		out = self.encoder.output(x)
		# out_unlab = self.encoder.output(x_unlab)
		return out

#  outputs the predictions from softmax layer ...
	def predict(self, x):	
		return self.softmaxLayer.predict(x)


	def predict_np(self, wc, bc, z_np):
		softmaxout = np.dot(z_np, wc) + bc
		preds = np.argmax(softmaxout, axis=1)
		return preds

# 
	def get_cost_updates(self):
		out_unlab = self.encoder.output(self.x_unlab)
		out_lab = self.encoder.output(self.x_lab)
		z_unlab = self.decoder.output(out_unlab)
		z_lab = self.decoder.output(out_lab)
		preds_lab = self.softmaxLayer.predict(out_lab)
		self.preds_lab = preds_lab
		# alpha=0
		beta_range=[1, 10, 100, 200, 500, 800, 1000, 2000, 5000]
		# beta=700
		# alpha = 3
		print "value of alpha is:", self.alpha
		print "value of beta is:", self.beta
		lr = Learning_Rate_Linear_Decay(start_rate=0.02)

		# accuracy = self.softmaxLayer.calcAccuracy(out_lab, y_lab)
		# cost_reconstruction_unlab = T.mean((z_unlab-x_unlab)*(z_unlab-x_unlab))
		# cost_reconstruction_lab = T.mean((z_lab - x_lab)*(z_lab - x_lab))  
		if self.activation == 'sigmoid':
			crl = -T.sum(self.x_lab * T.log(z_lab) + (1 - self.x_lab) * T.log(1 - z_lab), axis=1)
			cost_reconstruction_lab = T.mean(crl)
			cost_reconstruction_unlab = T.mean(-T.mean(self.x_unlab * T.log(z_unlab) + (1 - self.x_unlab) * T.log(1-z_unlab), axis=1))
		elif self.activation == 'tanh':
			cost_reconstruction_lab = T.mean(T.mean((self.x_lab - z_lab)*(self.x_lab - z_lab), axis=1), axis=0)
			cost_reconstruction_unlab = T.mean(T.mean((self.x_unlab - z_unlab)*(self.x_unlab - z_unlab), axis=1), axis=0)

		preds = self.softmaxLayer.predict(out_lab)
		accuracy = self.softmaxLayer.calcAccuracy(out_lab, self.y_lab)
		cost_classification = self.softmaxLayer.cost(out_lab, self.y_lab) 
		cost1 = self.beta * (cost_reconstruction_lab + cost_reconstruction_unlab) 
		cost2 = self.alpha * cost_classification
		cost = self.beta * (cost_reconstruction_lab + cost_reconstruction_unlab) + self.alpha * cost_classification  
		# debugprint(cost)
		# if self.debug_mode == True:
		# theano.printing.pydotprint(cost, outfile='symbolic_graph_costx' + str(self.layer_num) + '.png', var_with_name_simple=True)
		
		updates = OrderedDict()
		gparams = T.grad(cost, wrt=self.paramsAll)
		gparams2 = T.grad(cost1, wrt=self.params)
		for p, gp in zip(self.paramsAll, gparams):
			updates[p] = p - gp*self.learning_rate


		# for p, gp in zip(self.params, gparams2):
		# 	updates[p] = p - gp*self.learning_rate

		# debugprint(cost)
		return [cost, cost1, cost_classification, accuracy, updates]


	# for a better control, this fn will take numpy arrays. 
	# make batches such that they have some respresentation from labelled data as well and if possible with the same amount of points per class.
	def train(self, x_lab_numpy, y_lab_numpy, xunlab_numpy):
		pass
		# batch_sgd_train = theano.function(inputs=[index_unlab, index_lab], outputs=[cost, accuracy], givens={xlab:xlab[]})


	def getWc(self):
		return self.softmaxLayer.get_weight()


	def setWc(self, wc):
		self.softmaxLayer.set_weight(wc)


	def get_weight(self):
		return self.encoder.get_weight()


	def get_bias(self):
		return self.encoder.get_bias()


	# function to update classifier weight manually without using the computational graph ...
	# use one-of-K  encoding here ....
	# alternative way to update Wc, using its numpy value.


	def update_Wc(self, target, output, z):
		eta = 0.01
		wc = self.getWc()
		n_rows, n_cols = wc.shape[0], wc.shape[1]
		for i in xrange(n_rows):
			for j in xrange(n_cols):
				if target[i, j] == 1:
					# wc[i,j] = wc[i,j] - eta * (target[i,j] - output[i,j])* z[i]  
					wc[i,j] = wc[i,j] - eta * (target[i,j] - output[i,j])

		self.setWc(wc)


	def  update_Wc_We(self, x, We, oldWc, newWc):
		eta = 0.01
		delta_Wc = newWc - oldWc
		error = enc_out * ( 1- enc_out)				
		We = We - eta * T.dot(x.T , error)
		return We





# semi-supervised contractive auto-encoder
class  SSCAELayer(SSDAELayer):
	''' This class represents one single  layer of Semi-Supervised 
	contractive auto-encoder , which can be stacked to gether to form 
	a contractive auto encoder whose objective will be a sum of the reconstruction 
	error and cross entropy loss and an additional term corresponfing to the frobenius norm
	of the jacbobian.
	'''
	def __init__(self, numpy_rng, theano_rng, n_inputs, n_outputs, n_targets, x_lab=None, x_unlab=None, y_lab=None, learning_rate = 0.07, corruption=0.20, batch_size=500, tie=False, activation='tanh'):
		super(SSCAELayer, self).__init__(numpy_rng, theano_rng, n_inputs, n_outputs, n_targets, x_lab, x_unlab, y_lab, learning_rate, corruption, batch_size, tie, activation)		



	def get_cost_updates(self):
		out_unlab = self.encoder.output(self.x_unlab)
		out_lab = self.encoder.output(self.x_lab)
		z_unlab = self.decoder.output(out_unlab)
		z_lab = self.decoder.output(out_lab)
		preds_lab = self.softmaxLayer.predict(out_lab)
		# alpha=0
		beta=1
		alpha = 30
		lamda = 0.1 
		if self.activation == 'sigmoid':
			crl = -T.sum(self.x_lab * T.log(z_lab) + (1 - self.x_lab) * T.log(1-z_lab), axis=1) 
			dh_lab = out_lab * (1 - out_lab) 
			dh_unlab = out_unlab * (1 - out_unlab)
			dhdx_lab = T.dot(dh_lab, self.encoder.w.T)
			dhdx_unlab = T.dot(dh_unlab, self.encoder.w.T)
			frobenius_cost_lab = lamda * T.sum((dhdx_lab ** 2), axis=1)
			frobenius_cost_unlab = lamda * T.sum((dhdx_unlab **2), axis=1)
			cost_reconstruction_lab = T.mean(crl + frobenius_cost_lab)
			crul = -T.sum(self.x_unlab * T.log(z_unlab) + (1 - self.x_unlab) * T.log(1-z_unlab), axis=1)
			cost_reconstruction_unlab = T.mean(crul + frobenius_cost_unlab)

		elif self.activation == 'tanh':
			dh_lab = out_lab * (1 - out_lab) 
			dh_unlab = out_unlab * (1 - out_unlab)
			dhdx_lab = T.dot(dh_lab, self.encoder.w.T)
			dhdx_unlab = T.dot(dh_unlab, self.encoder.w.T)
			frobenius_cost_lab = lamda * T.sum((dhdx_lab ** 2), axis=1)
			frobenius_cost_unlab = lamda * T.sum((dhdx_unlab **2), axis=1)
			crl = T.sum((self.x_lab - z_lab)*(self.x_lab - z_lab), axis=1)
			crul = T.sum((self.x_unlab - z_unlab)*(self.x_unlab - z_unlab), axis=1)
			cost_reconstruction_lab = T.mean(crl + frobenius_cost_lab)
			cost_reconstruction_unlab = T.mean(crul + frobenius_cost_unlab)

		preds = self.softmaxLayer.predict(out_lab)
		cost_classification = self.softmaxLayer.cost(out_lab, self.y_lab)

		cost1 = beta * (cost_reconstruction_lab + cost_reconstruction_unlab) 
		cost2 = alpha * cost_classification
		cost = cost1 + cost2 

		updates = OrderedDict()
		gparams = T.grad(cost, wrt=self.paramsAll)
		# gparams2 = T.grad(cost2, wrt=self.paramsAll)
		for p, gp in zip(self.paramsAll, gparams):
			updates[p] = p - gp*self.learning_rate

		return [cost, cost1, cost_classification, preds, updates]

