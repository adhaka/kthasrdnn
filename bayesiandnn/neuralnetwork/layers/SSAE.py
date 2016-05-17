import numpy as np
import os
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams 


class SSDAELayer(object):

	# class variable to keep track of layers created 
	__layer_nums = count(0)
	def __init__(self, numpy_rng, theano_rng, n_inputs, n_outputs, n_targets, x_lab=None, x_unlab=None, y_lab=None, learning_rate = 0.014, corruption=0.20, batch_size=400, tied=False, activation='tanh'):
		self.numpy_rng = numpy_rng
		self.theano_rng = theano_rng
		self.n_inputs = n_inputs
		self.n_outputs = n_outputs 
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
		beta=1
		alpha = 10
		lr = Learning_Rate_Linear_Decay(start_rate=0.02)

		# accuracy = self.softmaxLayer.calcAccuracy(out_lab, y_lab)
		# cost_reconstruction_unlab = T.mean((z_unlab-x_unlab)*(z_unlab-x_unlab))
		# cost_reconstruction_lab = T.mean((z_lab - x_lab)*(z_lab - x_lab))  
		if self.activation == 'sigmoid':
			crl = -T.sum(self.x_lab * T.log(z_lab) + (1 - self.x_lab) * T.log(1 - z_lab), axis=1)
			cost_reconstruction_lab = T.mean(crl)
			cost_reconstruction_unlab = T.mean(-T.sum(self.x_unlab * T.log(z_unlab) + (1 - self.x_unlab) * T.log(1-z_unlab), axis=1))
		elif self.activation == 'tanh':
			cost_reconstruction_lab = T.mean(T.sum((self.x_lab - z_lab)*(self.x_lab - z_lab), axis=1), axis=0)
			cost_reconstruction_unlab = T.mean(T.sum((self.x_unlab - z_unlab)*(self.x_unlab - z_unlab), axis=1), axis=0)

		preds = self.softmaxLayer.predict(out_lab)
		accuracy = self.softmaxLayer.calcAccuracy(out_lab, self.y_lab)
		cost_classification = self.softmaxLayer.cost(out_lab, self.y_lab) 
		cost1 = beta * (cost_reconstruction_lab + cost_reconstruction_unlab) 
		cost2 = alpha * cost_classification
		cost = beta * (cost_reconstruction_lab + cost_reconstruction_unlab) + alpha * cost_classification  
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


