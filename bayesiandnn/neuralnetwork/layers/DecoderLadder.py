import  numpy as np
import theano
import theano.tensor as T 
from collections import OrderedDict
from theano.tensor.shared_randomstreams import RandomStreams 


class DecoderLadder(object):
	def __init__(self, z_denoised_top, encoder, encoder_corrupt, n_outputs, hyper_params, top=False):
		self.top = top
		self.V = theano.shared(
			value=np.asarray(
				rng.uniform(
					low=-4*np.sqrt(4. /(n_inputs + n_outputs)),
					high=4*np.sqrt(4. /(n_inputs + n_outputs)),
					size=(n_inputs, n_outputs)
					),
					dtype=theano.config.floatX),
					name='W',
					borrow=True
					)		

		h_clean, d_clean = encoder.get_layer_params()
		h_corrupt, d_corrupt = encoder_corrupt.get_layer_params()
		self.z_denoised_top = z_denoised_top
		self.z_enc_clean = d_clean['unlabelled']['z']
		self.z_enc_corrupt = d_corrupt['unlabelled']['z']
		self.n_inputs = z_denoised_top.get_value(borrow=True).shape[0]
		self.z_est = None

		#  should contain a list of 10 elements ...
		self.hyper_params_values = hyper_params
		self.init_hyper_params()
		self.mu = d['unlabelled']['mu']
		self.var = d['unlabelled']['sigma']
		# self.n_outputs


	def init_hyper_prams(self):
		a1_val = self
		init_param = lambda val, name:theano.shared(value=val * np.ones(self.n_outputs,), name=name)
		self.a1 = init_param(self.hyper_params_values[0], 'a1')
		self.a2 = init_param(self.hyper_params_values[1], 'a2')
		self.a3 = init_param(self.hyper_params_values[2], 'a3')
		self.a5 = init_param(self.hyper_params_values[4], 'a5')
		self.a6 = init_param(self.hyper_params_values[5], 'a6')
		self.a7 = init_param(self.hyper_params_values[6], 'a7')
		self.a8 = init_param(self.hyper_params_values[7], 'a8')
		self.a9 = init_param(self.hyper_params_values[8], 'a9')
		self.a10 = init_param(self.hyper_params_values[9], 'a10')


	def _decode(self):
		if self.top:
			self.u = 
		else:
			self.u = T.dot(self.z_denoised_top, self.V)

		self.batch_normalize(self.u)
		mu = self.a1 * T.nnet.sigmoid(self.a2 * self.u + self.a3) + T.dot(self.a4, self.u) + self.a5
		var = self.a6 * T.nnet.sigmoid(self.a7 * self.u + self.a8) + T.dot(self.a9, self.u) + self.a10 
		z_est = (self.z_enc_corrupt - mu)*var + mu
		z_est_bn = (z_est - self.mu) / T.sqrt(self.var)
		self.z_est = z_est
		self.z_est_bn = z_est_bn
		diff_vec = self.z_enc_clean - self.z_est_bn 
		self.reconstruction_cost =  T.sum(T.sum(T.square(diff_vec), axis=1), axis=0)
		
		# reconstruction cost should be a scalar quantity
		return self.reconstruction_cost


	def getCost(self):
		return self.reconstruction_cost


    def batch_normalize(self, z, mean=None, var=None):
    	mu = mean
    	sigma = var
    	if not mu or not sigma:
    		mu, sigma = self._calculate_moments(z)
    	mu_mat = mu.dimshuffle(0,'x')
    	sigma_mat = sigma.dimshuffle(0, 'x')
    	z_norm = (z - mu_mat) / (T.sqrt(sigma_mat + 1e-10))
    	return z_norm























