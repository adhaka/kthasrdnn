import numpy as np 
import theano
import theano.tensor as T 
from theano.tensor.shared_randomstreams import RandomStreams 

# single layer of encoder of the form z = sigm(X*W+b)
class Encoder(object):
	def __init__(self, numpy_rng, inp, n_inputs, n_outputs, corruption=0.0, theano_rng=None, activation='sigmoid') :
		self.numpy_rng = numpy_rng
		self.n_inputs = n_inputs
		self.n_outputs = n_outputs
		self.x = inp
		if in

		if not theano_rng:
			theano_rng = RandomStreams(numpy_rng.randint( 2 ** 30))

		self.w = theano.shared(
			value=np.asarray(
				numpy_rng.uniform(
					low =-4*np.sqrt(6. / (n_inputs + n_outputs)),
					high = 4*np.sqrt(6. / (n_inputs + n_outputs)),
					size = (n_inputs, n_outputs)
					),
					dtype = theano.config.floatX),
					name='w',
					borrow=True
			)

		self.b = theano.shared(value=np.zeros(n_outputs, dtype=theano.config.floatX), name='b', borrow=True)

		self.params = [self.w, self.b]

		self.delta_w = theano.shared(np.zeros_like(self.w.get_value(), dtype=theano.config.floatX), borrow=True)
		self.delta_b = theano.shared(np.zeros_like(self.b.get_value))


