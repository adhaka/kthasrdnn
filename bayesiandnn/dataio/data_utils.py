
import numpy as np 
import theano
import os



def make_context_feature(feature, left, right):
	feature = [feature]

	for i in xrange(left):
		feature.append(np.vstack((feature[-1][0], feature[-1][:-1])))
	feature.reverse()

	for i in xrange(right):
		feature.append(np.vstack((feature[-1][1:], feature[-1][-1])))

	return np.hstack(feature)



def make_shared(self, shared_xy):
	x, y =  shared_xy





