#  this is a package to implement independent and single helper funcitonalities which dont go specifically in any other module.
#  They are clubbed here together at one place.
import numpy as np 
import os, sys
import theano
import theano.tensor as T 
from theano import config

# this function will only work for MNIST or any other dataset where the classes are ordered from 0-K.
def one_of_K_encoding(out, num_classes):
	if type(out) == list:
		numel = len(out)
		out = np.asarray(out)

	new_out = np.zeros((out.shape[0], num_classes))
	numel = new_out.shape[0] 
	for i in xrange(numel):
		val = out[i]
		new_out[i][val] = 1
	return new_out



def reduce_encoding(out, start_class=0):
	pass


# this function maps output list with classes with new classes which range from 0 to num_classes - 1. 
def make_classes_ordered(targets):
	out_classes = list(set(targets))
	out_classes = out_classes.sort()
	new_classes = range(len(out_classes))
	classes_map_new_to_old = {}
	classes_map_old_to_new = {}
	
	for k,v in zip(new_classes, out_classes):
		classes_map_new_to_old[k] = v
		classes_map_old_to_new[v] = k

	new_targets = [classes_map[k] for k in targets]
	return new_targets, classes_map_old_to_new, classes_map_new_to_old



def get_shared(x, borrow=True):
	x_shared = theano.shared(np.asarray(x, dtype=x.dtype), borrow=borrow)
	return x_shared

def get_shared_int(y, borrow=True):
	y_shared = theano.shared(np.asarray(y, dtype=y.dtype), borrow=borrow)
	return T.cast(y_shared, 'int32')


# def init_shared_weight(inp_dims, out_dims):
# 	numpy_rng = 
# 	w = theano.shared(
# 		value=np.asarray(

# 			))



class ObjDict(dict):
	def __getattr__(self, name):
		if name in self:
			return self[name]
		else:
			raise AttributeException("The key does not exist.")

	def __setattr__(self, name, value):
		self[name] = value


	def __delattr__(self, name):
		if name in self:
			del self[name]
		else:
			raise AttributeException("The key does not exist.")


