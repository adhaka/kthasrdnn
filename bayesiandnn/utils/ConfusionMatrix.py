# some basic util and helper functions 
#  some inplementation based on DTU SUmmer school on Deep Learning .....
import numpy as np 
import os, sys

class ConfusionMatrix(object):

	def __init__(self, classes_num, classes_names=None):
		self.classes_num = classes_num
		if classes_names == None:
			self.classes = range(classes_num)
			self.classes_names = map(lambda x: str(x), self.classes)
		else:
			self.classes = classes_names
			self.classes_names = map(lambda x: str(x), classes_names)

		self.matrix = np.zeros((self.classes_num, self.classes_num), dtype='int')
		self.percentMatrix = np.zeros((self.classes_num, self.classes_num), dtype='float32')


	def __repr__(self):
		string = ''
		firstline = ['\t']
		# str_row = ''
		firstline = firstline + self.classes_names 
		str_first = '|\t'.join(firstline)
		str_first += '\n'
		str_row = str_first
		rows = []
		for i, name in enumerate(self.classes_names):
			row = [name]
			# row.append(name)
			row.append(list(self.matrix[i,:]))
			str_row = '\t'.join(row)
			str_row += '\n'

		print str_row


	# works only for classes 
	def calculateMatrix(self, predictions, actuals):
		#  this check only is preds and estimates are nd arrays. 
		assert predictions.shape == actuals.shape
		assert len(predictions) == len(actuals)
		assert max(predictions) < self.classes_num

		predictions = predictions.flatten()
		actuals = actuals.flatten()
		for i in xrange(len(actuals)):
			self.matrix[actuals[i]][predictions[i]] += 1



	def get_error_rates(self, predictions=None, actuals=None):
		if predictions == None or actuals == None:
			self.calculateMatrix(predictions, actuals)

		tp_per_class = np.asarray(np.diag(self.matrix).flatten())
		fn_per_class = np.sum(self.matrix, axis=1).flatten() - tp_per_class
		fp_per_class = np.sum(self.matrix, axis=0).flatten() - tp_per_class
		tn_per_class = np.asarray([np.sum(self.matrix)] * self.classes_num) - fp_per_class - fn_per_class - tp_per_class
		# tn_per_class =  

		return tp_per_class, fn_per_class, fp_per_class, tn_per_class


		# fn_per_row = 
		# true_positives = np.trace(self.matrix)
		# false_positives = 

	def accuracy(self):
		tp, fn, fp, fn = self.get_error_rates()
		tp = np.sum(tp)
		self.num_samples = np.sum(np.asarray(self.mat.flatten()))
		return tp / float(self.num_samples)


	def recall(self):
		tp, fn, fp, tn = self.get_error_rates()
		return tp / float(tp + fn)

	def specificity(self):
		tp, fn, fp, tn = self.get_error_rates()
		result = tn / (tn + fp)
		return result 

	def precision(self):
		tp, fn, fp, tn = self.get_error_rates()
		return tp / float(tp + fp)

	def false_positive_rate(self):
		tp, fn, fp, tn = self.get_error_rates()
		result = fp / float(fp + tp)
		return result


	def negative_error_rate(self):
		tp, fn, fp, tn = self.get_error_rates()
		result = fn / float(fn + tn)
		return result



















