# author:Akash


import numpy as  np 
import math, random 
import theano


class Learning_Rate(object):
	def __init__(self, rate=0.04):
		self.rate = rate
		self.epochs = 1
		self.stop = False

# getter and setter methods 

	def getRate(self):
		return self.rate 

	def getEpochs(self):
		return self.epochs 

	def getStopFlag(self):
		return self.stop 

	def setRate(self, lr):
		self.rate = lr 

	def setEpochs(self, epochs):
		self.epochs = epochs

	def setStopFlag(self, signal):
		self.stop = signal



# this class keeps learning constant and implements Early Stopping .... 
class Learning_Rate_Constant(Learning_Rate):
	def __init__(self, tolerance=0.1, max_epochs=50):
		super(Learning_Rate_Constant, self).__init__()
		self.errors = [0.]
		self.delta_errors = []
		self.max_epochs = max_epochs
		self.tolerance = tolerance


	def updateError(self, error):
		self.errors.append(abs(error))
		self.delta_errors.append(abs(self.errors[-1] - self.errors[-2]))



	def _checkEarlyStopping(self):
		if self.epochs <= 3:
			return 
		val1 = self.delta_errors[-1]
		val2 = self.delta_errors[-2]
		if val1 < round(self.tolerance*val2, 3):
			self.stop = True



	def updateRate(self):
		self.epochs += 1
		self._checkEarlyStopping()
		if self.epochs >= 50 or self.stop == True:
			self.stop = True
			self.rate = 0.0

		return self.rate

	def getRate(self):
		return self.rate



# this implements learning rate scheme with linear_decay
class Learning_Rate_Linear_Decay(Learning_Rate):
	def __init__(self, start_rate=0.06, end_rate=0.005, decay_rate=0.5, min_epochs_start_decay=10, min_error_diff=0.001, tolerance=0.2):
		super(Learning_Rate_Linear_Decay, self).__init__(start_rate)
		self.end_rate = end_rate
		self.rate = start_rate
		self.decay_rate = decay_rate
		self.tolerance = tolerance
		self.min_epochs_start_decay = min_epochs_start_decay
		self.min_error_diff = min_error_diff
		self.errors = [0.]
		self.delta_errors = []
		self.stop = False


	def updateError(self, error):
		self.errors.append(abs(error))
		self.delta_errors.append(abs(self.errors[-1] - self.errors[-2]))


	def _checkEarlyStopping(self):
		if self.epochs <= self.min_epochs_start_decay:
			return 
		val1 = abs(self.delta_errors[-1])
		val2 = abs(self.delta_errors[-2])
		if (self.delta_errors[-1] < self.min_error_diff):
			self.stop = True
			self.rate = 0.0

		if val1 < round(self.tolerance*val2, 4) and self.epochs >= self.min_epochs_start_decay:
			self.rate = self.decay_rate*self.rate


	def updateRate(self):
		self.epochs += 1
		# self.updateError(error)
		self._checkEarlyStopping()
		
		# one final check
		if self.stop == True:
			self.rate = 0.0

		return self.rate 


	def getRate(self):
		return self.rate


# Learning Rate optimised by a GP coming soon here .... :)










