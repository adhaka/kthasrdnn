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
	def __init__(self, tolerance=0.1, max_epochs=50, early_stopping=True):
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
		if val1 < round(self.tolerance*val2, 3) and early_stopping:
			self.stop = True



	def updateRate(self):
		self.epochs += 1
		self._checkEarlyStopping()
		if self.epochs >= 50 or self.stop == True:
			self.stop = True
			self.rate = 0.0
		return self.rate



# this implements learning rate scheme with linear_decay
class Learning_Rate_Linear_Decay(Learning_Rate):
	def __init__(self, start_rate=0.06, end_rate=0.003, decay_rate=0.7, delay_epochs=60, min_error_diff=0.001, tolerance=0.2, early_stopping=True):
		super(Learning_Rate_Linear_Decay, self).__init__(start_rate)
		self.end_rate = end_rate
		self.rate = start_rate
		self.decay_rate = decay_rate
		self.tolerance = tolerance
		self.delay_epochs = delay_epochs
		self.min_error_diff = min_error_diff
		self.errors = [0.]
		self.delta_errors = []
		self.stop = False


	def updateError(self, error):
		self.errors.append(abs(error))
		self.delta_errors.append(abs(self.errors[-1] - self.errors[-2]))


	def _checkEarlyStopping(self):
		if self.epochs <= self.delay_epochs:
			return

		err1 = self.errors[-1]
		err2 = self.errors[-2] 
		val1 = abs(self.delta_errors[-1])
		val2 = abs(self.delta_errors[-2])
		if self.rate <= self.end_rate and (self.epochs >= self.delay_epochs):
			self.stop = True
			self.rate = 0.0

		if ((err1 > err2) or val1 < round(self.tolerance*val2, 4)) and self.epochs >= self.delay_epochs:
			self._update_decay_rate()
			# self.rate = self.decay_rate*self.rate


	def _update_decay_rate(self):
		self.rate = self.decay_rate*self.rate 


	def updateRate(self):
		self.epochs += 1
		# self.updateError(error)
		self._checkEarlyStopping()
		
		# one final check
		if self.stop == True:
			self.rate = 0.0

		return self.rate 


	# def getRate(self):
	# 	return self.rate



# this implements learning rate scheme with exponential decay.
# lr = lr * scalar^( min((t - to), 0) /tau )

class Learning_Rate_Exponential_Decay(Learning_Rate_Linear_Decay):
	def __init__(self, start_rate=0.06, end_rate=0.005, decay_rate=0.5, delay_epochs=10, min_error_diff=0.001, tolerance=0.2, early_stopping=True, pow_constant=10, tau=3):
		super(Learning_Rate_Exponential_Decay, self).__init__(start_rate=start_rate, decay_rate=decay_rate, min_error_diff=min_error_diff, early_stopping=early_stopping)
		self.delay_epochs = delay_epochs
		self.pow_constant = pow_constant
		self.tau = tau


	def _update_decay_rate(self, delay_epochs=0):
		# if self.epochs -  delay  > 0:
		self.rate = self.rate * self.pow_constant ** ( min(0., -(self.epochs - self.delay_epochs)) / self.tau )




class Learning_Rate_Power_Scheduling(Learning_Rate_Linear_Decay):
	def __init__(self, start_rate=0.06, end_rate=0.005, decay_rate=0.5, delay_epochs=10, min_error_diff=0.001, tolerance=0.2, early_stopping=True, pow_constant=1, tau=3):
		super(Learning_Rate_Exponential_Decay, self).__init__(start_rate)
		self.delay_epochs = delay_epochs
		self.pow_constant = pow_constant
		self.tau = tau


	def _update_decay_rate(self, decay_epochs=0):
		self.rate = self.rate * (1 + self.epochs/self.tau) ** (-self.pow_constant)



class Learning_Rate_Performance_Scheduling(Learning_Rate):
	def __init__(self):
		super(Learning_Rate_Performance_Scheduling, self).__init__(start_rate)


	def updateRule(self):
		pass




class Learning_Rate_GP_Optimised(Learning_Rate):
	pass
# Learning Rate optimised by a GP coming soon here .... :)










