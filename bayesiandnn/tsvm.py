
import numpy as np 
import os
import svmlight
from datasets import mnist



# class for implementing transductive SVMs
class TSVM():

	def __init__(self, C= 10, gamma= 0.01):
		self.C = C
		self.gamma = gamma
		self.models = []
		self.trained = False


	def train_binary(self, x, y):
		train_data_svml = svmlfeaturisexy(x, y)
		model = svmlight.learn(train_data_svml, type='classification', verbosity=0, kernel='rbf', C=self.C, rbf_gamma=self.gamma)
		svmlight.write_model(model, 'tsvm_mnist.dat')


# we use one vs all strategy here ... 
	def train_multi_onevsall(self, x, y, unlab_x, strategy=1):

		num_classes = int(np.max(y) + 1)
		# num_classes = 10
		print x.shape, y.shape
		x, y = x[1:25000:5], y[1:25000:5]
		unlab_x = unlab_x[1:25000:5]

		x_feat = self.svmlfeaturise(x)
		unlab_x_feat = self.svmlfeaturise(unlab_x)
		print len(x_feat)
		print len(unlab_x_feat)

		for i in xrange(num_classes):
			y_feat = (y==i)*2 - 1
			feats = []
			lab_feats = []
			unlab_feats = []
			for j in xrange(len(x_feat)):
				lab_feats.append((y_feat[j], x_feat[j]))

			if unlab_x != None:
				for j in xrange(len(unlab_x_feat)):
					unlab_feats.append((0, unlab_x_feat[j]))

			feats = lab_feats + unlab_feats
			print "======SVM Model Training started======="
			model = svmlight.learn(feats, type='classification', verbosity=0, kernel='rbf', C=self.C, rbf_gamma=self.gamma)
			print i
			print "======SVM Model Training terminated======"
			self.models.append(model)
		self.trained = True

		# pass

	# predict the class of a single data point.
	def predict(self, x_test):
		if self.trained != True:
			raise Exception("first train a model")

		x = self.svmlfeaturise(x_test)
		y_score = []
		for j in xrange(len(self.models)):
			m = np.array(svmlight.classify(self.models[j], x))
			y_score.append(m)

		y_predicted = np.argmax(y_score, axis=0)
		return y_predicted


	# predict the accuracy on a test set ...
	def predictAccuracy(self, x_test, y_test):
		y_preds = []
		for i in xrange(x_test.shape[0]):
			_y = self.predict(x_test[i:i+1,:])
			y_preds.append(_y[0])

		
		y_test = list(y_test)
		accuracy =  sum([1 for i in range(10000) if y_preds[i] == y_test[i]]) / float(len(y_test))
		# print accuracy

		return accuracy


	def predictFull(self, x_test, y_test):
		if self.trained != True:
			raise Exception("first train a model")

		x = self.svmlfeaturise(x_test)
		y_preds_mat = []
		feats = []
		for i in xrange(len(x)):
			feats.append((0, x[i]))

		for i in range(len(self.models)):
			y_preds_mat.append(np.array(svmlight.classify(self.models[i], feats)))

		y_preds = np.argmax(np.vstack(tuple(y_preds_mat)), axis=0)
		print y_preds
		accuracy =  sum([1 for i in range(len(y_test)) if y_preds[i] == y_test[i]]) / float(len(y_test))
		return accuracy




	@staticmethod
	def svmlfeaturise(x):
		td = []
		for i in xrange(x.shape[0]):
			ft = []
			for j in xrange(x.shape[1]):
				ft_temp = (j + 1, x[i,j])
				ft.append(ft_temp)
			td.append(ft)
		return td


	@staticmethod
	def svmlfeaturisexy(x, y):
		if x.shape[0] == len(y):
			raise Exception("x and y are not the same in dimension.")

		td = []
		for i in xrange(len(s.shape[0])):
			ft = []
			for j in xrange(x.shape[1]):
				ft_temp = (j + 1, x[i,j])
				ft.append(ft_temp)
			feat = (y[i], ft)
			td.append(feat)

		return td




def main():
	train_set, valid_set, test_set = mnist.load_mnist_ssl('mnist.pkl.gz')
	x_tr_lab, y_tr_lab, x_tr_unlab, y_tr_unlab = train_set
	x_va, y_va = valid_set
	x_te, y_te = test_set
	tsvm = TSVM()
	# x_tr_lab = tsvm.svmlfeaturise(x_tr_lab)
	# x_tr_unlab = tsvm.svmlfeaturise(x_tr_unlab)
	# x_va, x_te = tsvm.svmlfeaturise(x_va), tsvm.svmlfeaturise(x_te)

	tsvm.train_multi_onevsall(x_tr_lab, y_tr_lab, x_tr_unlab)
	# accuracy = tsvm.predictAccuracy(x_te, y_te)
	accuracy = tsvm.predictFull(x_te, y_te)
	print accuracy

	# print tr_x.shape[0]



if __name__ == '__main__':
	main()
