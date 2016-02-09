import numpy as np 
import os
import cPickle, gzip
import theano 
import theano.tensor as T 
# from theano import sharedstreams
import sys
import struct
from data_utils import *


# bad hack should be fixed

class PfileIO(object):

	def __init__(self, datapath):
		self.feats = None	
		self.labels = None
		# self.partCounter = 0
		self.datapath = datapath

		# pfile information
		self.header_size = 32768
		self.feat_start_column = 2
		self.feat_dim = 1024
		self.label_start_column = 442
		self.num_labels = 1

		self.partition_value = 1024*1024*60
		self.total_frame_num = 0
		self.partition_num = 0
		self.frame_per_partition = 0
		self.end_reading = False
		self._loadData()
		self.featsGenerated = False


	def _loadData(self):
		dirpath, filename = os.path.split(self.datapath)

		if dirpath == '' and not os.path.isfile(self.datapath):
			new_dirpath = os.path.join(
			os.path.split(__file__)[0],
			'../datasets/rawdata/',
			filename
			)
		# dirpath = '/home/akash/masters-thesis/bayesiandnn/bayesiandnn/datasets/timit-fbank.pfile.g'

		if os.path.isfile(new_dirpath) or self.datapath == 'speechtrain1.pickle.gz':
			self.datapath = new_dirpath


		self.f = gzip.open(self.datapath, 'rb')
		# print self.f



	def readpfileInfo(self):
		line = self.f.readline()
		if line.startswith('-pfile_header') == False:
			print "Error:Wrong format"
			exit(1)

		# self.header_size = int(line.split(' ')[-1])

		while (not line.startswith('-end')):
			if line.startswith('-num_sentences'):
				self.num_sentences = int(line.split(' ')[-1])
			elif line.startswith('-num_frames'):
				self.total_frame_num = int(line.split(' ')[-1])
			elif line.startswith('-first_feature_column'):
				self.feat_start_column = int(line.split(' ')[-1])
			elif line.startswith('-num_features'):
				self.original_feat_dim = int(line.split(' ')[-1])
			elif line.startswith('-first_label_column'):
				self.label_start_column = int(line.split(' ')[-1])
			elif line.startswith('-num_labels'):
				self.num_labels = int(line.split(' ')[-1])
			line = self.f.readline()
			print line

		# this line is a bit fishy, will have to check this .....
		# divide the data into number of batches

		self.feat_dim = self.original_feat_dim
		self.frame_per_partition = 1024*1024*500 / (self.feat_dim *4)
		batch_residual = self.frame_per_partition % 256
		self.frame_per_partition = self.frame_per_partition - batch_residual



	def readPfile(self, left=5, right =5):

		self.dtype = np.dtype({'names':['d', 'l'],
								'formats':[('>f', self.original_feat_dim), '>i'],
								'offsets': [self.feat_start_column * 4, self.label_start_column * 4]})

		self.f.seek(self.header_size + 4 * (self.label_start_column + self.num_labels) * self.total_frame_num)
		sentence_offset = struct.unpack(">%di" % (self.num_sentences + 1), self.f.read(4 * (self.num_sentences + 1)))
		self.feats = []
		self.labels = []

		self.f.seek(self.header_size)

		#  read the file copied directly from pdnn github... 

		for i in xrange(self.num_sentences):
			num_frames = sentence_offset[i+1] - sentence_offset[i]
			if self.f is file:  # Not a compressed file
				sentence_array = np.fromfile(self.f, self.dtype, num_frames)
			else:
				nbytes = 4 * num_frames * (self.label_start_column + self.num_labels)
				d_tmp = self.f.read(nbytes)
				sentence_array = np.fromstring(d_tmp, self.dtype, num_frames)

			# print sentence_array['d']
			feats = np.asarray(sentence_array['d'], dtype=np.float32)
			labels = np.asarray(sentence_array['l'])
			# print feats.shape
			# print labels

			# this step is already being done by kaldi scripts for me, the feature is already concatenated by the left and right context frames and so we dont need it here ,
			# but it can be definitely used for a genreal use-case, uncomment in that case			
			# feats = make_context_feature(feats, left, right)

			if len(self.feats) > 0 and read_frames < self.frame_per_partition:
				num_frames = min(len(feats), self.frame_per_partition - read_frames)
				self.feats[-1][read_frames : read_frames + num_frames] = feats[:num_frames]
				self.labels[-1][read_frames : read_frames + num_frames] = labels[:num_frames]
				feats = feats[num_frames:]
				labels = labels[num_frames:]
				read_frames += num_frames
			if len(feats) > 0:
				read_frames = len(feats)
				self.feats.append(np.zeros((self.frame_per_partition, self.feat_dim), dtype = theano.config.floatX))
				self.labels.append(np.zeros(self.frame_per_partition, dtype = np.int32))
				self.feats[-1][:read_frames] = feats
				self.labels[-1][:read_frames] = labels


		# finish reading; close the file
		self.f.close()
		self.feats[-1] = self.feats[-1][:read_frames]
		self.labels[-1] = self.labels[-1][:read_frames]

		self.partition_num = len(self.feats)
		self.partition_index = 0
		self.featsGenerated = True
		print "number of partitions:"
		# print len(self.feats)



	# def load_next_partition(self, shared_xy):
	# 	feat = self.feats[self.partition_index]
	# 	label = self.labels[self.partition_index]

	# 	shared_x, shared_y = shared_xy

	# 	shared_x.set_value(feat.astype(theano.config.floatX), borrow=True)
	# 	shared_y.set_value(feat.astype(theano.config.floatX), borrow=True)

	# 	self.cur_frame_num = len(feat)


	def generate_features(self):
		if self.featsGenerated != True:
			self.readPfile()

		return self.feats[0], self.labels[0]
		return self.feats, self.labels




	def make_shared(self):
		print len(self.feats), self.feats[0].shape
		feat = self.feats[0]
		label = self.labels[0].astype(theano.config.floatX)

		if len(self.feats) > 0:
			self.shared_feats = theano.shared(feat.astype(theano.config.floatX), borrow=True, name='x')
			self.shared_labels = theano.shared(label, borrow=True, name='y')
			self.shared_labels = T.cast(self.shared_labels, 'int32')

		return self.shared_feats, self.shared_labels


		























