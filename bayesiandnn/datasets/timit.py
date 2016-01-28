

import numpy as np 
import cPickle, gzip
import os
import theano
import math
from collections import Counter
from theano import tensor as T
from os import sys, path

sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from dataio.pfileio import PfileIO
# from .. import dataio.PickleIO as PickleIO
# 

sys.path.append('~/masters-thesis/bayesiandnn/bayesiandnn/datasets/rawdata')


def _load_raw_data(datapath):
	dirpath, filename = os.path.split(datapath)

	if dirpath == '' and not os.path.isfile(datapath):
		new_dirpath = os.path.join(
			os.path.split(__file__)[0],
			'datasets/rawdata/',
			filename
			)

		if os.path.isfile(new_dirpath) or datapath == '.pickle.gz':
			datapath = new_dirpath

	f = gzip.open(datapath, 'rb')



if __name__ == '__main__':
	file_reader = PfileIO('tr95.pfile.gz')
	file_reader.readpfileInfo()
	file_reader.readPfile()
	x, y = file_reader.make_shared()
	print x.shape, y.shape

	# print x, y

