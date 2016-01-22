
import numpy as np
from datasets import mnist, tidigits, tidigits1
from tsvm import TSVM




if __name__ == "__main__":
	tsvm = TSVM()
	x_lab, y_lab, x_unlab, y_unlab = tidigits1.load_data_ssl('train_not_isolated_mfcc.pickle.gz')
	# x,y = tidigits1._load_raw_data('train_isolated_mfcc.pickle.gz')
	print x_lab.shape, y_lab.shape

	