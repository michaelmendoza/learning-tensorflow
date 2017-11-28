
'''
Formats CiFAR-10 data
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import data_downloader

def unpickle( file ):
	""" unpickle data - supported for python 2 and python 3 """
	try: 
		import cPickle
		fo = open(file, 'rb')
		dict = cPickle.load(fo)
	except ImportError:
		import _pickle as cPickle
		fo = open(file, 'rb')
		dict = cPickle.load(fo, encoding='latin-1')

	fo.close()
	return dict

def one_hot_vec(label):
	""" returns a single one hot vector for given label index """
	vec = np.zeros(10)
	vec[label] = 1
	return vec

def whiten_data(features):
	""" whiten our data - zero mean and unit standard deviation """
	features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
	return features

def load_data(ratio = 0.8):
	""" loads training/test data using a sl"""

	# Load training data batches
	for _ in range(5):
		d = unpickle( 'cifar-10-batches-py/data_batch_' + str(_+1) )

		# Grab data and labels 
		if(_ == 0):
			_x = d['data']
			_y = d['labels']
		else:
			_x = np.vstack((_x, d['data']))
			_y = np.concatenate((_y, d['labels']), axis=0)

	# Load test data batch
	d = unpickle('cifar-10-batches-py/test_batch')
	_x = np.vstack((_x, d['data']))
	_y = np.concatenate((_y, d['labels']), axis=0)

	# Process data i.e. whitten input data, and set output data as a one hot vector
	_x = whiten_data(_x)
	_y = np.array(list(map(one_hot_vec, _y)))

	# Split data into test/training sets
	index = int(ratio * len(_x)) # Split index
	x_train = _x[0:index, :]
	y_train = _y[0:index]
	x_test = _x[index:,:]
	y_test = _y[index:]

	# Print out data sizes for train/test batches
	print("Data Split: ", ratio)
	print("Train => x:", x_train.shape, " y:", len(y_train))
	print("Test  => x:", x_test.shape, " y:", len(y_test))

	return [x_train, y_train, x_test, y_test]

class DataLoader():
	""" data loader for cifar-10 dataset """
	def __init__(self):
		data_downloader.download_and_extract()
		self.x_train, self.y_train, self.x_test, self.y_test = load_data()

	def next_batch(self, batch_size):
		length = self.x_train.shape[0]
		indices = np.random.randint(0, length, batch_size) # Grab batch_size values randomly
		return [self.x_train[indices], self.y_train[indices]]






