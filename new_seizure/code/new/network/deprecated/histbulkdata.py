import tables
import os 
import numpy as np


def load_data(seed=52754):
	"""
	Loads HOG/HOF/MGH data

	Arguments
		seed: enter a seed to shuffle the data, no seed = no shuffle
		test: if you would like to use test data, False = use val data
	
	Returns
		tuple of numpy arrays: '(x_train, y_train), (x_test, y_test)'
	"""
	path = '/home/taliah/Documents/Course/Project/new_seizure/data/6464/train/bulk-{}.h5'

	count = 8000 * 5
	features = 6272 

	x = np.zeros((count * 6, features))
	y = np.zeros((count * 6,))
	r = np.random.RandomState(seed)

	for i in range(6):
		print(i)
		f = tables.open_file(path.format(i),'r')

		ind = r.permutation(range(count))
		x[i*count:(i+1)*count,:] = f.root.data[:][ind,:]
		y[i*count:(i+1)*count] = f.root.targets[:][ind]


	return (x, y)



