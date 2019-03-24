import tables
import os 
import numpy as np



def load_data(seed=None,test=False,balanced=True,shuffled=False):
	"""
	Loads HOG/HOF/MGH data

	Arguments
		seed: enter a seed to shuffle the data, no seed = no shuffle
		test: if you would like to use test data, False = use val data
	
	Returns
		tuple of numpy arrays: '(x_train, y_train), (x_test, y_test)'
	"""
	path = '/home/taliah/Documents/Course/Project/new_seizure/data/6464/{}/'

	if balanced:
		path += 'balanced_shuffled_data.h5'
	elif shuffled:
		path += 'shuffled_data.h5'
	else:
		path += 'data.h5'

	f = tables.open_file(path.format('train'),'r')

	x_train = f.root.data[:]
	y_train = f.root.targets[:]

	f.close()

	if test:
		name = 'test'
	else:
		name = 'val'

	f = tables.open_file(path.format(name),'r')

	x_test = f.root.data[:]
	y_test = f.root.targets[:]

	f.close()


	if seed is not None:
		r = np.random.RandomState(seed)
		r.shuffle(x_train)
		r.shuffle(x_test)	
		r = np.random.RandomState(seed)
		r.shuffle(y_train)
		r.shuffle(y_test)	


	return (x_train, y_train), (x_test, y_test)



