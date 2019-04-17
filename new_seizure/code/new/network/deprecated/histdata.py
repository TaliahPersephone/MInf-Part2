import tables
import os 
import numpy as np



def load_data(fold=-1):
	"""
	Loads HOG/HOF/MGH data

	Arguments
		fold: enter which fold to load, -1 will load full data as train
	
	Returns
		tuple of numpy arrays: '(x_train, y_train), (x_test, y_test)'
	"""
	path = '/home/taliah/Documents/Course/Project/new_seizure/data/6464/{}/{}.h5'



	if (fold < 0):
		f = tables.open_file(path.format('train','contiguous'),'r')
		x = f.root.data[:]
		y = f.root.targets[:]
		f.close()
		return (x,y)
	else:
		f = tables.open_file(path.format('train','fold'+str(fold)),'r')
		x_train = f.root.data[:]
		y_train = f.root.targets[:]
		f.close()

		f = tables.open_file(path.format('val','fold'+str(fold)),'r')
		x_test = f.root.data[:]
		y_test = f.root.targets[:]
		f.close()

		return (x_train, y_train), (x_test,y_test)
	





