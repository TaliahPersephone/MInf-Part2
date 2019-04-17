import tables
from keras.utils import Sequence
import numpy as np
from itertools import product
from threading import Lock

path = '/home/taliah/Documents/Course/Project/new_seizure/data/6464/h5/{}'

class combined_data_generator(Sequence):
	def __init__(self,files, flips = ['original','h_flip'],orientations = ['','_1','_n1','_2','_n2'], batch_size = 623, seed = 2573, contiguous = True, lock = None):

		trans = [''.join(i) for i in product(flips,orientations)]

		self.files = product(files,trans)
		self.batch_size = batch_size

		src = []

		r = np.random.RandomState(seed)

		for f in self.files:
			ind = np.arange(7476)
			if not contiguous:
				r.shuffle(ind)

			inds = np.array([ind[i*batch_size:(i+1)*batch_size] for i in range(7476 // batch_size)])

			src.extend(list(product([f],inds)))


		r.shuffle(src)
		self.src = src

		if lock == None:
			raise ValueError
		self.mutex = lock


	def __len__(self):
		return len(self.src)

	def __getitem__(self,idx):
		((filename,orientation),inds) = self.src[idx]

		#print('{},{}'.format(filename,orientation))

		self.mutex.acquire()

		f = tables.open_file(path.format(filename),'r')

		#data = np.zeros((self.batch_size,64*64+6272))

		if (orientation == 'original'): 
			#data[:,:64*64] = f.root.original[:][inds].reshape((-1,64*64))
			#data[:,64*64:] = f.root.hist_original[:][inds]
			cnn_data = f.root.original[:][inds].reshape((-1,64*64))
			hist_data = f.root.hist_original[:][inds]
		elif (orientation == 'original_1'): 
			#data[:,:64*64] = f.root.original_1[:][inds].reshape((-1,64*64))
			#data[:,64*64:] = f.root.hist_original_1[:][inds]
			cnn_data = f.root.original[:][inds].reshape((-1,64*64))
			hist_data = f.root.hist_original[:][inds]
		elif (orientation == 'original_n1'): 
			#data[:,:64*64] = f.root.original_n1[:][inds].reshape((-1,64*64))
			#data[:,64*64:] = f.root.hist_original_n1[:][inds]
			cnn_data = f.root.original[:][inds].reshape((-1,64*64))
			hist_data = f.root.hist_original[:][inds]
		elif (orientation == 'original_2'): 
			#data[:,:64*64] = f.root.original_2[:][inds].reshape((-1,64*64))
			#data[:,64*64:] = f.root.hist_original_2[:][inds]
			cnn_data = f.root.original[:][inds].reshape((-1,64*64))
			hist_data = f.root.hist_original[:][inds]
		elif (orientation == 'original_n2'): 
			#data[:,:64*64] = f.root.original_n2[:][inds].reshape((-1,64*64))
			#data[:,64*64:] = f.root.hist_original_n2[:][inds]
			cnn_data = f.root.original[:][inds].reshape((-1,64*64))
			hist_data = f.root.hist_original[:][inds]
		elif (orientation == 'h_flip'): 
			#data[:,:64*64] = f.root.h_flip[:][inds].reshape((-1,64*64))
			#data[:,64*64:] = f.root.hist_h_flip[:][inds]
			cnn_data = f.root.original[:][inds].reshape((-1,64*64))
			hist_data = f.root.hist_original[:][inds]
		elif (orientation == 'h_flip_1'): 
			#data[:,:64*64] = f.root.h_flip_1[:][inds].reshape((-1,64*64))
			#data[:,64*64:] = f.root.hist_h_flip_1[:][inds]
			cnn_data = f.root.original[:][inds].reshape((-1,64*64))
			hist_data = f.root.hist_original[:][inds]
		elif (orientation == 'h_flip_n1'): 
			#data[:,:64*64] = f.root.h_flip_n1[:][inds].reshape((-1,64*64))
			#data[:,64*64:] = f.root.hist_h_flip_n1[:][inds]
			cnn_data = f.root.original[:][inds].reshape((-1,64*64))
			hist_data = f.root.hist_original[:][inds]
		elif (orientation == 'h_flip_2'): 
			#data[:,:64*64] = f.root.h_flip_2[:][inds].reshape((-1,64*64))
			#data[:,64*64:] = f.root.hist_h_flip_2[:][inds]
			cnn_data = f.root.original[:][inds].reshape((-1,64*64))
			hist_data = f.root.hist_original[:][inds]
		elif (orientation == 'h_flip_n2'): 
			#data[:,:64*64] = f.root.h_flip_n2[:][inds].reshape((-1,64*64))
			#data[:,64*64:] = f.root.hist_h_flip_n2[:][inds]
			cnn_data = f.root.original[:][inds].reshape((-1,64*64))
			hist_data = f.root.hist_original[:][inds]
		else:
			raise Exception

		targets = f.root.targets[:][inds]

		f.close()

		self.mutex.release()

		return [cnn_data, hist_data], targets

