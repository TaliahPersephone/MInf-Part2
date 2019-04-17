import tables
from keras.utils import Sequence
import numpy as np
from itertools import product
from threading import Lock

path = '/home/taliah/Documents/Course/Project/new_seizure/data/6464/h5/{}'

class cnn_data_generator(Sequence):
	def __init__(self,files, flips = ['original','h_flip'],orientations = ['','_1','_n1','_2','_n2'], batch_size = 623, seed = 2573, contiguous = False, lock = None):

		trans = [''.join(i) for i in product(flips,orientations)]

		self.files = product(files,trans)

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

		#print(inds)

		self.mutex.acquire()

		f = tables.open_file(path.format(filename),'r')

		if (orientation == 'original'): data = f.root.original[:][inds]
		elif (orientation == 'original_1'): data = f.root.original_1[:][inds]
		elif (orientation == 'original_n1'): data = f.root.original_n1[:][inds]
		elif (orientation == 'original_2'): data = f.root.original_2[:][inds]
		elif (orientation == 'original_n2'): data = f.root.original_n2[:][inds]
		elif (orientation == 'h_flip'): data = f.root.h_flip[:][inds]
		elif (orientation == 'h_flip_1'): data = f.root.h_flip_1[:][inds]
		elif (orientation == 'h_flip_n1'): data = f.root.h_flip_n1[:][inds]
		elif (orientation == 'h_flip_2'): data = f.root.h_flip_2[:][inds]
		elif (orientation == 'h_flip_n2'): data = f.root.h_flip_n2[:][inds]

		targets = f.root.targets[:][inds]

		f.close()

		self.mutex.release()

		return data, targets
