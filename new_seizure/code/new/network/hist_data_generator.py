import tables
from keras.utils import Sequence
import numpy as np
from itertools import product
from threading import Lock

path = '/home/taliah/Documents/Course/Project/new_seizure/data/6464/h5/{}'

class hist_data_generator(Sequence):
	def __init__(self,files, flips = ['original'],orientations = ['','_1','_n1','_2','_n2'], batch_size = 623, seed = 2573, contiguous = False, lock = None):
		flips = ['hist_{}'.format(i) for i in flips]

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

		data = f.root[orientation][:][inds]

		targets = f.root.targets[:][inds]

		f.close()

		self.mutex.release()

		return data, targets

