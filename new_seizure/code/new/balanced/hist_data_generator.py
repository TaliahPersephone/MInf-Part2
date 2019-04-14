import tables
from keras.utils import Sequence
import numpy as np
from itertools import product
from threading import Lock

path = '/home/taliah/Documents/Course/Project/new_seizure/data/6464/h5/{}'

class hist_data_generator(Sequence):
	def __init__(self,files, flips = ['original'],orientations = ['','_1','_n1','_2','_n2'], batch_size = 534, seed = 2573, contiguous = False, lock = None, coords = None):
		flips = ['balance_hist_{}'.format(i) for i in flips]

		trans = [''.join(i) for i in product(flips,orientations)]

		self.files = product(files,trans)

		self.coords = coords

		src = []

		r = np.random.RandomState(seed)

		for name in self.files:
			f = tables.open_file(path.format(name[0]))
			n = f.root.balance_targets[:].size
			f.close()
		
			ind = np.arange(n)
			if not contiguous:
				r.shuffle(ind)

			inds = np.array([ind[i*batch_size:(i+1)*batch_size] for i in range(n // batch_size)])
			#print(n)
			#print(inds)

			src.extend(list(product([name],inds)))


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

		targets = f.root.balance_targets[:][inds]

		coords = f.root.balance_coords[:][inds]

		f.close()

		self.mutex.release()


		if self.coords is None:
			return data, targets
		elif self.coords == 'start':
			return np.append(data,coords,1), targets
		else:
			return [data,coords],targets
		
