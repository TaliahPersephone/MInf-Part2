import numpy as np
import tables
import os
from itertools import product
import sys

path = '/home/taliah/Documents/Course/Project/new_seizure/data/6464/h5'

batch_size = 534
number = int(7476 / batch_size)



for filename in os.listdir(path):
	if filename.endswith('.h5'):
		print(filename)
		f = tables.open_file('{}/{}'.format(path,filename),'a')

		coords = np.load('{}/new_features_{}.npy'.format(path,filename[:-3]))

		a = f.create_earray(f.root,'coords',tables.Float32Atom(),(0,4),expectedrows=7476)
	
		a.append(coords)	

		t = f.root.targets[:]
	
		pos = np.sum(t)
	
		batches = [any(t[i*batch_size:(i+1)*batch_size]) for i in range(number)]
	
		n = np.sum(batches) * batch_size
	
		j = 0
		while (n < pos * 2 - 300):
			batches[j] = True
			n = np.sum(batches) * batch_size
			j += 1
	

		a = f.create_earray(f.root,'balance_coords',tables.Float32Atom(),(0,4),expectedrows=n)
	
		
		for i in range(number):
			if batches[i]:
				a.append(coords[i*batch_size:(i+1)*batch_size])
	
				
	
		f.close()		
	
	
