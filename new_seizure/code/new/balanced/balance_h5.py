import numpy as np
import tables
import os
from itertools import product
import sys

path = '/home/taliah/Documents/Course/Project/new_seizure/data/6464/h5'

batch_size = 534
number = int(7476 / batch_size)

flip = ['original','h_flip']	
orientations = ['','_1','_n1','_2','_n2']

trans = [''.join(i) for i in product(flip,orientations)]

m = ['','hist_']

total = [''.join(i) for i in product(m,trans)]


for filename in os.listdir(path):
	if filename.endswith('.h5'):
		print(filename)
		f = tables.open_file('{}/{}'.format(path,filename),'a')
	
		t = f.root.targets[:]
	
		pos = np.sum(t)
	
		batches = [any(t[i*batch_size:(i+1)*batch_size]) for i in range(number)]
	
		n = np.sum(batches) * batch_size
	
		j = 0
		while (n < pos * 2 - 300):
			batches[j] = True
			n = np.sum(batches) * batch_size
			j += 1
	
	
		a = f.create_earray(f.root,'balance_targets',tables.Int8Atom(),(0,),expectedrows=n)
				
		for i in range(number):
			if batches[i]:
				a.append(t[i*batch_size:(i+1)*batch_size])
	
	
		
	
		for name in total:
			data = f.root[name][:]
			if name.startswith('hist'):
				features = (6272,)
			else:
				features = (64,64)
	
			a = f.create_earray(f.root,'balance_{}'.format(name),tables.Float32Atom(),(0,) +features,expectedrows=n)
			for i in range(number):
				if batches[i]:
					a.append(data[i*batch_size:(i+1)*batch_size])
					
				
	
		f.close()		
	
