import numpy as np
import tables
import os
from itertools import product
import sys

path = '/home/taliah/Documents/Course/Project/new_seizure/data/6464/{}/'

batch_size = 534
number = int(7476 / batch_size)

flip = ['original','h_flip']	
orientations = ['','_1','_n1','_2','_n2']

trans = [''.join(i) for i in product(flip,orientations)]

m = ['balance_','balance_hist_']

total = [''.join(i) for i in product(m,trans)] + ['balance_coords']


for filename in os.listdir(path.format('h5')):
	if filename.endswith('.h5') and not filename.startswith('balance'):
		print(filename)

		f = tables.open_file('{}/{}'.format(path.format('h5'),filename))

		d = tables.open_file('{}/{}'.format(path.format('balance'),filename),'w')

		data = f.root['balance_targets'][:]
		
		a = d.create_earray(d.root,'balance_targets',tables.IntAtom(),(0,),expectedrows=data.shape[0])
		a.append(data)


		for name in total:
			data = f.root[name][:]
			if name.startswith('balance_hist'):
				features = (6272,)
			elif name == 'balance_coords':
				features = (4,)
			else:
				features = (64,64)
			a = d.create_earray(d.root,name,tables.Float32Atom(),(0,) +features,expectedrows=data.shape[0])

			a.append(data)


		f.close()
		d.close()
