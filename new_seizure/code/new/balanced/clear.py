import numpy as np
import tables
import os
from itertools import product
import sys

path = '/home/taliah/Documents/Course/Project/new_seizure/data/6464/h5'
name = '{}/{:06}-{:05}.h5' 

files = [(0,67500)]

flip = ['original','h_flip']	
orientations = ['','_1','_n1','_2','_n2']

trans = [''.join(i) for i in product(flip,orientations)]

m = ['','hist_']

total = [''.join(i) for i in product(m,trans)]

balance = ['/balance_{}'.format(i) for i in total]

for thing in files:
	print(thing)
	f = tables.open_file(name.format(path,thing[0],thing[1]), 'a')

	f.remove_node('/balance_targets')

	for t in balance:
		f.remove_node(t)


	f.close()
	
	

