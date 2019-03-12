import numpy as np
import tables
import os
import scipy.io as sio
import sys

path = '/home/taliah/Documents/Course/Project/new_seizure/data/{}/'.format(sys.argv[1])

load_path = path + '{}/data.h5'
save_path = path + '{}/balanced_shuffled_data.h5'


seed = 883952


for set_name in ['train/','val/','test/']:

	print('Starting')

	f = tables.open_file(load_path.format(set_name),'r')

	features = f.root.data.shape[1]
	targets_old = f.root.targets[:]


	is_one = np.nonzero(targets_old)
	count_one = is_one[0].size

	targets_inv = np.nonzero(targets_old == 0)[0]

	r = np.random.RandomState(seed)

	ind = r.choice(targets_inv,count_one)

	print('Loading data')




	s = tables.open_file(save_path.format(set_name),'w')
	
	
	data = s.create_earray(s.root,'data',tables.Float32Atom(),(0,features),expectedrows=2*count_one)
	targets = s.create_earray(s.root,'targets',tables.Int64Atom(),(0,),expectedrows=2*count_one)

	print('Writing initial')
	
	data_balance = f.root.data[:][is_one]
	data.append(data_balance)

	print('two')

	data_balance = f.root.data[:][ind[:int(ind.size/2)]]
	data.append(data_balance)

	print('three')

	data_balance = f.root.data[:][ind[int(ind.size/2):]]
	data.append(data_balance)

	f.close()
	
	targets_balance = np.zeros(2*count_one,dtype=np.int64)
	targets_balance[:count_one] = 1

	targets.append(targets_balance)
	
	s.close()

	print('Reloading')

	f = tables.open_file(save_path.format(set_name),'r')

	data_balance = f.root.data[:]
	targets_balance = f.root.targets[:]

	f.close()

	print('Opening New')

	f = tables.open_file(save_path.format(set_name),'w')
	
	
	data = f.create_earray(f.root,'data',tables.Float32Atom(),(0,features),expectedrows=2*count_one)
	targets = f.create_earray(f.root,'targets',tables.Int64Atom(),(0,),expectedrows=2*count_one)

	print('Shuffling data')

	r = np.random.RandomState(seed)
	r.shuffle(data_balance)

	print('Shuffling targets')

	r = np.random.RandomState(seed)
	r.shuffle(targets_balance)

	print('Writing')

	data.append(data_balance)
	targets.append(targets_balance)

	f.close()
