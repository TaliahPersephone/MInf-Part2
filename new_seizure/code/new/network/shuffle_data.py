import numpy as np
import tables
import os
import scipy.io as sio
import sys

path = '/home/taliah/Documents/Course/Project/new_seizure/data/{}/'.format(sys.argv[1])

load_path = path + '{}/data.h5'
save_path = path + '{}/shuffled_data.h5'


seed = 236987


for set_name in ['train/','val/','test/']:
	f = tables.open_file(load_path.format(set_name),'r')

	data_old = f.root.data[:]
	targets_old = f.root.targets[:]

	f.close()

	r = np.random.RandomState(seed)
	r.shuffle(data_old)

	r = np.random.RandomState(seed)
	r.shuffle(targets_old)


	f = tables.open_file(save_path.format(),'w')
	
	
	data = f.create_earray(f.root,'data',tables.Float32Atom(),(0,data_old.shape[1]),expectedrows=data_old.shape[0])
	targets = f.create_earray(f.root,'targets',tables.Int64Atom(),(0,),expectedrows=targets_old.size)
	
	data.append(data_old)
	targets.append(targets_old)

	f.close()
