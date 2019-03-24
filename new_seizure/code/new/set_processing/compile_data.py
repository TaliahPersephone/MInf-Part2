import numpy as np
import tables
import os
import scipy.io as sio
import sys
import csv

path = '/home/taliah/Documents/Course/Project/new_seizure/data/{}/'.format(sys.argv[1])
#condense = int(sys.argv[2])
#limit = int(float(sys.argv[3]) * condense) 
count = [0,0,0]

load_path = path + '{}/'
save_path = path + '{}/data.h5'

s = 0

for set_name in ['train/','val/','test/']:
	for filename in os.listdir(path + set_name):
		if filename.endswith('.mat'):
			count[s] += 1
	s += 1



mat = sio.loadmat(path+'train/0000.mat')
features = mat['dataFull'].shape[1]
atom = tables.Float64Atom()



s_n = 0
for set_name in ['train','val','test']:
	try:
		os.remove(save_path.format(set_name))
	except OSError:
		pass

	print(set_name)
	
	f = tables.open_file(save_path.format(set_name),mode='w')

	data = f.create_earray(f.root,'data',tables.Float32Atom(),(0,features),expectedrows=7476*count[s_n])
	targets = f.create_earray(f.root,'targets',tables.Int64Atom(),(0,),expectedrows=7476*count[s_n])
	
	for i in range(count[s_n]):
		print(i)
		mat = sio.loadmat(load_path.format(set_name)+'{:04}.mat'.format(i))
		
		data.append(mat['dataFull'])

		
		t = open(load_path.format(set_name) +'targets/{:04}.csv'.format(i))
		targets_csv = csv.reader(t)
		targets_single = []

		for row in targets_csv:
			targets_single += [row[1]]

		t.close()

		targets.append(np.array(targets_single))
		
	f.close()
	s_n+=1

		#print('{}\t{}'.format(np.array(targets,ndmin=2).T.shape,mat['dataFull'].shape))



		
