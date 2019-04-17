import numpy as np
import scipy.io as sio
import tables
import sys
import os
import csv

base = '/home/taliah/Documents/Course/Project/new_seizure/'

path = base + 'data/6464/mats/original{}/'
dst = base + 'data/6464/train/contiguous.h5'
target_dst = base + 'data/6464/targets/{}.csv' 

features = 6272 

f = tables.open_file(dst,'w')
data = f.create_earray(f.root,'data',tables.Float32Atom(),(0,features),expectedrows=24*5*7476)
targets = f.create_earray(f.root,'targets',tables.Int64Atom(),(0,),expectedrows=24*5*7476)

for filename in os.listdir(path.format('')):
	if filename[5] != '4':
		print(filename)
		t = open(target_dst.format(filename[:12]))
		targets_csv = csv.reader(t)
		targets_single = []

		for row in targets_csv:
			targets_single += [row[1]]

		t.close()

		targets_single = targets_single[:7476]

		targets.append(np.array(targets_single))
		targets.append(np.array(targets_single))
		targets.append(np.array(targets_single))
		targets.append(np.array(targets_single))
		targets.append(np.array(targets_single))

		for d in ['','_-1','_1','_-2','_2']:
			print(d)
			mat = sio.loadmat(path.format(d)+filename)['dataFull']
			data.append(mat[:7476])
		

f.close()
			

	

