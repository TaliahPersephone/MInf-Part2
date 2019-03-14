import numpy as np
import scipy.io as sio
import tables
import sys
import os
import csv

seed_choose = 2574
seed_shuffle = 1859

base = '/home/taliah/Documents/Course/Project/new_seizure/'

path = base + 'data/6464/mats/original{}/{:06}-{:05}.mat'
dst = base + 'data/6464/{}/bulk-{}.h5'
target_dst = base + 'data/6464/targets/{:06}-{:05}.csv' 

f = open('selections.txt','r')
vid_num = [0,0,1,2,2,3]
features = 6272 

count = 0

r = np.random.RandomState(seed_choose)

for row in f:
	print(count)
	sel = np.fromstring(row,dtype=int,sep=',').reshape((-1,2))

	vid = vid_num[count]	

	bulk = tables.open_file(dst.format('train',count),'w')
	data = bulk.create_earray(bulk.root,'data',tables.Float32Atom(),(0,features),expectedrows=45000)
	targets = bulk.create_earray(bulk.root,'targets',tables.Int64Atom(),(0,),expectedrows=45000)

	chunks = np.array([],dtype=int)

	to_take = {}

	for pair in sel:
		chunk = int(pair[0] - (pair[0] % 7500))
		chunk2 = int(pair[1] - (pair[1] % 7500))
		
		if chunk not in chunks:
			chunks = np.append(chunks,chunk)
		
		if chunk2 not in chunks:
			chunks = np.append(chunks,chunk2)

		if chunk not in to_take.keys():
			to_take[chunk] = []
		if chunk2 not in to_take.keys():
			to_take[chunk2] = []

		if chunk == chunk2:
			to_take[chunk] += [(pair[0],pair[1])]
		else:
			to_take[chunk] += [(pair[0],chunk+7476)]
			to_take[chunk2] += [(chunk2,pair[1])]
	
	#print(chunks)
	#print(to_take)


	zero_from_each = round(10000/chunks.shape[0])

	for chunk in chunks:

		print(chunk)

		ones = np.arange(to_take[chunk][0][0]-chunk,to_take[chunk][0][1]-chunk)

		free = r.choice(np.setdiff1d(np.arange(7476),ones),zero_from_each)

		mat = sio.loadmat(path.format('',vid,chunk))['dataFull']

		new = mat[free]

		data.append(new)
		targets.append(np.zeros(zero_from_each,dtype=int))

		new = mat[ones]
			
		data.append(new)
		
		for i in ['_-1','_1','_-2','_2']:
			print(i)
			mat = sio.loadmat(path.format(i,vid,chunk))['dataFull']

			new = mat[ones]
			
			data.append(new)

		target_file = open(target_dst.format(vid,chunk))

		target = csv.reader(target_file)

		new_targets = []

		print('targets')
		for row in target:
			if int(row[0]) in ones:
				new_targets += [int(row[1])]

		new_targets = np.array(new_targets)
		
		targets.append(new_targets)
		targets.append(new_targets)
		targets.append(new_targets)
		targets.append(new_targets)
		targets.append(new_targets)
		

	bulk.close()

	count += 1
