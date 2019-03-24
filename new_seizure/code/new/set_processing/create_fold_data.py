import numpy as np
import random
from random import shuffle
import scipy.io as sio
import tables
import sys
import os
import csv
import threading
import queue

seed = 28385

q = queue.Queue()
files = []


base = '/home/taliah/Documents/Course/Project/new_seizure/'

path = base + 'data/6464/mats/original{}/'
dst = base + 'data/6464/{}/fold{}.h5'
target_dst = base + 'data/6464/targets/{}.csv' 

features = 6272 


def create_fold_data():
	while True:
		fold = q.get()
	
		print(fold)
	
		files_d = {}
	
		files_d['val'] = files[fold*3:(fold+1)*3]
	
		files_d['train'] = files[:fold*3]
		files_d['train'] += files[(fold+1)*3:]
	
		for set_name in ['train','val']:
	
			print('{}\t{}'.format(fold,set_name))
	
			n = len(files_d[set_name])
	
			f = tables.open_file(dst.format(set_name,fold),'w')
			data = f.create_earray(f.root,'data',tables.Float32Atom(),(0,features),expectedrows=n*5*7476)
			targets = f.create_earray(f.root,'targets',tables.Int64Atom(),(0,),expectedrows=n*5*7476)
	
			
			for filename in files_d[set_name]:
	
				print('{}\t{}'.format(fold,filename))
				
				t = open(target_dst.format(filename[:12]))
				targets_csv = csv.reader(t)
				targets_single = []
	
				for row in targets_csv:
					targets_single += [row[1]]
	
				t.close()
	
				targets_single = targets_single[:7476]
	
				for i in range(5):
					targets.append(np.array(targets_single))
	
				for d in ['','_-1','_1','_-2','_2']:
					print ('{}\t{}'.format(fold,d))
					mat = sio.loadmat(path.format(d)+filename)['dataFull']
					data.append(mat[:7476])
	
			f.close()
	
		q.task_done()


for filename in os.listdir(path.format('')):
	if filename[5] != '4':
		
		files += [filename]

		random.seed(seed)
		shuffle(files)	


for i in range(7):
	q.put(i)		

for i in range(1):
	t = threading.Thread(target=create_fold_data)
	t.daemon = True
	t.start()

	
q.join()
