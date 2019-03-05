import csv
import re
import os
import sys
import threading
import queue
import scipy.io as sio

path = '/home/taliah/Documents/Course/Project/new_seizure/data/{}/'.format(sys.argv[1])

q = queue.Queue()

condense = int(sys.argv[2])
limit = int(float(sys.argv[3]) * condense) 
count = 0 
lst = []

mats = path + '/mats/original/'
targets = path + 'targets/c{}-l{}/'.format(condense,limit) 



def check_targets():
	while True:
		root, name = q.get()

		mat = sio.loadmat('{}/{}'.format(root,name))

		mat_shape = mat['dataFull'].shape

		f = open('{}{}.csv'.format(targets,name[:-4]))
		r = csv.reader(f)

		count = 0

		for row in r:
			count += 1

		oki = mat_shape[0]== count

		print('File {}\n{}\n'.format(os.path.join(root,name),oki))

		if not oki:
			lst.append('{}\tMat-{}\tTargets-{}'.format(name,mat_shape[0],count))
		
		f.close()

		q.task_done()
		

for i in range(8):
	t = threading.Thread(target=check_targets)
	t.daemon = True
	t.start()

for root, dirs, files in os.walk(mats):
	for name in files:
		if name.endswith('.mat'):
			q.put((root,name))

q.join()

with open('not_oki.txt','w') as f:
	for item in lst:
		f.write('{}\n'.format(item))
