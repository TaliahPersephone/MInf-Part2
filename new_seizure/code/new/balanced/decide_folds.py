import numpy as np
import tables
import os
import sys
import random

path = '/home/taliah/Documents/Course/Project/new_seizure/data/6464/h5'

seed = 8767

folds = []
n = np.array([])
files = []

for filename in os.listdir(path):
	if filename[5] != '4' and filename.endswith('h5'): 
		f = tables.open_file('{}/{}'.format(path,filename))
		n = np.append(n, f.root.balance_targets[:].size)
		files += [filename]
		f.close()

N = np.sum(n)

inds = np.arange(len(files))

r = np.random.RandomState(seed)

r.shuffle(inds)

count = 0
fold = []
print(N/10)

for i in range(len(files)):
	count += n[inds[i]]
	fold += [files[inds[i]]]

	if count >= (N/10- 1000):
		folds.append(fold)

		print('{}\t{}'.format(fold,count))
		fold = []
		count = 0



