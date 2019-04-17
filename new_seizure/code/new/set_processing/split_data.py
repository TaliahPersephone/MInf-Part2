import shutil
import os
import sys
import numpy as np

#condense = int(sys.argv[2])
#limit = int(float(sys.argv[3]) * condense) 

base = '/home/taliah/Documents/Course/Project/new_seizure/'

path = base + 'data/{}/mats/'.format(sys.argv[1])
dst = base + 'data/{}/'.format(sys.argv[1])
#con_target = base + 'data/{}/targets/' #c{}-l{}/{}.csv'.format(sys.argv[1],condense,limit,'{}') 
target = base + 'data/{}/targets/{}.csv'.format(sys.argv[1],'{}') 


count = 0 
lst = []

seed = 3489572

count = 0 
lst = []

for root, dirs, files in os.walk(path):
	for name in files:
		if name.endswith('.mat'):
			count += 1
			lst.append('{}/{}'.format(root,name))

random.seed(seed)

random.shuffle(lst)

current = 0
name = 0;


for f in lst:
	vid = f.split('/')[-1][:-4]
	if current < (0.8 * count):
		place = 'train'
		name = current
	elif current < (0.9 * count):
		place = 'val'
		name = int(np.floor((current - 0.8 * count)))
	else:
		place = 'test'
		name = int(np.floor((current - 0.9 * count)))

	try:
		os.remove('{}{}/{:04}.mat'.format(dst,place,name))
	except OSError:
    		pass

	os.symlink(f,'{}{}/{:04}.mat'.format(dst,place,name))
	#shutil.copyfile(con_target.format(vid),'{}{}/targets/c{}-l{}/{:04}.csv'.format(dst,place,condense,limit,name))
	shutil.copyfile(target.format(vid),'{}{}/targets/{:04}.csv'.format(dst,place,name))
	current += 1
	
