import shutil
import random 
import os
import sys


base = '/home/taliah/Documents/Course/Project/new_seizure/'

path = base + 'video/{}_chunks/'.format(sys.argv[1])
dst = base + 'data/{}/'.format(sys.argv[1])
target = base + 'video/video_chunks/targets/{}.csv' 

seed = int(sys.argv[2])

count = 0 
lst = []

for root, dirs, files in os.walk(path):
	for name in files:
		if name.endswith('.avi'):
			count += 1
			lst.append('{}/{}'.format(root,name))

random.seed(seed)

random.shuffle(lst)

current = 0


for f in lst:
	vid = f.split('/')[-1][:-4]
	if current < (0.8 * count):
		place = 'train'
	elif current < (0.9 * count):
		place = 'val'
	else:
		place = 'test'

	shutil.copyfile(f,'{}{}/{:04}.avi'.format(dst,place,current))
	shutil.copyfile(target.format(vid),'{}{}/targets/{:04}.csv'.format(dst,place,current))
	current += 1
	
