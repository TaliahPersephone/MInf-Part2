import shutil
import random 
import os
import sys



path = '/home/taliah/Documents/Course/Project/new_seizure/video/{}_chunks/'.format(sys.argv[1])
dst = '/home/taliah/Documents/Course/Project/new_seizure/data/{}/'.format(sys.argv[1])

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
	if current < (0.8 * count):
		shutil.copyfile(f,'{}{}/{:04}.avi'.format(dst,'train',current))
	elif current < (0.9 * count):
		shutil.copyfile(f,'{}{}/{:04}.avi'.format(dst,'val',current % int(0.8*count)))
	else:
		shutil.copyfile(f,'{}{}/{:04}.avi'.format(dst,'test',current % int(0.9*count)))
	current += 1
	
