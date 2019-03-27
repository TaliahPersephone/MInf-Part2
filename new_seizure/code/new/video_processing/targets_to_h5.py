import numpy as np
import random
import tables
import cv2 as cv
import os
import sys
import csv


base = '/home/taliah/Documents/Course/Project/new_seizure/'

targets_location = base + 'data/6464/targets/' 
dst = base + 'data/6464/h5/{}.h5'


files = []

for filename in os.listdir(targets_location):
	files += [filename]

	
	

for filename in files:
	f = tables.open_file(dst.format(filename[:12]),'w')
	targets = f.create_earray(f.root,'targets',tables.Int8Atom(),(0,),expectedrows=7476)
	print('{}'.format(filename))

	t = open(targets_location + filename)
	targets_csv = csv.reader(t)
	targets_single = []

	for row in targets_csv:
		targets_single += [row[1]]

	t.close()

	targets_single = targets_single[:7476]
	targets.append(np.array(targets_single))


	f.close()

