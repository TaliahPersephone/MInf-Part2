import numpy as np
import tables
import cv2 as cv
import os
import sys
import csv

base = '/home/taliah/Documents/Course/Project/new_seizure/'

vids = base + 'video/6464_chunks/{}/'
dst = base + 'data/6464/h5/{}.h5'

files = []

for filename in os.listdir(vids.format('original')):
	files += [filename]

	
	

for filename in files:
	f = tables.open_file(dst.format(filename[:12]),'a')
	print('{}'.format(filename))

	for flip in ['original{}','h_flip{}']:
		for rot in ['','_1','_n1','_2','_n2']:
			trans = flip.format(rot)
			print(trans)
			data = f.create_earray(f.root,trans,tables.Float32Atom(),(0,64,64),expectedrows=7476)

			cap = cv.VideoCapture(vids.format(trans) + filename)

			count = 0
			while(count < 7476):
				ret, frame = cap.read()
				
				frame = np.float32(cv.cvtColor(frame,cv.COLOR_BGR2GRAY)/255)

				data.append(np.reshape(frame,(1,64,64)))

				count += 1

		
			cap.release()


	f.close()



