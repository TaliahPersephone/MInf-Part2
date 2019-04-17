import numpy as np
import tables
import cv2 as cv
import scipy.io as sio
import os
import sys
import csv

base = '/home/taliah/Documents/Course/Project/new_seizure/'

mats = base + 'data/6464/mats/{}/'
dst = base + 'data/6464/h5/{}.h5'


for filename in os.listdir(mats.format('original')):
	f = tables.open_file(dst.format(filename[:12]),'a')
	print('{}'.format(filename))

	flip = 'h_flip{}'
	for rot in ['','_1','_n1','_2','_n2']:
		trans = flip.format(rot)
		print(trans)
		try:
			data = f.create_earray(f.root,'hist_' + trans,tables.Float32Atom(),(0,6272),expectedrows=7476)
		except:
			data = f.root.hist_original

		mat = sio.loadmat(mats.format(trans)+filename)
	
		data.append(mat['dataFull'])

	


	f.close()




