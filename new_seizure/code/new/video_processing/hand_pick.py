import numpy
import scipy.io as sio
import tables
import sys
import os


base = '/home/taliah/Documents/Course/Project/new_seizure/'

path = base + 'data/6464/mats/'
dst = base + 'data/6464/{}/{}.h5'
target = base + 'data/6464/targets/{}.csv' 

f = open('selections.txt','r')
vid_num = [0,0,1,2,2,3]

count = 0
for row in f:
	print(count)
	sel = np.fromstring(row,dtype=int,sep=',').reshape((-1,2))

	vid = vid_num[count]	
