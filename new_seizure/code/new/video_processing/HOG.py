import os
import sys
import cv2 as cv
import numpy as np
import queue

path = '/home/taliah/Documents/Course/Project/new_seizure/data/6464/'#.format(sys.argv[1])

q = queue.Queue()

def HOG(im):
	im = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
	im = np.float32(im) / 255.0

	dim = im.shape

	gx = cv.Sobel(im, cv.CV_32F, 1, 0, ksize=1)
	gy = cv.Sobel(im, cv.CV_32F, 0, 1, ksize=1)
	
	mag, angle = cv.cartToPolar(gx,gy, angleInDegrees=True)

	left_bin = ((angle - (angle % 20)) / 20).astype(int)
	angle_right = (angle % 20) / 20
	angle_left = 1 - angle_right

	mag_left = angle_left * mag
	mag_right = angle_right * mag
	
	hists = np.zeros((int(dim[0]/8),int(dim[1]/8),18))

	for i in range(dim[0]):
		for j in range(dim[1]):
			ind_i = np.floor(i/8).astype(int)
			ind_j = np.floor(j/8).astype(int)
			hists[ind_i,ind_j,left_bin] += mag_left[i,j]
			hists[ind_i,ind_j,(left_bin+1)%18] += mag_right[i,j]
				
			
	return hists

	

def HOG_chunk():
	set_name,vid = q.get()

	cap = cv.VideoCapture('{}{}/{:04}.avi'.format(path,set_name,vid))

	count = 0
	while cap.isOpened():
		if (count % 1000 == 0):
			print ('Vid:{}-{} Count:{}'.format(set_name,vid,count))

		ret, frame = cap.read()

		if (not ret):
			break
		
		

	

'''
for set_name in ['train','val','test']:
	for filename in os.listdir('{}{}/'.format(path,set_name)):
		if filename.endswith('.avi'):
				count += 1

	hog_writer = csv.writer(open())
'''
