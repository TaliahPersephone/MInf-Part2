import os
import threading
import sys
import cv2 as cv
import numpy as np
import queue

aspect_0 = sys.argv[1]
aspect_1 = sys.argv[2]
path = '/home/taliah/Documents/Course/Project/new_seizure/data/{}{}/'.format(aspect_0,aspect_1)

q = queue.Queue()

def HOG(im):
	im = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
	im = np.float32(im) / 255.0

	dim = im.shape
	dim = (int(dim[0]/8),int(dim[1]/8))

	gx = cv.Sobel(im, cv.CV_32F, 1, 0, ksize=1)
	gy = cv.Sobel(im, cv.CV_32F, 0, 1, ksize=1)
	
	mag, angle = cv.cartToPolar(gx,gy, angleInDegrees=True)

	left_bin = ((angle - (angle % 20)) / 20).astype(int)
	angle_right = (angle % 20) / 20
	angle_left = 1 - angle_right

	mag_left = (angle_left * mag).reshape((8,8,8,8)).transpose((0,2,1,3))
	mag_right = (angle_right * mag).reshape((8,8,8,8)).transpose((0,2,1,3))

	left_bin = left_bin.reshape((8,8,8,8)).transpose((0,2,1,3))	
	
	hists = np.zeros((dim[0],dim[1],18))


	for i in range(18):
		mask = left_bin == i
		hists[:,:,i] += (mask * mag_left).sum((2,3))
		mask = left_bin == (i+1)%18
		hists[:,:,i] += (mask * mag_right).sum((2,3))

	h1 = hists[:7,:7]
	h2 = hists[:7,1:8]
	h3 = hists[1:8,:7]
	h4 = hists[1:8,1:8]

	hists = np.append(h1,h2,axis=2)	
	hists = np.append(hists,h3,axis=2)	
	hists = np.append(hists,h4,axis=2)	

	norm = np.linalg.norm(hists,axis=2)

	b = norm.reshape((7,7,1))
	hists_norm = np.divide(hists,b,out=np.zeros_like(hists),where = b!=0) 

	return hists_norm.flatten()

	

def HOG_chunk():
	while True:
		set_name,vid = q.get()
		print('Starting {}/{}'.format(set_name,vid))
	
		cap = cv.VideoCapture('{}{}/{:04}.avi'.format(path,set_name,vid))
	
		feature_size = (int(aspect_0)/8 - 1) * (int(aspect_1)/8 - 1) * 72
		hogs = np.empty((0,int(feature_size)))
		
		count = 0
		while cap.isOpened():
			if (count % 100 == 0):
				print ('Vid:{}-{} Count:{}'.format(set_name,vid,count))
	
			ret, frame = cap.read()
	
			if (not ret):
				break
			
			h = HOG(frame)
	
			hogs = np.append(hogs,h)

			count += 1
	
		np.save('{}{}/{:04}.npy'.format(path,set_name,vid), hogs)	

		q.task_done()

for i in range(8):
	t = threading.Thread(target=HOG_chunk)
	t.daemon = True
	t.start()


for set_name in ['train','val','test']:
	count = 0
	for filename in os.listdir('{}{}/'.format(path,set_name)):
		if filename.endswith('.avi'):
				count += 1
				q.put((set_name,count))


q.join()
