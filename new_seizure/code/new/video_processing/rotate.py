import threading
import os
import cv2 as cv
import numpy as np
import queue

path = '/home/taliah/Documents/Course/Project/new_seizure/video/6464_chunks/'

aspect = (64,64)
fourcc = cv.VideoWriter_fourcc(*'FLV1')
q = queue.Queue()

def rotate_chunk():
	while True:
		title, flipped, angle = q.get()

		count = 0
		
		if(flipped):
			cap = cv.VideoCapture(path +'h_flip/' + title)
			out = cv.VideoWriter(path + 'h_flip_' + str(angle) + '/' + title,fourcc, 25.0, aspect)
		else:
			cap = cv.VideoCapture(path +'original/' + title)
			out = cv.VideoWriter(path + 'original_' + str(angle) + '/' + title,fourcc, 25.0, aspect)

		while(cap.isOpened()):	
			if (count % 1000 == 0):
				print ('Vid:{} Flipped:{} Angle:{} Count:{}'.format(title,flipped,angle,count))
			ret, frame = cap.read()

			if (not ret):
				break

			rows, cols, _ = frame.shape

			M = cv.getRotationMatrix2D(((cols-1.0)/2,(rows-1.0)/2),angle,1)	
			img = cv.warpAffine(frame,M,(cols,rows),borderMode=cv.BORDER_REPLICATE)

			out.write(img)
			
			count += 1

		out.release()
		cap.release()

		q.task_done()

for i in range(8):
	t = threading.Thread(target=rotate_chunk)
	t.daemon = True
	t.start()


for filename in os.listdir(path + 'original/'):
	if filename.endswith('.avi'):
		for i in [-2,-1,1,2]:
			q.put((filename,0,i))
			q.put((filename,1,i))

q.join()
