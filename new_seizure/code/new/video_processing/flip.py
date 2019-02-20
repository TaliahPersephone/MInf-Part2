import threading
import os
import cv2 as cv
import numpy as np
import queue

path = '/home/taliah/Documents/Course/Project/new_seizure/video/6464_chunks/'

aspect = (64,64)
fourcc = cv.VideoWriter_fourcc(*'FLV1')
q = queue.Queue()

def flip_chunk():
	while True:
		title = q.get()

		count = 0
		
		cap = cv.VideoCapture(path + 'original/' + title)
		out = cv.VideoWriter(path + 'h_flip/' + title,fourcc, 25.0, aspect)

		while(cap.isOpened()):	
			if (count % 1000 == 0):
				print ('{} : {}'.format(title,count))
			ret, frame = cap.read()

			if (not ret):
				break

			img = cv.flip(frame,1)

			out.write(img)
			
			count += 1

		out.release()
		cap.release()

		q.task_done()

for i in range(8):
	t = threading.Thread(target=flip_chunk)
	t.daemon = True
	t.start()


for filename in os.listdir(path + 'original/'):
	if filename.endswith('.avi'):
		q.put(filename)

q.join()
