# -*- coding: utf-8 -*-
import numpy as np
import cv2 as cv
import os
import csv
import threading
import queue

DISPLAY = 0

# Open file
path = '/home/taliah/Documents/Course/Project/new_seizure/video/'

fileName_list = ['1439328827509_000000_AZ324hrsno5and8_1.flv',
	'1439328827509_000001_AZ324hrsno5and8_1.flv',
	'1439328827509_000002_AZ324hrsno5and8_1.flv',
	'1439328827509_000003_AZ324hrsno5and8_1.flv',
	'1439328827509_000004_AZ324hrsno5and8_1.flv']

q = queue.Queue()

def cut_vid():

	fileName = q.get()

	print("Open file: ", fileName)
	cap = cv.VideoCapture(path+fileName)
	#cap = cv.VideoCapture(0)
	
	# Define the codec and create VideoWriter object
	fourcc = cv.VideoWriter_fourcc(*'FLV1')
	
	fn = 0
	#out = cv.VideoWriter(path+'video-chunks/'+fileName[:-4]+"-"+str(fn)+'.avi',fourcc, 25.0, (1200,500))
	
	out = cv.VideoWriter(path+'video_chunks/'+fileName[:-4]+"-"+"%05d" % fn+'.avi',fourcc, 25.0, (1200,500))
	
	chunk_size = 5*60*25
	
	while(cap.isOpened()):
		ret, frame = cap.read()
		if fn%250 == 0:
			print(fn)
		if fn%chunk_size == 0:
			print('next-video/t',fn)
			out.release()
			#out = cv.VideoWriter(path+'video-chunks/'+fileName[:-4]+"-"+str(fn)+'.flv',fourcc, 25.0, (1200,500))
			out = cv.VideoWriter(path+'video_chunks/'+fileName[:-4]+"-"+"%05d" % fn+'.avi',fourcc, 25.0, (1200,500))
		
		if ret==True:
			#frame = cv.flip(frame,0)
			# write the flipped frame
			out.write(frame)
			if DISPLAY == 1:
				cv.imshow('frame',frame)
				if cv.waitKey(1) & 0xFF == ord('q'):
					break
		else:
			break
		fn += 1
	# Release everything if job is finished
	cap.release()
	out.release()

	q.task_done()



for i in fileName_list:
	q.put(i)


for i in range(5):
	t = threading.Thread(target=cut_vid)
	t.daemon = True
	t.start()


q.join()
