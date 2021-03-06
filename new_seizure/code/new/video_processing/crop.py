import cv2 as cv
import threading
import numpy as np
import csv
import os
import queue


path = '/home/taliah/Documents/Course/Project/new_seizure/'
spatial = path + 'spatial_annotations/1439328827509_{:06}_AZ324hrsno5and8_1_bb.csv'
vids = path + 'video/video_chunks/1439328827509_{:06}_AZ324hrsno5and8_1-{:05}.avi' 
out_vid = path + 'video/6464_chunks/original/{:06}-{:05}.avi' 

fourcc = cv.VideoWriter_fourcc(*'FLV1')
aspect = (64,64)

q = queue.Queue()

def crop_chunk():

	vid_num = q.get()

	f = open(spatial.format(vid_num))
	boxes = csv.DictReader(f,delimiter=',')
	count = 0

	for box in boxes:	
		
		frame_n = round(float(box['Frame_number']))
		
		if (frame_n % 7500 == 0):
			cap = cv.VideoCapture(vids.format(vid_num,frame_n))
			out = cv.VideoWriter(out_vid.format(vid_num,frame_n),fourcc, 25.0, aspect)
			t_f = open(path+'video/video_chunks/targets/{:06}-{:05}.csv'.format(vid_num,frame_n))
			target = csv.DictReader(t_f)

			try:
				t_o_f.close()
			except:
				pass

			t_o_f = open(path + 'video/6464_chunks/targets/{:06}-{:05}.csv'.format(vid_num,frame_n),'w')
			target_out = csv.DictWriter(t_o_f,fieldnames=target.fieldnames)
			print('Vid - {}; Chunk - {}'.format(vid_num,frame_n))
			count = 0
	

		time = next(target)

		while(int(time['Frame_number']) != count):
			time = next(target)

		count += 1


		target_out.writerow(time)

		if (count % 1000 == 0):
			print('Vid - {}; Count - {}'.format(vid_num,count))
		
		c = [round(float(box['centre_x'])),round(float(box['centre_y']))]
		l = float(max(box['width'],box['height']))
		
		ret, frame = cap.read()
		
		l = round(l/2)

		y_0 = max(0,c[1]-l)
		y_1 = min(500,c[1]+l)
		x_0 = max(0,c[0]-l)
		x_1 = min(1200,c[0]+l)


		crop = frame[y_0:y_1,x_0:x_1] 	
		crop_64 = cv.resize(crop,aspect)

		out.write(crop_64)
		
	f.close()

	try:
		t_f.close()
		t_o_f.close()
	except:
		pass

	q.task_done()


for i in range(5):
	q.put(i)


for i in range(5):
	t = threading.Thread(target=crop_chunk)
	t.daemon = True
	t.start()


q.join()
