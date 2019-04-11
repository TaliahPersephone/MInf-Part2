'''
' Takes an input stream and queue of bounding box and will store the cropped frames in the given queue
'
' Constructed such that the bounding boxes can be provided using any interface
'
' With some of the original data, some of the frames did not have bounding box information
' Due to this, the implementation checks that the frames match up
'
' This could be easily modifies with a call to a function that extracts the bounding box
' From the frame directly
'''

import queue
import cv2 as cv


'''
'Input:
'	i - input stream
'	boxes - boxes queue, expected in the form [frame,centre_x,centre_y,width,height]
'Output:
'	o - output queue
'
'''
def get_frames(i, boxes, o):
	cap = cv.VideoReader(i)

	count = 0

	while True:
		ret, frame = cap.read()

		if not ret:
			print('End of stream has been read')
			break

		box = boxes.get()

		box_frame = box[0]

		while (box_frame < count):
			print('Skipping box')
			box = boxes.get()
	

		old_count = 0
		while (box_frame > count):
			ret, frame = cap.read()
			count += 1

		if old_count != count: print('Skipped {} frames'.format(count - old_count))
			
		if not ret:
			print('End of stream has been read')
			break

		if frame.shape[-1] == 3:
			frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)


		c = [round(box[1])),round(box[2])]
		# The max ensures that the box is square
		l = max(box[3],box[4]))
		
		l = round(l/2)

		# The various max and mins here ensure that the program does not attempt to take
		# from outside the frame
		y_0 = max(0,c[1]-l)
		y_1 = min(frame.shape[0],c[1]+l)
		x_0 = max(0,c[0]-l)
		x_1 = min(frame.shape[1],c[0]+l)


		crop = frame[y_0:y_1,x_0:x_1] 	
		crop_64 = np.array(cv.resize(crop,(64,64)))

		o.put((count,crop_64))

		count += 1
		
