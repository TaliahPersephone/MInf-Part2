'''
' This particular implementation will read bounding box information from a csv file and output it
' Straight to a queue
'''

import queue
import numpy as np
import csv


'''
' Input:
'	i - bounding boxes csv file
' Output:
'	o - queue to ouput box information to
'''
def get_boxes(i, o, coords):
	boxes = csv.reader(open(i))
	old_x = None
	old_y = None
	prev_x = None
	prev_y = None

	prev = 0

	for row in boxes:
		box = [float(val) for val in row]

		o.put(box)

		if (box[0] > prev + 1):
			old_x = None
			old_y = None
			prev_x = None
			prev_y = None

		prev = box[0]
		

		curr_x = box[1]/1200
		curr_y = box[2]/500


		if prev_x is None:
			change_x = 0.0
			change2_x = 0.0
			change_y = 0.0
			change2_y = 0.0
			
		else:
			change_x = np.abs(curr_x - prev_x)
			change_y = np.abs(curr_y - prev_y)

			if old_x is None:
				change2_x = 0.0
				change2_y = 0.0
			else:
				change2_x = np.abs(curr_x - old_x)
				change2_y = np.abs(curr_y - old_y)

		feature = np.array([change_x, change_y, change2_x, change2_y])
	

		for q in coords:
			q.put(feature)
