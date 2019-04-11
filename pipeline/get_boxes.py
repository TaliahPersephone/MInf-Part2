'''
' This particular implementation will read bounding box information from a csv file and output it
' Straight to a queue
'''

import queue
import csv


'''
' Input:
'	i - bounding boxes csv file
' Output:
'	o - queue to ouput box information to
'''
def get_boxes(i, o):
	boxes = csv.DictReader(i)

	for row in boxes:
		box = [float(idx) for idx in list(row.values())]
		
		o.put(box)
	
