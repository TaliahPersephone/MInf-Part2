import csv
import numpy as np
import time
import threading
import queue

path = '/home/taliah/Documents/Course/Project/new_seizure/'
file_in = path + 'temporal_annotations/1439328827509_{:06}_AZ324hrsno5and8_1.csv'
out =	path + 'data/targets/{:06}_{:05}.csv' 
full = path + 'video/targets/{:06}.csv'

q = queue.Queue()

split_targets():


make_targets():
	vid_num = q.get() 	
	annotations = csv.DictReader(file_in.format(vid_num),delimiter=',')







for i in range(5):
	t = threading.Thread(target=crop_chunk)
	t.daemon = True
	t.start()

for i in range(5):
	q.put(i)
