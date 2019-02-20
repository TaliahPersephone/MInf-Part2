import csv
import numpy as np
import time
import threading

path = '/home/taliah/Documents/Course/Project/new_seizure/'
file_in = 'temporal_annotations/1439328827509_00000{}_AZ324hrsno5and8_1.csv'
out =	'data/targets/{:06}_{:05}.csv' 
full = 'video/targets/{:06}.csv'


split_targets():


make_targets():
		







for i in range(5):
	t = threading.Thread(target=crop_chunk)
	t.daemon = True
	t.start()

for i in range(5):
	q.put(i)
