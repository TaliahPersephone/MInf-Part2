import os
import csv
import numpy as np
import time
import threading
import queue

path = '/home/taliah/Documents/Course/Project/new_seizure/video/6464_chunks/targets/'

q = queue.Queue()

def or_targets():
	while True:
		f_name = q.get()


		f = open(path+f_name)

		r = csv.reader(f)

		exists = False

		for row in r:
			if row[1] == '1':
				exists = True
				break

		if not exists:
			print('{}'.format(f_name))

		q.task_done()
		

		
for i in range(8):
	t = threading.Thread(target=or_targets)
	t.daemon = True
	t.start()



for filename in os.listdir(path):
	if filename.endswith('.csv'):
		q.put(filename)

q.join()
