import csv
import numpy as np
import time
import threading
import queue

path = '/home/taliah/Documents/Course/Project/new_seizure/'
file_in = path + 'temporal_annotations/1439328827509_{:06}_AZ324hrsno5and8_1.csv'
out = path + 'video/video_chunks/targets/{:06}-{:05}.csv'

q = queue.Queue()

def make_targets():
	vid_num = q.get()


	f = open(file_in.format(vid_num))

	annotations = csv.DictReader(f)
	fieldnames=['Frame_number','Target']

	for row in annotations:
		frame = round(float(row['t'])) //40
		if (frame % 7500 == 0):
			print('Starting {:06}-{:05}'.format(vid_num,frame))
			try:
				t.close()
			except:
				pass
			t = open(out.format(vid_num,frame),'w')
			targets = csv.DictWriter(t,fieldnames)
			targets.writeheader()
		target = bool(float(row['Clonic seizures'])) or \
			bool(float(row['Absent seizures'])) or \
			bool(float(row['Generalised full motor seizures'])) or \
			bool(float(row['Rearing (seizures)'])) or \
			bool(float(row['Tonic seizures']))

		targets.writerow({'Frame_number': (frame % 7500),'Target':int(target)})	

	t.close()
	f.close()
	q.task_done()


for i in range(5):
	q.put(i)


for i in range(5):
	t = threading.Thread(target=make_targets)
	t.daemon = True
	t.start()


q.join()
