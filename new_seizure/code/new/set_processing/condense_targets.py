import csv
import re
import os
import sys
import threading
import queue

path = '/home/taliah/Documents/Course/Project/new_seizure/data/{}/targets/'.format(sys.argv[1])

q = queue.Queue()

condense = int(sys.argv[2])
limit = int(float(sys.argv[3]) * condense) 
count = 0 
lst = []

out = path + 'c{}-l{}'.format(condense,limit) 

if not os.path.exists(out):
    os.makedirs(out)

def condense_targets():
	while True:
		target_file = q.get()
		print('Starting - {}'.format(target_file))

		a = open(path+target_file)
		f = open('{}/{}'.format(out,target_file),'w')

		annotations = csv.reader(a)
		targets_condense = csv.writer(f)
	
	
		count = 0
		val = 0
	
		for row in annotations:
		
			count += 1
			val += int(row[1])
	
			if count % condense == 0:
				targets_condense.writerow([int((count / condense) - 1),int(val >= limit)])	
				val = 0
	
			if count % 1000 == 0:
				print('csv - {}\trow - {}'.format(target_file,count/condense - 1))
	
		
		a.close()
		f.close()

		q.task_done()



for i in range(8):
	t = threading.Thread(target=condense_targets)
	t.daemon = True
	t.start()


for filename in os.listdir(path):
	if filename.endswith('.csv'):
		q.put('{}'.format(filename))


q.join()
