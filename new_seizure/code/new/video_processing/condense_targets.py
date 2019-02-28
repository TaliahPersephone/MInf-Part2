import csv
import os
import sys
import threading
import queue

path = '/home/taliah/Documents/Course/Project/new_seizure/data/6464/'

q = queue.Queue()

condense = int(sys.argv[1])
limit = int(float(sys.argv[2]) * condense) 
count = 0 
lst = []

def condense_targets():
	target_file = q.get()

	annotations = csv.DictReader(open(target_file))
	targets_condense = csv.DictWriter(open('{}-c{}-l{}.csv'.format(target_file[:-4],condense,limit),'w'),fieldnames)
	targets_condense.writeheader()

	fieldnames=['Number','Target']

	count = 0
	val = 0

	for row in annotations:
	
		count += 1
		val += row['Target']

		if count % condense == 0:
			targets.writerow({'Number': ((count % 5) - 1),'Target':int(val >= limit)})	
			val = 0

	q.task_done()



for i in range(8):
	t = threading.Thread(target=condense_targets)
	t.daemon = True
	t.start()

for root, dirs, files in os.walk(path):
	for name in files:
		if name.endswith('.csv'):
			p.put('{}/{}'.format(root,name))

q.join()
