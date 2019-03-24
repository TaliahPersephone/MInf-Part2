import csv
import os
import threading
import queue

path = '/home/taliah/Documents/Course/Project/new_seizure/'
temporal = path+'temporal_annotations/1439328827509_{:06}_AZ324hrsno5and8_1.csv'

q = queue.Queue()
bouts = {0:[],1:[],2:[],3:[],4:[]}


def bout_details():
	vid = q.get()

	f = open(temporal.format(vid),'r')

	r = csv.DictReader(f) 

	target_old = 0
	bout = [0,0]

	for row in r:

		frame = round(float(row['t'])) //40

		target = bool(float(row['Clonic seizures'])) or \
			bool(float(row['Absent seizures'])) or \
			bool(float(row['Generalised full motor seizures'])) or \
			bool(float(row['Rearing (seizures)'])) or \
			bool(float(row['Tonic seizures']))

		
		if target != target_old:
			bouts[vid] += [frame]
			

		target_old = target
	
	f.close()

	q.task_done()


for i in range(5):
	q.put(i)


for i in range(5):
	t = threading.Thread(target=bout_details)
	t.daemon = True
	t.start()


q.join()

n = 0
tt = 0
for i in range(5):
	l = len(bouts[i])
	n += l/2
	total = 0
	print('Vid:{}\tBouts:{}'.format(i,l))
	for j in range(int(l/2)):
		first = bouts[i][2*j]
		second = bouts[i][2*j+1]
		total += second - first
		print('{}-{}\t\t{}'.format(first,second,second-first))
	print('Total frames \t\t{}\n'.format(total))
	tt += total
		
print('Bouts:{}\t\tFrames:{}'.format(n,tt))

