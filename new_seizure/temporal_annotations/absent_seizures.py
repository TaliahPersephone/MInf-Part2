import csv
import os

w = csv.writer(open('absent.csv','w'))

og = '1439328827509_00000{}_AZ324hrsno5and8_1.csv'


for filename in  [og.format(i) for i in range(4)]:

	r = csv.DictReader(open(filename))
	for row in r:
		w.writerow(row['Absent seizures'])

