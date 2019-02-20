import os
import sys
import csv
import cv2 as cv
import numpy as np


path = '/home/taliah/Documents/Course/Project/new_seizure/data/{}/'.format(sys.argv[1])


for set_name in ['train','val','test']:
	for filename in os.listdir('{}{}/'.format(path,set_name)):
		if filename.endswith('.avi'):
				count += 1

	hog_writer = csv.writer()
