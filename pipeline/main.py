'''
' This is main entry point for the pipeline to detect features (seizures or grooming) in rodents
' 
' Inputs:
' 	Model 	- Which model to use
'	Weights - Location of weights for selected model
'	Input 	- Location of video to detect features in
'
'
' The program will attempt to detect the features from the provided input, this can be a live camera feed
'
'''


import sys
import cv2 as cv
import numpy as np
import threading
from queue import Queue
import argparse


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', default='cnn', help='Which model to use, [cnn|hist|combined]')  
	parser.add_argument('--weights',help='Provide saved model weights')
	parser.add_argument('--input', type=int, help='Input handle, default will use /dev0')
	args = parser.parse_args()

	if args.model not in ['cnn','hist','combined']:
		raise AssertionError('Model must be [cnn|hist|combined]')	

	if args.weights is None:
		raise AssertionError('Provide model weights')

	frames = Queue(50)
	boxes = Queue(50)

	

	
