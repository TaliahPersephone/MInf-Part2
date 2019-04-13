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
from hist_preprocess import *
from get_frames import *
from get_boxes import *
#import keras
#from keras.models import model_from_json

def run_model(model,i,o):
	while True:
		w = i.get()

		o.put('nice')


def single_pipe(i, o):
	while True:
		parcel = i.get()

	
		for q in o:
			q.put(parcel)

def hist_pipe(hog_q,hof_q,mbh_q,o):
	while True:
		hog = hog_q.get()
		hof = hof_q.get()
		mbh = mbh_q.get()


		hist = np.array([hog,hof,mbh]).flatten()
		
		for q in o:
			q.put(hist)

def combined_pipe(cnn_q,hist_q,o):
	while True:
		cnn = cnn_q.get().flatten()

		hist = hist_q.get()

		combined = np.array([cnn,hist]).flatten()

		for q in o:
			q.put(combined)



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--models', default='cnn', nargs='+', help='Filepaths to json files containing models to use')  
	parser.add_argument('--weights',nargs='+',help='Provide saved model weights, same order as models')
	parser.add_argument('--labels',nargs='+',help='Names of the models, otherwise they will be numbered')
	parser.add_argument('--data',nargs='+',help='Which data each model accepts [hist|cnn|combined]')
	parser.add_argument('--input', help='Input handle, default will use /dev0')
	parser.add_argument('--boxes',help='Input to be passed to the get_boxes function')
	args = parser.parse_args()

	# Errors given arguments, reasonably self explanatory
	if args.boxes is None:
		raise AssertionError('Please supply bounding boxes')

	if args.weights is None:
		raise AssertionError('Please supply model weights')

	if len(args.weights) != len(args.models):
		raise AssertionError('Number of model weights supplied does not match the number of models')

	if len(args.data) != len(args.models):
		raise AssertionError('Number of data inputs supplied does not match the number of models')


	for d in args.data:
		if d not in ['hist','cnn','combined']:
			raise AssertionError('Data types accepted are [hist|cnn|combined]') 

	frames = Queue(50)
	boxes = Queue(50)

	frames_to = []	

	# Create necessary queues if at least one of the models uses the hist data
	if 'hist' in args.data or 'combined' in args.data:
		hist_to = []
		hof = [Queue(50),Queue(50)]
		hog = [Queue(50),Queue(50)]
		mbh = [Queue(50),Queue(50)]
		frames_to += [hof[0],hog[0],mbh[0]]

	# Create necessary queues if at least one of the models uses the combined data
	if 'combined' in args.data:
		combined_to = []
		combined = Queue(50)
		frames_to += [combined]
		hist_to += [combined]

	


	models_in = []
	models_out = []
	models = []

	
	# Create necessary queues to and from models
	for i in range(len(args.data)):
		models += [(args.models[i],args.weights[i])]

		models_in += [Queue(50)]
		if args.data[i] == 'cnn':
			frames_to += [models_in[i]]
		if args.data[i] == 'hist':
			hist_to += [models_in[i]]
		if args.data[i] == 'combined':
			combined_to += [models_in[i]]
		models_out += [Queue(50)]


	# Start the thread reading boxes	
	t = threading.Thread(target=get_boxes,args=(args.boxes,boxes))
	t.daemon = True
	t.start()
	
	# Start the thread reading frames
	t = threading.Thread(target=get_frames,args=(args.input,boxes,frames))
	t.daemon = True
	t.start()
	
	# Start the thread piping frames
	t = threading.Thread(target=single_pipe,args=(frames,frames_to))
	t.daemon = True
	t.start()
	
	# Start the hist threads
	if 'hist' in args.data:
		t = threading.Thread(target=HOG,args=(hog[0],hog[1]))
		t.daemon = True
		t.start()
		t = threading.Thread(target=HOF,args=(hof[0],hof[1]))
		t.daemon = True
		t.start()
		t = threading.Thread(target=MBH,args=(mbh[0],mbh[1]))
		t.daemon = True
		t.start()
		t = threading.Thread(target=hist_pipe,args=(hog[1],hof[1],mbh[1],hist_to))
		t.daemon = True
		t.start()
		
	# Start the combined thread
	if 'combined' in args.data:
		t = threading.Thread(target=combined_pipe,args=(combined,combined_to))
		t.daemon = True
		t.start()


	for i in range(len(args.data)):
		t = threading.Thread(target=run_model,args=(models[i],models_in[i],models_out[i]))
		t.daemon = True
		t.start()


	while True:
		for i in range(len(args.data)):
			pred = models_out[i].get()

			print('{}\t{}'.format(args.labels[i],pred))

		
		

