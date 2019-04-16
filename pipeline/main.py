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

def hist_pipe(coords_q, hog_q,hof_q,mbhr_q,mbhc_q,o):
	while True:
		n1,hog = hog_q.get()
		n2,hof = hof_q.get()
		n3,mbhc = mbhc_q.get()
		n4,mbhr = mbhr_q.get()
		n5,coords_q = coords_q.get()

		hist = np.array([hog,hof,mbhr,mbhc]).flatten()
		
		for q in o:
			q.put(hist)

def combined_pipe(coords_q,cnn_q,hist_q,o):
	while True:
		n1,cnn = cnn_q.get().flatten()

		n2,hist = hist_q.get()

		n3,coords = coords_q.get()

		combined = np.array([cnn,hist]).flatten()

		for q in o:
			q.put(combined)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--models', default='cnn', nargs='+', help='Filepaths to json files containing models to use')  
	parser.add_argument('--weights',nargs='+',help='Provide saved model weights, same order as models')
	parser.add_argument('--labels',nargs='+',help='Names of the models, otherwise they will be numbered')
	parser.add_argument('--data',nargs='+',help='Which data each model accepts [hist|cnn|combined]')
	parser.add_argument('--input', help='Input handle, default will use /dev0')
	parser.add_argument('--boxes',help='Input to be passed to the get_boxes function')
	parser.add_argument('--matlab',help='Numpy file of Matlab computed histograms if desired')
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
	coords_to[]

	# Create necessary queues if at least one of the models uses the hist data
	if 'hist' in args.data or 'combined' in args.data:
		
		hist_to = []
		if args.matlab is None:
			hist = [Queue(50),Queue(50)]
			frames_to += [hist[0]]
		else:
			hof = [Queue(50),Queue(50)]
			hog = [Queue(50),Queue(50)]
			mbhr = [Queue(50),Queue(50)]
			mbhc = [Queue(50),Queue(50)]
			frames_to += [hof[0],hog[0],mbhr[0],mbhc[0]]

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
		coords_to += [Queue(50)]
		models_in += [Queue(50)]
		if args.data[i] == 'cnn':
			frames_to += [models_in[i]]
		if args.data[i] == 'hist':
			hist_to += [models_in[i]]
		if args.data[i] == 'combined':
			combined_to += [models_in[i]]
		models_out += [Queue(50)]


	# Start the thread reading boxes	
	t = threading.Thread(target=get_boxes,args=(args.boxes,boxes,coords_to))
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
	if 'hist' in args.data or 'combined' in args.data:
		if args.matlab is None:
			t = threading.Thread(target=feed_hist,args=(hist[0],args.matlab,hist_to))
			t.daemon = True
			t.start()
		else:	
			t = threading.Thread(target=HOG,args=(hog[0],hog[1]))
			t.daemon = True
			t.start()
			t = threading.Thread(target=HOF,args=(hof[0],hof[1]))
			t.daemon = True
			t.start()
			t = threading.Thread(target=MBHr,args=(mbhr[0],mbhr[1]))
			t.daemon = True
			t.start()
			t = threading.Thread(target=MBHc,args=(mbhc[0],mbhc[1]))
			t.daemon = True
			t.start()
			t = threading.Thread(target=hist_pipe,args=(hog[1],hof[1],mbhr[1],mbhc[1],hist_to))
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

		
		

