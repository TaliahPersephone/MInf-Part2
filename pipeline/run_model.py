from queue import Queue
import keras
from keras.models import Model,model_from_json
from hist_model import coords_end_hist_model
import os
import threading
import numpy as np

'''
' This will load a keras model and evaluate inputs. It monitors over a window of values.
' Inputs:
'	model	- model to load (model_json, weights.hdf5
'	i	- input features queue
'	coords	- coords features queue
' Outputs:
'	o 	- queue outputting detection of seizures
'''
def run_model(model,i,coords,o):
	model_json, weights_name = model

	window = 500
	start_t = 365
	fin_t = 300
	current = 0
	seizure = 1
	q = Queue()

	for x in range(window):
		q.put(0)
	

	'''
	' This should load json but I was having issues
	'''
	#model = model_from_json(model_json)
	model = coords_end_hist_model(5,[1024,1024,1024,1024,1024])


	model.load_weights(weights_name)

	while True:

		features = i.get()
		coord = coords.get()


		p = int(np.round(model.predict([[features],[coord]]))[0,0])
		
		current += p
		current -= q.get()
		q.put(p)
		
		if seizure:
			seizure = int(current > fin_t)
		else:
			seizure = int(current > start_t)

		

		o.put(seizure)

