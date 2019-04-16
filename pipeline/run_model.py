import queue
import keras
from keras.models import Model,model_from_json
import os
import threading


def run_model(model,i,o):
	model_json, weights_name = model

	model = model_from_json(model_json)


	model.load_weights(weights_name)

	while True:

		features = i.get()

		p = model.predict_on_batch(features)

		o.put(p)
