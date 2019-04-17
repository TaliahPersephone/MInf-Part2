import keras
from keras.models import Sequential
from keras.layers import SimpleRNN,Dense, BatchNormalization, LSTM, Reshape, Conv2D, MaxPooling2D, Flatten, Activation
from keras.optimizers import RMSprop
from hist_data_generator import hist_data_generator
from cnn_data_generator import cnn_data_generator
from keras.callbacks import ModelCheckpoint
import sys
import os
import argparse
from threading import Lock
import numpy as np
import random
import keras.backend as K


def tp(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos) / (K.sum(y_pos) + 0.000000001)
    tn = K.sum(y_neg * y_pred_neg) / (K.sum(y_neg) + 0.000000001)
    return tp

def tn(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos) / (K.sum(y_pos) + 0.000000001)
    tn = K.sum(y_neg * y_pred_neg) / (K.sum(y_neg) + 0.000000001)
    return tn

def get_model(args,i):
	hidden_units = args.size
	layers = args.layers

	if args.model == 'cnn':
		
		model = Sequential()
		model.add(Reshape((64,64,1),input_shape=(64,64)))
		model.add(BatchNormalization())
		model.add(Conv2D(20, kernel_size=5,padding='same',activation=None))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
		model.add(Conv2D(40, kernel_size=3,padding='same',activation=None))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
		model.add(Flatten())
		model.add(Dense(256, activation=None))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(Dense(1, activation='sigmoid'))
		
		model.summary()
		
		model.compile(loss='binary_crossentropy',
		              optimizer='adam',
		              metrics=[tp,tn])
	
		filepath = 'models/fold{}.cnn.weights.best.hdf5'.format(i)

		model.load_weights(filepath)

	elif args.model == 'hist':

		model = Sequential()
		model.add(Dense(hidden_units, activation='relu', input_shape=(6272,)))
		model.add(BatchNormalization())
	
		for l in range(1,layers):
	
			model.add(Dense(hidden_units, activation='relu'))
			model.add(BatchNormalization())
		
		
		if args.last == 'Dense':
			model.add(Dense(1,activation='sigmoid'))
	
		elif args.last == 'LSTM':
			model.add(Reshape((1,-1)))
			model.add(LSTM(1, activation='sigmoid'))
	
		elif args.last == 'Simple':
			model.add(Reshape((1,-1)))
			model.add(SimpleRNN(1, activation='sigmoid'))
		
		model.summary()
		
		model.compile(loss='binary_crossentropy',
		              optimizer='adam',
		              metrics=[tp,tn])
		
		filepath = 'models/fold{}.last_lstm.{}l.{}h.weights.best.hdf5'.format(i,args.layers,args.size,args.last)
		model.load_weights(filepath)


	return model


parser = argparse.ArgumentParser()
parser.add_argument('--model',default='hist')
parser.add_argument('--layers', type=int, default = 2)
parser.add_argument('--size',type=int, default = 512)
parser.add_argument('--last',default='Dense')
parser.add_argument('--cont',type=bool)
args = parser.parse_args()

mutex = Lock()

seed = 287942
batch_size = 623


path = '/home/taliah/Documents/Course/Project/new_seizure/data/6464/h5'

files = []

for filename in os.listdir(path):
	if filename[5] != '4': 
		files += [filename]
		random.seed(seed)
		random.shuffle(files)

for i in range(4):

	v = np.arange(int(len(files)/10)*i,int(len(files)/10)*(i+1)) 

	if args.model == 'hist':
		val_gen = hist_data_generator(files = [files[i] for i in v],seed = seed, batch_size = batch_size,contiguous=args.cont,lock=mutex)
	elif args.model == 'cnn':
		val_gen = cnn_data_generator(files = [files[i] for i in v],seed = seed, batch_size = batch_size,contiguous=args.cont,lock=mutex)

	model = get_model(args,i)
	
	score = model.evaluate_generator(val_gen)
	
	print(score)
