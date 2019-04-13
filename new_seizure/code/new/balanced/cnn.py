import keras
import os
from tp_tn import tp, tn
from cnn_data_generator import cnn_data_generator
from keras.models import Sequential
from keras.layers import Flatten,Reshape, Conv2D, BatchNormalization, Dense, Dropout, Activation, MaxPooling2D, LSTM, SimpleRNN
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from lr import step_decay
import logging
import sys
import random
from threading import Lock
import numpy as np
import argparse
from get_folds import *

parser = argparse.ArgumentParser()
parser.add_argument('--layers', type=int, default = 2)
parser.add_argument('--size',type=int, default = 512)
parser.add_argument('--last',default='Dense')
parser.add_argument('--cont',type=bool,default=False)
parser.add_argument('--f',type=int,default=0)
parser.add_argument('--t',type=int,default=4)
parser.add_argument('--epoch',type=int,default=4)
args = parser.parse_args()
args = parser.parse_args()

seed = 287942


batch_size = 534
num_classes = 1
epochs = args.epoch

path = '/home/taliah/Documents/Course/Project/new_seizure/data/6464/h5'

files = []

for filename in os.listdir(path):
	if filename[5] != '4': 
		files += [filename]
		


folds = get_folds()

mutex = Lock()

for i in range(args.f,args.t):

	v = folds[i]
	t = []
	for j in files:
		if j not in v:
			t += [j]

	train_gen = cnn_data_generator(files = t,seed = seed, batch_size = batch_size,contiguous=args.cont,lock=mutex)
	val_gen = cnn_data_generator(files = v,seed = seed, batch_size = batch_size,contiguous=args.cont,lock=mutex)



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

	if args.last == 'Dense':
		model.add(Dense(1,activation='sigmoid'))

	elif args.last == 'LSTM':
		model.add(Reshape((1,-1)))
		model.add(LSTM(num_classes, activation='sigmoid'))

	elif args.last == 'Simple':
		model.add(Reshape((1,-1)))
		model.add(SimpleRNN(num_classes, activation='sigmoid'))
	
	model.summary()
	
	adam = Adam(0.00001)

	lrate = LearningRateScheduler(step_decay)
	model.compile(loss='binary_crossentropy',
	              optimizer=adam,
	              metrics=['accuracy',tp,tn])

	filepath = 'fold{}.cnn.weights.best.hdf5'.format(i)
	checkpoint = ModelCheckpoint(filepath, monitor='val_tp', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint,lrate]
	
	history = model.fit_generator(train_gen,
	                    epochs=epochs,
	                    verbose=1,
	                    validation_data=val_gen, max_queue_size = 5, 
                            callbacks = callbacks_list)
	
	
	del model
	del train_gen
	del val_gen

	keras.backend.clear_session()

