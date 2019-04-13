import keras
import argparse
from tp_tn import tp, tn
from keras.models import Sequential
from keras.layers import SimpleRNN,Dense, BatchNormalization, LSTM, Reshape
from keras.optimizers import RMSprop, Adam
from hist_data_generator import hist_data_generator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import sys
import os
from threading import Lock
import numpy as np
import random
from get_folds import *

from lr import step_decay

parser = argparse.ArgumentParser()
parser.add_argument('--layers', type=int, default = 2)
parser.add_argument('--size',type=int, default = 512)
parser.add_argument('--last',default='Dense')
parser.add_argument('--cont',type=bool,default=False)
parser.add_argument('--f',type=int,default=0)
parser.add_argument('--t',type=int,default=4)
parser.add_argument('--epoch',type=int,default=15)
args = parser.parse_args()

mutex = Lock()

seed = 287942

batch_size = 534
hidden_units = args.size
layers = args.layers

num_classes = 1
epochs = args.epoch

path = '/home/taliah/Documents/Course/Project/new_seizure/data/6464/h5'

files = []

for filename in os.listdir(path):
	if filename[5] != '4': 
		files += [filename]

print("batch = {}".format(batch_size))

folds = get_folds()


for i in range(args.f,args.t):

	v = folds[i]
	t = []
	for j in files:
		if j not in v:
			t += [j]

	train_gen = hist_data_generator(files = t,seed = seed, batch_size = batch_size,contiguous=args.cont,lock=mutex)
	val_gen = hist_data_generator(files = v,seed = seed, batch_size = batch_size,contiguous=args.cont,lock=mutex)

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
	
	filepath = 'models/fold{}.{}l.{}h.{}-last.weights.best.hdf5'.format(i,layers,hidden_units,args.last)
	checkpoint = ModelCheckpoint(filepath, monitor='val_tp', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint,lrate]
	
	history = model.fit_generator(train_gen,
	                    epochs=epochs,
	                    verbose=1,
	                    validation_data=val_gen, max_queue_size = 4, 
                            callbacks = callbacks_list)
	

	del model
	del train_gen
	del val_gen

	keras.backend.clear_session()

