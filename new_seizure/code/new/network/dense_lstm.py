import keras
import argparse
from tp_tn import tp, tn
from keras.models import Sequential
from keras.layers import SimpleRNN,Dense, BatchNormalization, LSTM, Reshape
from keras.optimizers import RMSprop
from hist_data_generator import hist_data_generator
from keras.callbacks import ModelCheckpoint
import sys
import os
from threading import Lock
import numpy as np
import random

parser = argparse.ArgumentParser()
parser.add_argument('--layers', type=int, default = 2)
parser.add_argument('--size',type=int, default = 512)
parser.add_argument('--last',default='Dense')
parser.add_argument('--cont',type=bool)
args = parser.parse_args()

mutex = Lock()

seed = 287942

batch_size = 623
hidden_units = args.size
layers = args.layers

num_classes = 1
epochs = 10

path = '/home/taliah/Documents/Course/Project/new_seizure/data/6464/h5'

files = []

for filename in os.listdir(path):
	if filename[5] != '4': 
		files += [filename]
		random.seed(seed)
		random.shuffle(files)

print("Basic network for baselines on histogram data\nbatch = {}".format(batch_size))


for i in range(4):

	v = np.arange(int(len(files)/10)*i,int(len(files)/10)*(i+1)) 
	t = np.setdiff1d(np.arange(len(files)),v)

	train_gen = hist_data_generator(files = [files[i] for i in t],seed = seed, batch_size = batch_size,contiguous=args.cont,lock=mutex)
	val_gen = hist_data_generator(files = [files[i] for i in v],seed = seed, batch_size = batch_size,contiguous=args.cont,lock=mutex)

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
	
	model.compile(loss='binary_crossentropy',
	              optimizer='adam',
	              metrics=['accuracy',tp,tn])
	
	filepath = 'models/fold{}.{}l.{}h.{}-last.weights.best.hdf5'.format(i,layers,hidden_units,args.last)
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]
	
	history = model.fit_generator(train_gen,
	                    epochs=epochs,
	                    verbose=1,
	                    validation_data=val_gen, max_queue_size = 5, 
#	                    use_multiprocessing = True,
                            callbacks = callbacks_list)
	
	score = model.evaluate_generator(val_gen)
	
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])


