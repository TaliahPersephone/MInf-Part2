import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, LSTM, Reshape
from keras.optimizers import RMSprop
from hist_data_generator import hist_data_generator
from keras.callbacks import ModelCheckpoint
import sys
import os
from threading import Lock
import numpy as np
import random


mutex = Lock()

seed = 287942

features = 6272
batch_size = 623
num_classes = 1
epochs = 3

path = '/home/taliah/Documents/Course/Project/new_seizure/data/6464/h5'

files = []

for filename in os.listdir(path):
	if filename[5] != '4': 
		files += [filename]
		random.seed(seed)
		random.shuffle(files)

print("LSTM network for baselines on histogram data\nbatch = {}".format(batch_size))

scores = np.zeros(4)

for i in range(4):

	v = np.arange(int(len(files)/10)*i,int(len(files)/10)*(i+1)) 
	t = np.setdiff1d(np.arange(len(files)),v)

	train_gen = hist_data_generator(files = [files[i] for i in t],seed = seed, batch_size = batch_size,lock=mutex)
	val_gen = hist_data_generator(files = [files[i] for i in v],seed = seed, batch_size = batch_size,lock=mutex)

	model = Sequential()
	model.add(Dense(512, activation='relu',input_shape=(6272,)))
	model.add(BatchNormalization())
	model.add(Dense(512, activation='relu'))
	model.add(BatchNormalization())
	model.add(Reshape((1,-1)))
	model.add(LSTM(num_classes, activation='sigmoid',return_sequences=False))
	
	model.summary()
	
	model.compile(loss='binary_crossentropy',
	              optimizer='adam',
	              metrics=['accuracy'])
	
	filepath = 'models/fold{}.LSTM_basic.weights.best.hdf5'.format(i)
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]
	
	history = model.fit_generator(train_gen,
	                    epochs=epochs,
	                    verbose=1,
	                    validation_data=val_gen, max_queue_size = 4, 
	                    use_multiprocessing = False,
	                    #workers = 4)
                            callbacks = callbacks_list)

	
	score = model.evaluate_generator(val_gen)
	scores[i] = score[1]

	
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

print(np.mean(scores))

