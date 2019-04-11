import keras
import numpy as np
import os
from combined_data_generator import combined_data_generator
from keras.models import Model,Sequential
from keras.optimizers import RMSprop
from keras.layers import Input, concatenate, LSTM, Flatten,Reshape, Conv2D, BatchNormalization, Dense, Dropout, Activation, MaxPooling2D
from combined_model import CombinedModel
from keras.callbacks import ModelCheckpoint
import logging
import sys
import random
from threading import Lock

def get_model(batch_size=623):

	cnn_input = Input(shape=(64*64,), name='cnn_input')

	x = Reshape((64,64,1))(cnn_input)
	x = BatchNormalization()(x)
	x = Conv2D(20, kernel_size=5,padding='same',activation='relu')(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
	x = Conv2D(40, kernel_size=3,padding='same',activation='relu')(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
	x = Flatten()(x)
	x = Dense(512, activation='relu')(x)

	hist_input = Input(shape=(6272,), name='hist_input')

	y = BatchNormalization()(hist_input)
	y = Dense(1024,activation='relu')(y)
	y = Dense(512,activation='relu')(y)

	x = concatenate([x, y])

	x = BatchNormalization()(x)

	x = Dense(256)(x)

	x = Reshape((1,-1))(x)

	out = LSTM(1,activation='sigmoid')(x)

	combined = Model(inputs = [cnn_input, hist_input], outputs = out)

	return combined

print("Combine Model\n")


seed = 287942

batch_size = 534
num_classes = 1
epochs = 3

features = 6272

path = '/home/taliah/Documents/Course/Project/new_seizure/data/6464/h5'

files = []

for filename in os.listdir(path):
	if filename[5] != '4': 
		files += [filename]
		random.seed(seed)
		random.shuffle(files)
		


print('Batch_size {}, epochs {}'.format(batch_size,epochs))
scores = np.zeros(4)

mutex = Lock()

for i in range(4):

	v = np.arange(int(len(files)/10)*i,int(len(files)/10)*(i+1)) 
	t = np.setdiff1d(np.arange(len(files)),v)

	files_train = files
	train_gen = combined_data_generator(files = [files[i] for i in t],seed = seed, batch_size = batch_size,lock=mutex)
	val_gen = combined_data_generator(files = [files[i] for i in v],seed = seed, batch_size = batch_size,lock=mutex)
	
	model = get_model(batch_size)
	
	
	model.compile(loss='binary_crossentropy',
	              optimizer='adam',
	              metrics=['accuracy'])
	
	print(model.summary())
	
	filepath = 'models/fold{}.combined_dense.weights.best.hdf5'.format(i)
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]
	
	history = model.fit_generator(train_gen,
			epochs=epochs,
			verbose=1,
			validation_data=val_gen, max_queue_size = 4, 
			use_multiprocessing = True,
			callbacks = callbacks_list,
			workers = 4)
	
	score = model.evaluate_generator(val_gen)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
	

