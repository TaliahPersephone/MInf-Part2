import keras
import argparse
from keras.models import Sequential, Model
from keras.layers import SimpleRNN,Dense, BatchNormalization, LSTM, Reshape, Input, concatenate
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import sys
import os
from threading import Lock
import numpy as np
import random



def hist_model(layers = 2, size = [512,512], last = 'LSTM'):
	model = Sequential()
	model.add(Dense(size[0], activation='relu',input_shape=(6272,)))
	model.add(BatchNormalization())

	for l in range(1,layers-1):

		model.add(Dense(size[l], activation='relu'))
		model.add(BatchNormalization())
	
	
	if last == 'Dense':
		model.add(Dense(1,activation='sigmoid'))

	elif last == 'LSTM':
		model.add(Reshape((1,-1)))
		model.add(LSTM(1, activation='sigmoid'))

	elif last == 'Simple':
		model.add(Reshape((1,-1)))
		model.add(SimpleRNN(1, activation='sigmoid'))
	
	adam = Adam(0.01)

	
	model.compile(loss='binary_crossentropy',
	              optimizer=adam,
	              metrics=['accuracy',tp,tn])

	return model

def coords_start_hist_model(layers = 2, size = [512,512]):
	hist_input = Input(shape=(6276,))

	y = Dense(size[0],activation='relu')(hist_input)
	

	for i in range(1,layers):
		y = Dense(size[i],activation='relu')(y)
		y = BatchNormalization()(y)


	y = Reshape((1,-1))(y)
	out = LSTM(1,activation='sigmoid')(y)

	model = Model(inputs = hist_input, outputs = out)

	return model

def coords_end_hist_model(layers = 2, size = [512,512]):
	hist_input = Input(shape=(6272,))

	y = Dense(size[0],activation='relu')(hist_input)
	

	for i in range(1,layers):
		y = Dense(size[i],activation='relu')(y)
		y = BatchNormalization()(y)


	coord_input = Input(shape=(4,))

	x = concatenate([coord_input,y])

	x = Dense(256, activation='relu')(x)

	x = Reshape((1,-1))(x)

	out = LSTM(1,activation='sigmoid')(x)

	model = Model(inputs = [hist_input,coord_input], outputs = out)

	return model


