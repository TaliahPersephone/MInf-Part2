import keras
import argparse
from tp_tn import tp, tn
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D,Flatten,SimpleRNN,Dense, BatchNormalization, LSTM, Reshape, Input, concatenate
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

def cnn_model(coords = True):

	
	cnn_input = Input(shape=(64,64), name='cnn_input')

	x = Reshape((64,64,1))(cnn_input)
	x = BatchNormalization()(x)
	x = Conv2D(20, kernel_size=5,padding='same',activation='relu')(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
	x = Conv2D(40, kernel_size=3,padding='same',activation='relu')(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
	x = Flatten()(x)

	coords_input = Input(shape=(4,),name='coords_input')

	if coords:
		x = concatenate([x,coords_input])

	x = Dense(256,activation='relu')(x)

	x = BatchNormalization()(x)
	
	x = Reshape((1,-1))(x)

	out = LSTM(1,activation='sigmoid')(x)

	model = Model(inputs = [cnn_input,coords_input], outputs = out)
	
	model.summary()
	
	adam = Adam(0.01)

	model.compile(loss='binary_crossentropy',
	              optimizer=adam,
	              metrics=['accuracy',tp,tn])

	return model
