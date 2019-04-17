import keras
from tp_tn import *
import numpy as np
from keras.models import Model,Sequential
from keras.optimizers import RMSprop, Adam
from keras.layers import Input, concatenate, LSTM, Flatten,Reshape, Conv2D, BatchNormalization, Dense, Dropout, Activation, MaxPooling2D

def get_model(batch_size=524):

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
	x = Dense(256, activation='relu')(x)

	hist_input = Input(shape=(6272,), name='hist_input')

	y = BatchNormalization()(hist_input)
	y = Dense(1024,activation='relu')(y)
	y = Dense(1024,activation='relu')(y)

	coords = Input(shape=(4,),name='coords')

	x = concatenate([x, y, coords])

	x = BatchNormalization()(x)

	x = Dense(256)(x)

	x = Reshape((1,-1))(x)

	out = LSTM(1,activation='sigmoid')(x)

	combined = Model(inputs = [cnn_input, hist_input, coords], outputs = out)

	combined.summary()
	adam = Adam(0.01)

	
	combined.compile(loss='binary_crossentropy',
	              optimizer=adam,
	              metrics=['accuracy',tp,tn])


	return combined

