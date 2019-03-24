import keras
from histdata import load_data
from keras.models import Sequential
from keras.layers import Reshape, Dense, LSTM, Activation, BatchNormalization
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.initializers import RandomNormal
import numpy as np
import logging
import sys




logging.basicConfig(filename='logs/{}.log'.format(sys.argv[1]), level=logging.INFO)

folds = 7
count = 7476 * 5 * 4
features = 6272 

batch_size = 623
epochs = 5
num_classes = 1
layer_size = 512


scores = np.zeros((folds,2))

logging.info('Folds:{}\nLayers:{}\tSize:{}\nEpochs:{}\nBatch:{}'.format(folds,2,512,20,128))

for i in range(folds):	

	(x_train, y_train), (x_test,y_test) = load_data(fold=i)

	model = Sequential()
	model.add(Reshape((1,x_train.shape[1]),input_shape=(x_train.shape[1],)))
	model.add(LSTM(layer_size, input_shape=(batch_size,1,x_train.shape[1]), return_sequences=True,kernel_initializer='random_normal'))
	#model.add(BatchNormalization())
	#model.add(Activation('relu'))
	model.add(LSTM(layer_size, return_sequences=False,kernel_initializer='random_normal', dropout=0.2))
	#model.add(BatchNormalization())
	#model.add(Activation('relu'))
	model.add(Dense(num_classes, activation='sigmoid'))
	
	model.summary()
	
	model.compile(loss='binary_crossentropy',
	              optimizer=RMSprop(),
	              metrics=['accuracy'])
	
	filepath = 'fold{}.2*512rnn.weights.best.hdf5'.format(i)
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]
	history = model.fit(x_train, y_train,
		                batch_size=batch_size,
				epochs=epochs,
	                	verbose=1,
				callbacks = callbacks_list,
	                    	validation_data=(x_test, y_test))
	score = model.evaluate(x_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
	
	logging.info('Fold:{}\tTest Loss:{}\tTest Acc:{}'.format(i,score[0],score[1]))

	scores[i,:] = score

	x_test = None
	y_test = None
	x_train = None
	y_train = None
	model = None

print('Test loss:', np.mean(scores,0)[0])
print('Test accuracy:', np.mean(scores,0)[1])

logging.info('==================\nAcc:{}\tLoss:{}\n================'.format(np.mean(scores,0)[1],np.mean(scores,0)[0]))
