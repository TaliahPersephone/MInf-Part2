import keras
from histbulkdata import load_data
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import RMSprop
import numpy as np
import logging
import sys


print("Basic network for cross val baselines on histogram data\n")

seed = 199876
r = np.random.RandomState(seed)

l = logging.getLogger('logs/{}.log'.format(sys.argv[1]))

folds = 6
count = 8000 * 5

batch_size = 128
num_classes = 1
epochs = 20
layer_size = 1024
layers = 3

# the data, split between train and test sets
(x, y) = load_data()

scores = np.zeros((folds,2))

l.info('Seed:{}'.format(seed))
l.info('Folds:{}\nLayers:{}\tSize:{}\nEpochs:{}\nBatch:{}'.format(folds,2,512,20,128))

for i in range(folds):

	print('Fold {}'.format(i))

	x_test = x[i*count:(i+1)*count,:]
	y_test = y[i*count:(i+1)*count]

	ind = r.permutation(np.setdiff1d(np.arange(count*folds),np.arange(i*count,(i+1)*count)))

	x_train = x[ind,:]
	y_train = y[ind]

	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')
	
	
	model = Sequential()
	model.add(Dense(layer_size, activation=None, input_shape=(x_train.shape[1],),use_bias = False))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	for j in range(1,layers):
		model.add(Dense(layer_size, activation=None,use_bias = False))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		
	model.add(Dense(num_classes, activation='sigmoid'))
	
	model.summary()
	
	model.compile(loss='binary_crossentropy',
	              optimizer=Adam(),
	              metrics=['accuracy'])
	
	history = model.fit(x_train, y_train,
	                    batch_size=batch_size,
	                    epochs=epochs,
	                    verbose=1,
	                    validation_data=(x_test, y_test))
	score = model.evaluate(x_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	l.info('Fold:{}\tAcc:{}\tLoss:{}\n'.format(i,score[0],score[1]))

	scores[i,:] = score

	x_test = None
	y_test = None
	x_train = None
	y_train = None
	model = None

print('Test loss:', np.mean(scores,0)[0])
print('Test accuracy:', np.mean(scores,0)[1])

l.info('==================\nAcc:{}\tLoss:{}\n================'.format(np.mean(scores,0)[1],np.mean(scores,0)[0]))
