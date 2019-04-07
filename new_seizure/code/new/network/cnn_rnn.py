import keras
import os
from cnn_data_generator import cnn_data_generator
from keras.models import Sequential
from keras.layers import LSTM,Flatten,Reshape, Conv2D, BatchNormalization, Dense, Dropout, Activation, MaxPooling2D
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
import logging
import sys
import random
from threading import Lock

print("Basic cnn\n")

logging.basicConfig(filename='logs/{}.log'.format(sys.argv[1]), level=logging.INFO)

seed = 287942

batch_size = 534
num_classes = 1
epochs = 3

path = '/home/taliah/Documents/Course/Project/new_seizure/data/6464/h5'

files = []

for filename in os.listdir(path):
	if filename[5] != '4': 
		files += [filename]
		random.seed(seed)
		random.shuffle(files)
		


logging.info('Basic with 20cnn k=5, 40cnn k=3, 256 lstm tanh, batch normalisation, relu')
logging.info('Batch_size {}, epochs {}'.format(batch_size,epochs))

mutex = Lock()

train_gen = cnn_data_generator(files = files[int(0.1*len(files)):],seed = seed, contiguous=True,batch_size = batch_size,lock=mutex)
val_gen = cnn_data_generator(files = files[:int(0.1*len(files))],seed = seed,contiguous=True, batch_size = batch_size,lock=mutex)



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
model.add(Reshape((1,-1)))
model.add(LSTM(256, activation=None))
model.add(BatchNormalization())
model.add(Activation('tanh'))
model.add(Dense(num_classes, activation='sigmoid'))

logging.info(model.summary())

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

i = 0

filepath = 'fold{}.cnn_rnn.weights.best.hdf5'.format(i)
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit_generator(train_gen,
                    epochs=epochs,
                    verbose=1,
                    validation_data=val_gen, max_queue_size = 5, 
                    callbacks = callbacks_list,
                    use_multiprocessing = True,
                    workers = 4)

score = model.evaluate_generator(val_gen)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

logging.info(score)

