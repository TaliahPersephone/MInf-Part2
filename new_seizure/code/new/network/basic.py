import keras
from histdata import load_data
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import logging
import sys

print("Basic network for baselines on histogram data\n")

logging.basicConfig(filename='logs/{}.log'.format(sys.argv[1]), level=logging.INFO)

seed = 287942

batch_size = 128
num_classes = 1
epochs = 5

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = load_data(0)

logging.info('Basic on fold 0 with 2x512 dense layers, 0.2 dropout, relu')

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

logging.info(score)

