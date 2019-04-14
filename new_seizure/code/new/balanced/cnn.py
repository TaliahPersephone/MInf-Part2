import keras
import os
from tp_tn import tp, tn
from cnn_data_generator import cnn_data_generator
from keras.models import Sequential
from keras.layers import Flatten,Reshape, Conv2D, BatchNormalization, Dense, Dropout, Activation, MaxPooling2D, LSTM, SimpleRNN
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from lr import step_decay
import logging
import sys
import random
from threading import Lock
import numpy as np
import argparse
from cnn_model import *
from get_folds import *

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--cont',type=str2bool,default=True)
parser.add_argument('--coords',type=str2bool,default=True)
parser.add_argument('--f',type=int,default=0)
parser.add_argument('--epoch',type=int,default=2)
args = parser.parse_args()
args = parser.parse_args()

seed = 287942

print(args.coords)


batch_size = 534
num_classes = 1
epochs = args.epoch

path = '/home/taliah/Documents/Course/Project/new_seizure/data/6464/h5'

files = []

for filename in os.listdir(path):
	if filename[5] != '4'and filename.endswith('h5'): 
		files += [filename]
		


folds = get_folds()

mutex = Lock()

i = args.f
v = folds[i]
t = []
for j in files:
	if j not in v:
		t += [j]

train_gen = cnn_data_generator(files = t,seed = seed, batch_size = batch_size,contiguous=args.cont,lock=mutex,coords=args.coords)
val_gen = cnn_data_generator(files = v,seed = seed, batch_size = batch_size,contiguous=args.cont,lock=mutex,coords=args.coords)

model = cnn_model(args.coords)

lrate = LearningRateScheduler(step_decay)

filepath = 'models/fold{}.cnn_lstm_coords_end.weights.best.hdf5'.format(i)
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint,lrate]

history = model.fit_generator(train_gen,
                    epochs=epochs,
                    verbose=1,
                    validation_data=val_gen, max_queue_size = 5, 
                    callbacks = callbacks_list)

keras.backend.clear_session()

del model
del train_gen
del val_gen


