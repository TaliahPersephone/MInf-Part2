import keras
import argparse
from tp_tn import tp, tn
from keras.models import Sequential
from keras.layers import SimpleRNN,Dense, BatchNormalization, LSTM, Reshape
from keras.optimizers import RMSprop, Adam
from hist_data_generator import hist_data_generator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
import sys
import os
from threading import Lock
import numpy as np
import random
from get_folds import *
from hist_model import *
from lr import step_decay

parser = argparse.ArgumentParser()
parser.add_argument('--layers', type=int, default = 2)
parser.add_argument('--size',type=int, default = 512)
parser.add_argument('--cont',type=bool,default=True)
parser.add_argument('--f',type=int,default=0)
parser.add_argument('--v',type=int,default=1)
#parser.add_argument('--t',type=int,default=4)
parser.add_argument('--epoch',type=int,default=5)
parser.add_argument('--coords',default='start')
args = parser.parse_args()

mutex = Lock()

seed = 287942

batch_size = 534
hidden_units = args.size
layers = args.layers

num_classes = 1
epochs = args.epoch

path = '/home/taliah/Documents/Course/Project/new_seizure/data/6464/h5'

files = []

for filename in os.listdir(path):
	if filename[5] != '4' and filename.endswith('h5'): 
		files += [filename]

print("batch = {}".format(batch_size))

folds = get_folds()

i = args.f

v = folds[i]
t = []
for j in files:
	if j not in v:
		t += [j]

train_gen = hist_data_generator(files = t,seed = seed, batch_size = batch_size,contiguous=args.cont,lock=mutex,coords=args.coords)
val_gen = hist_data_generator(files = v,seed = seed, batch_size = batch_size,contiguous=args.cont,lock=mutex,coords=args.coords)

sizes = []

for j in range(layers):
	sizes += [hidden_units]


if args.coords == 'start':
	model = coords_start_hist_model(layers,sizes)
else:
	model = coords_end_hist_model(layers,sizes)


lrate = LearningRateScheduler(step_decay)
csv_logger = CSVLogger('logs/full_hist_{}_{}x{}_{}.log'.format(i,layers,hidden_units,args.coords),append=True)

filepath = 'models/fold{}.{}l.{}h.coords-{}.weights.best.hdf5'.format(i,layers,hidden_units,args.coords)
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=verbose=args.v, save_best_only=True, mode='max')
callbacks_list = [checkpoint,lrate,csv_logger]

history = model.fit_generator(train_gen,
                    epochs=epochs,
                    verbose=args.v,
                    validation_data=val_gen, max_queue_size = 4, 
                    callbacks = callbacks_list)



keras.backend.clear_session()

del model
del train_gen
del val_gen



