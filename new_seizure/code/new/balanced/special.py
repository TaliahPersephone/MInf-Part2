import keras
import argparse
from tp_tn import tp, tn
from keras.models import Sequential
from keras.layers import SimpleRNN,Dense, BatchNormalization, LSTM, Reshape
from keras.optimizers import RMSprop, Adam
from hist_data_generator import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,CSVLogger
import sys
import os
from threading import Lock
import numpy as np
import random
from get_folds import *
from hist_model import *
from lr import step_decay

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
parser = argparse.ArgumentParser()
parser.add_argument('--layers', type=int, default = 5)
parser.add_argument('--size',type=int, default = 1024)
parser.add_argument('--last',default='LSTM')
parser.add_argument('--cont',type=str2bool,default=True)
parser.add_argument('--f',type=int,default=5)
parser.add_argument('--v',type=int,default=1)
#parser.add_argument('--t',type=int,default=4)
parser.add_argument('--epoch',type=int,default=5)
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


if args.f == 5:
	t = files
	v = []
	for filename in os.listdir(path):
		if filename[5] == '4' and filename.endswith('h5'): 
			v += [filename]
else:
	v = folds[i]
	t = []
	for j in files:
		if j not in v:
			t += [j]
	
	

train_gen = hist_data_generator(files = t,seed = seed, batch_size = batch_size,contiguous=args.cont,lock=mutex)
val_gen = hist_data_generator(files = v,orientations=[''],seed = seed, batch_size = batch_size,contiguous=args.cont,lock=mutex,coords=True)

sizes = []

for j in range(layers):
	sizes += [hidden_units]



model = coords_end_hist_model(layers,sizes)
lrate = LearningRateScheduler(step_decay)

filepath = '5x1024_coords.68.hdf5'

model.load_weights(filepath)

score = model.predict_generator(val_gen,max_queue_size=4, verbose=1)

print(score)

keras.backend.clear_session()

del model
del train_gen
del val_gen



