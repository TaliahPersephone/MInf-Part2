#!/bin/bash

#python dense_lstm.py > logs/full_.log

#python lstm.py > logs/full_.log
#python coords_hist_run.py   --f 0 --size 1024 --coords end >> logs/full_coords_end_2x1024.log
#python coords_hist_run.py   --f 0 --size 1024 --coords end >> logs/full_coords_end_2x1024.log
#python coords_hist_run.py   --f 1 --size 1024 --coords end >> logs/full_coords_end_2x1024.log
#python coords_hist_run.py   --f 2 --size 1024 --coords end >> logs/full_coords_end_2x1024.log
#python coords_hist_run.py   --f 3 --size 1024 --coords end >> logs/full_coords_end_2x1024.log

#python combined.py --f 0 > logs/full_combined_mk1.log
#python combined.py --f 1 >> logs/full_combined_mk1.log
#python combined.py --f 2 >> logs/full_combined_mk1.log
#python combined.py --f 3 >> logs/full_combined_mk1.log
#
#python dense_lstm.py --last LSTM --cont True --f 3  --layers 3 >> logs/full_3x512Dense_LSTM_v4.log
#
#python dense_lstm.py --f 0 --layers 5 --size 512 >> logs/full_5x512Dense_LSTM.log
#python dense_lstm.py --f 1 --layers 2 --size 4096 >> logs/full_2x4096Dense_LSTM.log
#python dense_lstm.py --f 2 --layers 2 --size 4096 >> logs/full_2x4096Dense_LSTM.log
#python dense_lstm.py --f 3 --layers 2 --size 4096 >> logs/full_2x4096Dense_LSTM.log

python dense_lstm.py --f 0 --layers 3 --size 1024 >> logs/full_10x1024Dense_LSTM.log 
python dense_lstm.py --f 0 --layers 3 --size 1024 >> logs/full_10x1024Dense_LSTM.log 

python dense_lstm.py --f 0 --layers 10 --size 1024 >> logs/full_10x1024Dense_LSTM.log 
python dense_lstm.py --f 0 --layers 10 --size 1024 >> logs/full_10x1024Dense_LSTM.log 
python dense_lstm.py --f 1 --layers 10 --size 1024 >> logs/full_10x1024Dense_LSTM.log 
python dense_lstm.py --f 1 --layers 10 --size 1024 >> logs/full_10x1024Dense_LSTM.log 
python dense_lstm.py --f 2 --layers 10 --size 1024 >> logs/full_10x1024Dense_LSTM.log 
python dense_lstm.py --f 2 --layers 10 --size 1024 >> logs/full_10x1024Dense_LSTM.log 
python dense_lstm.py --f 3 --layers 10 --size 1024 >> logs/full_10x1024Dense_LSTM.log 
python dense_lstm.py --f 3 --layers 10 --size 1024 >> logs/full_10x1024Dense_LSTM.log 

python dense_lstm.py --f 0 --layers 10 --size 2048 >> logs/full_10x2048Dense_LSTM.log 
python dense_lstm.py --f 0 --layers 10 --size 2048 >> logs/full_10x2048Dense_LSTM.log 
python dense_lstm.py --f 1 --layers 10 --size 2048 >> logs/full_10x2048Dense_LSTM.log 
python dense_lstm.py --f 1 --layers 10 --size 2048 >> logs/full_10x2048Dense_LSTM.log 
python dense_lstm.py --f 2 --layers 10 --size 2048 >> logs/full_10x2048Dense_LSTM.log 
python dense_lstm.py --f 2 --layers 10 --size 2048 >> logs/full_10x2048Dense_LSTM.log 
python dense_lstm.py --f 3 --layers 10 --size 2048 >> logs/full_10x2048Dense_LSTM.log 
python dense_lstm.py --f 3 --layers 10 --size 2048 >> logs/full_10x2048Dense_LSTM.log 
