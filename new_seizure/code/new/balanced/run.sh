#!/bin/bash

#python dense_lstm.py > logs/full_.log

#python lstm.py > logs/full_.log
#python coords_hist_run.py   --f 0 --size 1024 --coords end >> logs/full_coords_end_2x1024.log
#python coords_hist_run.py   --f 0 --size 1024 --coords end >> logs/full_coords_end_2x1024.log
python coords_hist_run.py   --f 1 --size 1024 --coords end >> logs/full_coords_end_2x1024.log
python coords_hist_run.py   --f 2 --size 1024 --coords end >> logs/full_coords_end_2x1024.log
python coords_hist_run.py   --f 3 --size 1024 --coords end >> logs/full_coords_end_2x1024.log

#python cnn.py --f 0 > logs/cnn_loss_test.log
python cnn.py --f 0 >> logs/full_cnn_last_LSTM_coords.log
python cnn.py --f 2 >> logs/full_cnn_last_LSTM_coords.log
python cnn.py --f 3 >> logs/full_cnn_last_LSTM_coords.log
#
#python cnn.py --f 0 >> logs/full_cnn_last_LSTM_coords.log
#python cnn.py --f 1 >> logs/full_cnn_last_LSTM_coords.log
#python cnn.py --f 2 >> logs/full_cnn_last_LSTM_coords.log
#python cnn.py --f 3 >> logs/full_cnn_last_LSTM_coords.log
#python dense_lstm.py --last LSTM --cont True --f 3  --layers 3 >> logs/full_3x512Dense_LSTM_v4.log
#
#python dense_lstm.py --last LSTM --cont True --f 0 --layers 3 --size 1024 >> logs/full_3x1024Dense_LSTM.log
#python dense_lstm.py --last LSTM --cont True --f 1 --layers 3 --size 1024 >> logs/full_3x1024Dense_LSTM.log
#python dense_lstm.py --last LSTM --cont True --f 2 --layers 3 --size 1024 >> logs/full_3x1024Dense_LSTM.log
#python dense_lstm.py --last LSTM --cont True --f 3 --layers 3 --size 1024 >> logs/full_3x1024Dense_LSTM.log


#python dense_lstm.py --last LSTM --cont True --f 0 --layers 4 >> logs/full_4x512Dense_LSTM_v4.log

#python dense_lstm.py --last LSTM --cont True --f 0 --size 2048 >> logs/full_2x2048Dense_LSTM.log
#python dense_lstm.py --last LSTM --cont True --f 1 --size 2048 >> logs/full_2x2048Dense_LSTM.log
#python dense_lstm.py --last LSTM --cont True --f 2 --size 2048 >> logs/full_2x2048Dense_LSTM.log
#python dense_lstm.py --last LSTM --cont True --f 3 --size 2048 >> logs/full_2x2048Dense_LSTM.log
#
#python dense_lstm.py --last LSTM --cont True --f 1 --size 1024 >> logs/full_2x1024Dense_LSTM.log
#python dense_lstm.py --last LSTM --cont True --f 2 --size 1024 >> logs/full_2x1024Dense_LSTM.log
#python dense_lstm.py --last LSTM --cont True --f 3 --size 1024 >> logs/full_2x1024Dense_LSTM.log
#
#python coords_hist_run.py   --f 0 --size 1024 >> logs/full_coords_start_2x1024.log
#python coords_hist_run.py   --f 0 --size 1024 >> logs/full_coords_start_2x1024.log
#python coords_hist_run.py   --f 0 --size 1024 >> logs/full_coords_start_2x1024.log
##python coords_hist_run.py   --f 1 --size 1024 >> logs/full_coords_start_2x1024.log
##python coords_hist_run.py   --f 2 --size 1024 >> logs/full_coords_start_2x1024.log
##python coords_hist_run.py   --f 3 --size 1024 >> logs/full_coords_start_2x1024.log
# 
#python coords_hist_run.py   --f 0 --size 1024 --coords end >> logs/full_coords_end_2x1024.log
#python coords_hist_run.py   --f 0 --size 1024 --coords end >> logs/full_coords_end_2x1024.log
