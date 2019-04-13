#!/bin/bash

#python dense_lstm.py > logs/full_.log

#python lstm.py > logs/full_.log

#python cnn.py --last LSTM --cont True --f 2 > logs/full_cnn_last_LSTM_v3.log

python dense_lstm.py --last LSTM --cont True --size 1024 > logs/full_2x1024Dense_LSTM.log

 
python dense_lstm.py --last LSTM --cont True  --layers 3 > logs/full_3x512Dense_LSTM.log


python dense_lstm.py --last LSTM --cont True --layers 4 > logs/full_4x512Dense_LSTM.log

python dense_lstm.py --last LSTM --cont True --layers 5 > logs/full_5x512Dense_LSTM.log

python dense_lstm.py --last LSTM --cont True  --layers 3 > logs/full_3x512Dense_LSTM_v2.log

python dense_lstm.py --last LSTM --cont True --size 1024 > logs/full_2x1024Dense_LSTM_v2.log

python dense_lstm.py --last LSTM --cont True --layers 4 > logs/full_4x512Dense_LSTM_v2.log

python dense_lstm.py --last LSTM --cont True --layers 5 > logs/full_5x512Dense_LSTM_v2.log
