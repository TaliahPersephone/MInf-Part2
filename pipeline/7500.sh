

python main.py 					\
	--models 5x1024_coords_end.json 	\
	--start 7500				\
	--show 1				\
 	--test 000004-07500-targets.csv 	\
	--input 000004-07500.avi 		\
	--boxes 000004-07500-boxes.csv		\
	--matlab 000004-07500.npy 		\
	--data hist 				\
	--labels hist 				\
	--weights 5x1024_coords.68.hdf5	
