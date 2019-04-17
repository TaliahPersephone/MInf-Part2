

python main.py 					\
	--models 5x1024_coords_end.json 	\
	--start 60000				\
 	--test 000004-60000-targets.csv 	\
	--show 1				\
	--input 000004-60000.avi 		\
	--boxes 000004-60000-boxes.csv		\
	--matlab 000004-60000.npy 		\
	--data hist 				\
	--labels hist 				\
	--weights 5x1024_coords.68.hdf5	
