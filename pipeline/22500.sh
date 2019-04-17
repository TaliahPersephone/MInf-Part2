

python main.py 					\
	--models 5x1024_coords_end.json 	\
	--start 22500				\
 	--test 000004-22500-targets.csv 	\
	--input 000004-22500.avi 		\
	--boxes 000004-22500-boxes.csv		\
	--matlab 000004-22500.npy 		\
	--data hist 				\
	--labels bum 				\
	--weights 5x1024_coords.68.hdf5	
