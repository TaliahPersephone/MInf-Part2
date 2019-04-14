import csv
import os
import numpy as np


og = '1439328827509_00000{}_AZ324hrsno5and8_1_bb.csv'

save_path = 'new_features_{:06}-{:05}'

for i in range(5):
	print(i)
	filename = og.format(i)

	r = csv.DictReader(open(filename))
	new_features = None
	save_frame = None

	for row in r:
		frame = round(float(row['Frame_number']))

		if frame % 7500 >= 7476:
			continue

		if frame % 7500 == 0:
			print(frame)
			if new_features is not None:
				np.save(save_path.format(i,save_frame),new_features)
			save_frame = frame
			new_features = np.zeros((7476,4))
			old_x = None
			old_y = None
			prev_x = None
			prev_y = None
	
		curr_x = float(row['centre_x'])/1200
		curr_y = float(row['centre_y'])/500

		if prev_x is None:
			change_x = 0.0
			change2_x = 0.0
			change_y = 0.0
			change2_y = 0.0
			
		else:
			change_x = np.abs(curr_x - prev_x)
			change_y = np.abs(curr_y - prev_y)

			if old_x is None:
				change2_x = 0.0
				change2_y = 0.0
			else:
				change2_x = np.abs(curr_x - old_x)
				change2_y = np.abs(curr_y - old_y)
				


		feature = np.array([change_x, change_y, change2_x, change2_y])


		new_features[frame%7500,:] = feature

		old_x = prev_x
		old_y = prev_y

		prev_x = curr_x
		prev_y = curr_y

	np.save(save_path.format(i,save_frame),new_features)
