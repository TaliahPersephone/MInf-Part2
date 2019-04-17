import numpy as np
import os
import scipy.io as sio
import sys
import csv

path = '/home/taliah/Documents/Course/Project/new_seizure/data/{}/'.format(sys.argv[1])
condense = int(sys.argv[2])
limit = int(float(sys.argv[3]) * condense) 
count = [0,0,0]

save_path = path  + '{}/c{}-l{}-liblinearData.txt'.format('{}',condense,limit)

s = 0

for set_name in ['train/','val/','test/']:
	for filename in os.listdir(path + set_name):
		if filename.endswith('.mat'):
			count[s] += 1
	s += 1



mat = sio.loadmat(path+'train/0000.mat')
size = (0,  1 + mat['dataFull'].shape[1])



s_n = 0
for set_name in ['train/','val/','test/']:
	try:
		os.remove(save_path.format(set_name))
	except OSError:
		pass

	print(set_name)
	for i in range(count[s_n]):
		print(i)
		mat = sio.loadmat(path+set_name+'{:04}.mat'.format(i))
		f = open(path+set_name+'targets/c{}-l{}/{:04}.csv'.format(condense,limit,i))
		targets_csv = csv.reader(f)
		targets = []

		for row in targets_csv:
			targets += [row[1]]

		f.close()

		temp_mat = np.append(np.array(targets,ndmin=2,dtype=np.float32).T,mat['dataFull'],axis=1)
		
		with open(save_path.format(set_name),'a+') as s:
			for j in range(temp_mat.shape[0]): 
				if(temp_mat[j][0] == 1):
					s.write('+1 ')
				else:
					s.write('-1 ')

				for k in range(1,temp_mat.shape[1]):
					s.write('{}:{:.6f} '.format(k,temp_mat[j][k]))

				s.write('\n')
			#np.savetxt(s,temp_mat,fmt='%10.6f',delimiter=' ',newline='\n')
			

	s_n+=1

		#print('{}\t{}'.format(np.array(targets,ndmin=2).T.shape,mat['dataFull'].shape))



		
