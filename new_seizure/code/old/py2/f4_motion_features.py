# -*- coding: utf-8 -*-


import cv2
import numpy as np
import time

import sys 
#path to folder of files: video
sys.path.append('C:/Users/Janilbols/DISS/supported') 

'''
Based on the result from 'f2_create_continuous_BBox.py',
Datastructure:  (frame_num,feature_num)    
Features:       [Central_point x, Central_point y, Width, Height, Speed x, Speed y, Acceleration x, Acceleration y] 

'''


path = 'C:/Users/Janilbols/DISS/data/video-chunks/'
videoName_list = ['1439328827509_000000_AZ324hrsno5and8_1',
                  '1439328827509_000001_AZ324hrsno5and8_1',
                  '1439328827509_000002_AZ324hrsno5and8_1',
                  '1439328827509_000003_AZ324hrsno5and8_1',
                  '1439328827509_000004_AZ324hrsno5and8_1',
                  ]
num_list = ['00000',  '07500', '15000', '22500', '30000', '37500', '45000', '52500', '60000',
       '67500', '75000', '82500', '90000']
obj_id = 0
#obj_id = input("Obj_ID: (int)\n")
#obj_id = int(obj_id)


videoName_chunks_list = []
for vn in videoName_list:
    for nl in num_list:
        videoName_chunks_list.append(vn + '-' + nl +'.flv')

N_vn = len(videoName_chunks_list)

vn = 2*13 +7 # video Number

videoName = videoName_chunks_list[vn]
fileName = videoName[:-4]+"-obj-"+str(obj_id)+"-BBox-continued.npz"
print("Feading File: ", fileName)


fopen = np.load(path+fileName)

fn_list = fopen["fn_list"]
pt_0_0_list = fopen["pt_0_0_list"] #top-left x
pt_0_1_list = fopen["pt_0_1_list"] #top-left y
pt_1_0_list = fopen["pt_1_0_list"] #bottom-right x
pt_1_1_list = fopen["pt_1_1_list"] #bottom-right y

print("data have been read!")





feature_list = []

i=0
pt = [(int(pt_0_0_list[i]),int(pt_0_1_list[i])),(int(pt_1_0_list[i]),int(pt_1_1_list[i]))] 
c_x = int((pt[0][0]+pt[1][0])/2)
c_y = int((pt[0][1]+pt[1][1])/2)    
c_prev = np.array([c_x,c_y])
v_prev = np.array([0,0])

print("start the loop:")
for i in np.arange(len(fn_list)):
    fn = fn_list[i]
    if fn%200==0:
        print("fn = ", fn)
        
    pt = [(int(pt_0_0_list[i]),int(pt_0_1_list[i])),(int(pt_1_0_list[i]),int(pt_1_1_list[i]))]
    c_x = int((pt[0][0]+pt[1][0])/2)
    c_y = int((pt[0][1]+pt[1][1])/2)
    width = int((pt[1][0]-pt[0][0]))
    height = int((pt[1][1]-pt[0][1]))
    
    c = np.array([c_x, c_y])
    v = c-c_prev
    a = v-v_prev
    
    feature = np.append(c,[width,height])
    feature = np.append(feature,v)
    feature = np.append(feature,a)
    feature_list.append(feature)
    
path_save = 'C:/Users/Janilbols/DISS/data/Motion/'
np.savez(path_save+videoName[:-4]+"-obj-"+str(obj_id)+"-motion.npz", feature = feature_list)                
                
    
#print(feature_list[:10])
print("Done.")
    