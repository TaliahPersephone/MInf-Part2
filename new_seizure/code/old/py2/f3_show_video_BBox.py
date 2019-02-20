# -*- coding: utf-8 -*-


import cv2
import numpy as np
import time

import sys 
#path to folder of files: video
sys.path.append('C:/Users/Janilbols/DISS/supported') 
from video import Video as video

FLAG_DISPLAY = 1
IF_DEBUG = 0
# Video Reading
#path = 'C:/Users/Janilbols/Desktop/DISS/'
#videoName = 'example_seizure.flv'


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
obj_id = input("Obj_ID: (int)\n")
obj_id = int(obj_id)

videoName_chunks_list = []
for vn in videoName_list:
    for nl in num_list:
        videoName_chunks_list.append(vn + '-' + nl +'.flv')
        
N_vn = len(videoName_chunks_list)

vn = 4*13 + 8 # video Number

videoName = videoName_chunks_list[vn]
fileName = videoName[:-4]+"-obj-"+str(obj_id)+"-BBox-continued.npz"
print("Feading File: ", fileName)


#videoName = '1439328827509_000000_AZ324hrsno5and8_1-30000.flv'
#fileName = '1439328827509_000000_AZ324hrsno5and8_1-30000-BBox-obj-0.npz'


in_res = [1200,500] # input video resolution
out_res = [32,32] # output boxed video resolution

v = video(path+videoName)
out = cv2.VideoWriter(path+videoName[:-4]+"-obj-"+str(obj_id)+'-Boxed.avi',-1, 25.0, (out_res[0],out_res[1]))
#fourcc = cv2.cv.CV_FOURCC(*'XVID')
#out = cv2.VideoWriter(path+videoName[:-4]+'-Boxed.avi',fourcc, 25.0, (out_res[0],out_res[1]))

fopen = np.load(path+fileName)

fn_list = fopen["fn_list"]
pt_0_0_list = fopen["pt_0_0_list"] #top-left x
pt_0_1_list = fopen["pt_0_1_list"] #top-left y
pt_1_0_list = fopen["pt_1_0_list"] #bottom-right x
pt_1_1_list = fopen["pt_1_1_list"] #bottom-right y

print("data have been read!")

def squareWindow(pt):
    m_x = int((pt[0][0]+pt[1][0])/2)
    m_y = int((pt[0][1]+pt[1][1])/2)
    center = [m_x, m_y]
    #print("center ", center)
    x_half_len = int((pt[1][0]-pt[0][0])/2)
    y_half_len = int((pt[1][1]-pt[0][1])/2)
    half_len = max(x_half_len,y_half_len)
    return [(center[0]-half_len,center[1]-half_len),(center[0]+half_len,center[1]+half_len)]


print("start the loop:")
for i in np.arange(len(fn_list)):
    fn = fn_list[i]
    if fn%200==0:
        print("fn = ", fn)
            
    frame = v.get_frame(fn)
    clone = frame.copy()
    
    pt = [(int(pt_0_0_list[i]),int(pt_0_1_list[i])),(int(pt_1_0_list[i]),int(pt_1_1_list[i]))]
    recPt = squareWindow(pt)
    recPt = np.array(recPt)
    #print recPt #DEBUG
    #print("pt", pt)
    #print("recPt", recPt)
    
        
    cv2.imshow('frame',frame)
    cv2.rectangle(frame, pt[0], pt[1], (0, 255, 0), 2)
    cv2.imshow("frame", frame)
    
    
    #roi = clone[pt[0][1]:pt[1][1], pt[0][0]:pt[1][0]]
    res = np.array([(0,0),(0,0)])
    
    #cv2.imshow("ROI", roi)
    len_h = int(recPt[1][1]-recPt[0][1])
    len_w = int(recPt[1][0]-recPt[0][0])
    
    if recPt[0][0]<0:
        print("recPt[0][0]",recPt[0][0])
        res[0][0] = 0 - recPt[0][0]
        recPt[0][0] = 0
    if recPt[0][1]<0:
        print("recPt[0][1]",recPt[0][1])
        res[0][1] = 0 - recPt[0][1]
        recPt[0][1] = 0
    if recPt[1][0]>in_res[0]:
        print("recPt[1][0]",recPt[1][0])
        res[1][0] = recPt[1][0] - in_res[0]
        recPt[0][1] = in_res[0]
    if recPt[1][1]>in_res[1]:
        print("recPt[1][1]",recPt[1][1])
        res[1][1] = recPt[1][1] - in_res[1]
        recPt[1][1] = in_res[1]
        
    if np.sum(res) == 0:
        recRoi = clone[recPt[0][1]:recPt[1][1], recPt[0][0]:recPt[1][0]]
    else:
        '''
        CODE UN-TESTED
        '''
        
        cut = clone[recPt[0][1]:recPt[1][1], recPt[0][0]:recPt[1][0]]
        height_cut = recPt[1][1] - recPt[0][1]
        width_cut = recPt[1][0] - recPt[0][0]
        if IF_DEBUG > 0:
            print("cut shape = ", np.shape(cut))
            print("len_h & w = ", len_h,len_w)
            print("cut_h & w = ", height_cut, width_cut)
            cv2.imshow("cut",cut)
        recRoi = np.zeros([len_h,len_w,3], dtype=np.uint8)
        recRoi[res[0][1]:res[0][1]+height_cut,res[0][0]:res[0][0]+width_cut] = cut

        
        
    
    resize = cv2.resize(recRoi, (out_res[0], out_res[1]))
    
    out.write(resize)
    
    if FLAG_DISPLAY>0:
        cv2.imshow("recROI", recRoi)
        cv2.imshow("resize",resize)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            FLAG_DISPLAY = 0
    
    if IF_DEBUG and fn>2800:
        LOOP = 1
        print("pause")
        FLAG_DISPLAY = 1
        while (LOOP>0):
            key = cv2.waitKey(1) & 0xFF
            if key == ord("f"):
                LOOP = 0
            if key == ord("g"):
                fn = fn + 25
            if key == ord("d"):
                print("fn:", fn)
                print("shape-recRoi",np.shape(recRoi))
                print(len_h,len_w)
                print(height_cut,width_cut)
                print(recPt[0][0])
                print(recPt[0][1])
                print(recPt[1][0])
                print(recPt[1][1])
                print(res[0][0])
                print(res[0][1])
                print(res[1][0])
                print(res[1][1])
            
                
                
    
    
print("close all windows & Shut Down.")
v.close()  # Release handle to underlying video file.
out.release()
cv2.destroyAllWindows()
    