# -*- coding: utf-8 -*-


import numpy as np
import cv2 as cv

DISPLAY = 0
path = "C:/Users/Janilbols/DISS/data/video-chunks/"
fileName_list = [              
                 
                 '1439328827509_000000_AZ324hrsno5and8_1-30000-obj-0-Boxed.avi',
                 '1439328827509_000000_AZ324hrsno5and8_1-37500-obj-0-Boxed.avi',
                 '1439328827509_000000_AZ324hrsno5and8_1-45000-obj-0-Boxed.avi',
                 '1439328827509_000000_AZ324hrsno5and8_1-52500-obj-0-Boxed.avi',
                 '1439328827509_000000_AZ324hrsno5and8_1-60000-obj-0-Boxed.avi',
                 '1439328827509_000000_AZ324hrsno5and8_1-67500-obj-0-Boxed.avi',
                 '1439328827509_000000_AZ324hrsno5and8_1-75000-obj-0-Boxed.avi',
                 
                 '1439328827509_000001_AZ324hrsno5and8_1-00000-obj-0-Boxed.avi',
                 '1439328827509_000001_AZ324hrsno5and8_1-75000-obj-0-Boxed.avi',
                 '1439328827509_000001_AZ324hrsno5and8_1-15000-obj-0-Boxed.avi',
                 '1439328827509_000001_AZ324hrsno5and8_1-45000-obj-0-Boxed.avi',
                 '1439328827509_000001_AZ324hrsno5and8_1-67500-obj-0-Boxed.avi',
                 '1439328827509_000001_AZ324hrsno5and8_1-75000-obj-0-Boxed.avi',
                 '1439328827509_000001_AZ324hrsno5and8_1-82500-obj-0-Boxed.avi',
                 
                 '1439328827509_000002_AZ324hrsno5and8_1-00000-obj-0-Boxed.avi',
                 '1439328827509_000002_AZ324hrsno5and8_1-07500-obj-0-Boxed.avi',
                 '1439328827509_000002_AZ324hrsno5and8_1-15000-obj-0-Boxed.avi',
                 '1439328827509_000002_AZ324hrsno5and8_1-22500-obj-0-Boxed.avi',
                 '1439328827509_000002_AZ324hrsno5and8_1-30000-obj-0-Boxed.avi',
                 '1439328827509_000002_AZ324hrsno5and8_1-37500-obj-0-Boxed.avi',
                 '1439328827509_000002_AZ324hrsno5and8_1-52500-obj-0-Boxed.avi',
                 
                 '1439328827509_000003_AZ324hrsno5and8_1-00000-obj-0-Boxed.avi',
                 '1439328827509_000003_AZ324hrsno5and8_1-07500-obj-0-Boxed.avi',
                 '1439328827509_000003_AZ324hrsno5and8_1-30000-obj-0-Boxed.avi',
                 '1439328827509_000003_AZ324hrsno5and8_1-37500-obj-0-Boxed.avi',
                 '1439328827509_000003_AZ324hrsno5and8_1-45000-obj-0-Boxed.avi',
                 '1439328827509_000003_AZ324hrsno5and8_1-52500-obj-0-Boxed.avi',
                 '1439328827509_000003_AZ324hrsno5and8_1-60000-obj-0-Boxed.avi',
                 '1439328827509_000003_AZ324hrsno5and8_1-67500-obj-0-Boxed.avi',
                 
                 '1439328827509_000004_AZ324hrsno5and8_1-07500-obj-0-Boxed.avi',
                 '1439328827509_000004_AZ324hrsno5and8_1-22500-obj-0-Boxed.avi',
                 '1439328827509_000004_AZ324hrsno5and8_1-60000-obj-0-Boxed.avi'  
                 ]
for fileName in fileName_list:
    print("Open file: ", fileName)
    cap = cv.VideoCapture(path+fileName)
    #cap = cv.VideoCapture(0)
    
    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'FLV1')
    
    fn = 0
    out = cv.VideoWriter(path+'/flv_cleaned/'+fileName[:-4]+'.flv',fourcc, 25.0, (32,32))
    
    print(cap.isOpened())
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        if fn%250 == 0:
            print(fn)
        if ret==True:
            #frame = cv.flip(frame,0)
            # write the flipped frame
            out.write(frame)
            if DISPLAY == 1:
                cv.imshow('frame',frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            break
        fn = fn + 1
    
    # Release everything if job is finished
    cap.release()
    out.release()
    cv.destroyAllWindows()