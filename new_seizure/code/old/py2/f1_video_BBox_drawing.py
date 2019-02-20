'''  
    ----   Usage of 'Video BoundingBox Drawing Tool'    ----
1.  package: cv2, video (file)
2.  press 'f' or left_click to 'freeze the video'
3.  while 'pause':
        press 'q' to quit the program
        press 'f' to continue the video 
        press 'b' to go back 'frame_step' frames
        press 'r' to reset BoundingBox
        press 'c' to confirm BoundingBox
        use 'left-click' and 'drag' then 'release' :
            From 'top-left' to 'bottom-right' to set the BoundingBox

        TODO: press 's' to save the progress
'''

import cv2
import numpy as np
import time
import os
import sys 


#path to folder of files: video
sys.path.append('C:/Users/Janilbols/DISS/supported') 
from video import Video as video
#from bgm_vibe import ViBeBGModelUnivariate


# initialize the list of reference points and boolean indicating
FLAG_RUN = 1
FLAG_PAUSE = 1
# whether cropping is being performed or not
refPt = []
cropping = False

# frame-num and boundingbox list
listLen = 0
fn_list = []
pt_0_0_list = []
pt_0_1_list = []
pt_1_0_list = []
pt_1_1_list = []
obj_list = []
obj_tag = 0


obj_tag = input("Obj_ID: (int)\n")
obj_tag = int(obj_tag)

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
    global refPt, cropping, FLAG_PAUSE

	
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        FLAG_PAUSE = 1
        refPt = [(x, y)]
        cropping = True

	# check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        FLAG_PAUSE = 1 #
        refPt.append((x, y))
        cropping = False
        if refPt[0][0]>refPt[1][0]:
            t = refPt[0][0]
            refPt[0][0] = refPt[1][0]
            refPt[1][0] = t
        if refPt[0][1]>refPt[1][1]:
            t = refPt[0][1]
            refPt[0][1] = refPt[1][1]
            refPt[1][1] = t
        # draw a rectangle around the region of interest
        if refPt[0][0]==refPt[1][0] | refPt[0][1]==refPt[1][1]:
            print("Invalid BoundingBox")
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image)


    


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

videoName_chunks_list = []
for vn in videoName_list:
    for nl in num_list:
        videoName_chunks_list.append(vn + '-' + nl +'.flv')

vn = 4*13 + 8 # video Number
N_vn = len(videoName_chunks_list)

videoName = videoName_chunks_list[vn]

v = video(path + videoName)
MAX_fn = v.get_length()

print("Video Open: ", videoName)


fn = 0 
frame_step = 25


fileName = videoName[:-4]+"-obj-"+str(obj_tag)+"-BBox.npz"

if os.path.exists(path+fileName):
    fileOpen = np.load(path+fileName)
    listLen = fileOpen["fLen"] #file length
    obj_list = fileOpen["obj_list"] #object tag list
    fn_list = fileOpen["fn_list"] # frame number list
    print fn_list
    pt_0_0_list = fileOpen["pt_0_0_list"] #top-left x
    pt_0_1_list = fileOpen["pt_0_1_list"] #top-left y
    pt_1_0_list = fileOpen["pt_1_0_list"] #bottom-right x
    pt_1_1_list = fileOpen["pt_1_1_list"] #bottom-right y
    if len(fn_list)>0:
        fn = int(fn_list[-1])
    print("BBox File Loaded: ", fileName)
    print("last BBox: ",pt_0_0_list[-1],pt_0_1_list[-1],pt_1_0_list[-1],pt_1_1_list[-1])
    IF_LOADFILE = 1
else:
    IF_LOADFILE = 0
    print("BBox File Not Found.")



TEMP = 1
while IF_LOADFILE:
    frame = v.get_frame(fn)
    image = frame
    refPt = [(int(pt_0_0_list[-1]),int(pt_0_1_list[-1]))]
    refPt.append((int(pt_1_0_list[-1]),int(pt_1_1_list[-1])))
    cv2.imshow('image',image)
    if TEMP>0:
        print refPt
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image)
        TEMP = 0
    key = cv2.waitKey(1) & 0xFF
    if key == ord('f'):
        fn = fn + frame_step
        IF_LOADFILE = 0

while(FLAG_RUN and fn < MAX_fn):
    
    frame = v.get_frame(fn)
    image = frame
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('image',image)
    
    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
    print("fn = ",fn)
    
    
    FLAG_PAUSE = 1
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('f'):
        FLAG_PAUSE = 1
    
    # keep looping until the 'q' key is pressed
    while FLAG_PAUSE:
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF
        
    
        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            refPt = None
            image= clone.copy()
    
        # if the 'c' key is pressed, break from the loop
        if key == ord("c"):
            # if there are two reference points, then crop the region of interest
            # from teh image and display it
            print("BBox Drawn")
            if len(refPt)>1:
                if refPt[0][0]==refPt[1][0] | refPt[0][1]==refPt[1][1]:
                    continue
                # Update BBox list
                if listLen==0:
                    fn_list = np.append(fn_list, fn)
                    pt_0_0_list = np.append(pt_0_0_list, refPt[0][0])
                    pt_0_1_list = np.append(pt_0_1_list, refPt[0][1])
                    pt_1_0_list = np.append(pt_1_0_list, refPt[1][0])
                    pt_1_1_list = np.append(pt_1_1_list, refPt[1][1])
                    obj_list = np.append(obj_list, obj_tag)
                    listLen = 1
                else:
                    ln = listLen - 1
                    if (fn == fn_list[ln]) and (obj_tag == obj_list[ln]) :
                        pt_0_0_list[ln] = refPt[0][0]
                        pt_0_1_list[ln] = refPt[0][1]
                        pt_1_0_list[ln] = refPt[1][0]
                        pt_1_1_list[ln] = refPt[1][1]
                    else:
                        fn_list = np.append(fn_list, fn)
                        obj_list = np.append(obj_list, obj_tag)
                        pt_0_0_list = np.append(pt_0_0_list, refPt[0][0])
                        pt_0_1_list = np.append(pt_0_1_list, refPt[0][1])
                        pt_1_0_list = np.append(pt_1_0_list, refPt[1][0])
                        pt_1_1_list = np.append(pt_1_1_list, refPt[1][1])
                        listLen = listLen + 1
                # end of update BBox list
                
                # Display the Cut img based on BBox
                if refPt[1][1] > refPt[0][1] and refPt[1][0] > refPt[0][0]:
                    roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
                    cv2.imshow("ROI", roi)
                    FLAG_PAUSE = 0
                
                # reset parameter
                #roi = None
                #refPt = None
                # Continue
                
                if fn%200 ==0:
                    np.savez(path+videoName[:-4]+"-obj-"+str(obj_tag)+"-BBox.npz",fLen = listLen, fn_list = fn_list,
                             pt_0_0_list=pt_0_0_list, pt_0_1_list = pt_0_1_list, 
                             pt_1_0_list = pt_1_0_list, pt_1_1_list = pt_1_1_list, obj_list = obj_list)
                    print("File auto-saved!")
        elif key == ord("f"):
            # Continue
            FLAG_PAUSE = 0
            
        elif key == ord("b"):
            fn = fn - frame_step * 10
            if fn<0:
                fn = 0
            break
        elif key == ord("n"):
            fn = fn + frame_step * 50
            break
        elif key == ord("q"):
            FLAG_RUN = 0
            break
        elif key == ord("s"):
            np.savez(path+videoName[:-4]+"-obj-"+str(obj_tag)+"-BBox.npz",fLen = listLen, fn_list = fn_list,
                     pt_0_0_list=pt_0_0_list, pt_0_1_list = pt_0_1_list, 
                     pt_1_0_list = pt_1_0_list, pt_1_1_list = pt_1_1_list, obj_list = obj_list)
            print("File saved!")
        elif key == ord("1"):
            #obj_tag = 1
            print(1)
        elif key == ord("2"):
            #obj_tag = 2
            print(2)
        
    #end of FLAG_PAUSE
       
    time.sleep(1)
    fn = fn + frame_step
#end of FLAG_RUN
    
# close all open windows
    
np.savez(path+videoName[:-4]+"-obj-"+str(obj_tag)+"-BBox.npz",fLen = listLen, fn_list = fn_list,
                     pt_0_0_list=pt_0_0_list, pt_0_1_list = pt_0_1_list, 
                     pt_1_0_list = pt_1_0_list, pt_1_1_list = pt_1_1_list, obj_list = obj_list)
print("File saved!")

v.close()  # Release handle to underlying video file.
cv2.destroyAllWindows()



