# -*- coding: utf-8 -*-


import numpy as np
import sys
'''
    NOW: Assume only 1 obj 
    TODO: consider different obj-tag
'''


#path = 'C:/Users/Janilbols/Desktop/DISS/'
#fileName = 'example_seizure-BBox.npz'

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
fileName = videoName[:-4]+"-obj-"+str(obj_id)+"-BBox.npz"


fileOpen = np.load(path+fileName)




fLen = fileOpen["fLen"] #file length
obj_list = fileOpen["obj_list"] #object tag list
fn_list = fileOpen["fn_list"] # frame number list
pt_0_0_list = fileOpen["pt_0_0_list"] #top-left x
pt_0_1_list = fileOpen["pt_0_1_list"] #top-left y
pt_1_0_list = fileOpen["pt_1_0_list"] #bottom-right x
pt_1_1_list = fileOpen["pt_1_1_list"] #bottom-right y


if fLen<1 :
    print("fLen < 1")
    sys.exit(1)


sorted_indices = np.argsort(fn_list, axis=0)
fn_list_sorted = fn_list[sorted_indices]
obj_list_sorted = np.array( obj_list[sorted_indices])
pt_0_0_list_sorted = pt_0_0_list[sorted_indices]
pt_0_1_list_sorted = pt_0_1_list[sorted_indices]
pt_1_0_list_sorted = pt_1_0_list[sorted_indices]
pt_1_1_list_sorted = pt_1_1_list[sorted_indices]


fn_prev = fn_list_sorted[0]
pt_0_0_prev = pt_0_0_list_sorted[0]
pt_0_1_prev = pt_0_1_list_sorted[0]
pt_1_0_prev = pt_1_0_list_sorted[0]
pt_1_1_prev = pt_1_1_list_sorted[0]

new_fn_list = []
new_pt_0_0_list = []
new_pt_0_1_list = []
new_pt_1_0_list = []
new_pt_1_1_list = []

for i in range(1,fLen):
    fn_next = fn_list[i]
    pt_0_0_next = pt_0_0_list_sorted[i]
    pt_0_1_next = pt_0_1_list_sorted[i]
    pt_1_0_next = pt_1_0_list_sorted[i]
    pt_1_1_next = pt_1_1_list_sorted[i]
    
    if fn_next-fn_prev == 0:
        continue
    new_fn_list = np.append(new_fn_list, np.arange(fn_prev,fn_next))
    #pt 0 0
    pt_prev = pt_0_0_prev
    pt_next = pt_0_0_next
    nstep = fn_next - fn_prev
    new_pt_0_0_list = np.append(new_pt_0_0_list,np.linspace(pt_prev,pt_next,nstep,endpoint=False))
    #step_pt = (pt_next - pt_prev)*1.0/(fn_next - fn_prev)
    #print("pt 0 0",pt_prev,pt_next,step_pt)
    #if step_pt==0:
    #    new_pt_0_0_list = np.append(new_pt_0_0_list,np.ones(fn_next-fn_prev)*pt_next)
    #else:
    #    new_pt_0_0_list = np.append(new_pt_0_0_list, np.arange(pt_prev,pt_next,step_pt))
    
    #pt 0 1
    pt_prev = pt_0_1_prev
    pt_next = pt_0_1_next
    new_pt_0_1_list = np.append(new_pt_0_1_list,np.linspace(pt_prev,pt_next,nstep,endpoint=False))
    
    #pt 1 0
    pt_prev = pt_1_0_prev
    pt_next = pt_1_0_next
    new_pt_1_0_list = np.append(new_pt_1_0_list,np.linspace(pt_prev,pt_next,nstep,endpoint=False))

    
    #pt 1 1
    pt_prev = pt_1_1_prev
    pt_next = pt_1_1_next
    new_pt_1_1_list = np.append(new_pt_1_1_list,np.linspace(pt_prev,pt_next,nstep,endpoint=False))
    
    
    #update
    fn_prev = fn_next
    pt_0_0_prev = pt_0_0_next
    pt_0_1_prev = pt_0_1_next
    pt_1_0_prev = pt_1_0_next
    pt_1_1_prev = pt_1_1_next
#End of creating loop
    
    
new_fn_list = np.append(new_fn_list,fn_list_sorted[-1])
new_pt_0_0_list = np.append(new_pt_0_0_list,pt_0_0_list_sorted[-1])
new_pt_0_1_list = np.append(new_pt_0_1_list,pt_0_1_list_sorted[-1])
new_pt_1_0_list = np.append(new_pt_1_0_list,pt_1_0_list_sorted[-1])
new_pt_1_1_list = np.append(new_pt_1_1_list,pt_1_1_list_sorted[-1])


np.savez(path+fileName[:-4]+"-continued.npz",fn_list = new_fn_list, 
         pt_0_0_list = new_pt_0_0_list, pt_0_1_list = new_pt_0_1_list,
         pt_1_0_list = new_pt_1_0_list, pt_1_1_list = new_pt_1_1_list)

