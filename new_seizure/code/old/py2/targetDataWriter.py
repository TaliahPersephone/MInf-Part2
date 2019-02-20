# -*- coding: utf-8 -*-

import csv
import numpy as np
import time



def findChunk(ifSeizure):
    fn = 0
    FLAG = 0
    chunk_num_list = np.zeros(int(len(ifSeizure)/7500)+1)

    while fn < len(ifSeizure):
        if FLAG==1:
            chunk_num_list[int(fn/7500)] = 1
        if FLAG==0 and ifSeizure[fn]>0:
            print("Chunk-Start:\t", fn)
            FLAG = 1
            
        if FLAG==1 and ifSeizure[fn]==0:
            print("Chunk-End:\t", fn-1)
            print("-------------------")
            FLAG = 0
            
        fn = fn + 1
    
    count = 0
    for t in chunk_num_list:
        if t>0:
            print count*7500
        count = count+1



path = '/home/taliah/Documents/Course/Project/new_seizure/temporal_annotations/'
fileName = '1439328827509_000004_AZ324hrsno5and8_1.csv'
title = []
data = []
ifSeizure = []

print("open file:", path+fileName)

with open(path+fileName) as csvDataFile:
    csvReader = csv.reader(csvDataFile,delimiter=',')
    count = 0
    for row in csvReader:
        if count<1:
            for t in row:
                title.append(t)
        else:
            line = []
            for value in row:
                line.append(float(value))
                
            ifSeizure.append( sum(line[1:]) )
            data.append(line)
        count = count + 1
#print(data)
#print(ifSeizure)
#print(np.shape(ifSeizure))
pos = np.array(ifSeizure)
neg = (pos<1)*-1
new_target = pos+neg
#print(sum(new_target))
#print(neg)


'''
Started to save file
'''

fn = 0
chunkSize = 7500

while fn<len(new_target):
    #if fn%250 == 0:
    #    print('progress: ',fn)
    if fn%chunkSize==0:
        if fn>0:
            print(np.shape(outputData))
            np.savez(outfileName, targets = outputData)
            print('file saved: ', fn, outfileName)
        outfileName = path+'/targets/'+fileName[:-4]+'-%05d-targets'%fn+'.npz'
        outputData = []
    outputData.append(new_target[fn])
    fn = fn + 1

print(len(new_target))
print(np.shape(outputData))
np.savez(outfileName, targets = outputData)
print('file saved: ', fn, outfileName)


#findChunk(ifSeizure)
