import matplotlib.pyplot as plt
import numpy as np
import csv


m_2x512_tp 	= [36.309,15.0824964998062]
m_2x512_acc	= [63.365,5.66345889599869]						
m_3x512_tp 	= [25.2798,17.9017745186522]						
m_3x512_acc	= [62.18,9.59180205522751]
m_4x512_tp	= [27.43,8.83257097339161]						
m_4x512_acc	= [64.785,6.67124426175507]
m_5x512_tp 	= [36.4938,14.15678038667]
m_5x512_acc	= [67.274,7.86160441640254]
m_10x512_tp	= [36.3575,7.18575616526658]
m_10x512_acc	= [69.637,6.11178015529573]

m_2x1024_tp 	= [52.6453,22.5194339534397]
m_2x1024_acc 	= [68.3625,10.5631951447782]
m_3x1024_tp	= [37.8388,19.266539256355]
m_3x1024_acc 	= [66.86,10.1108093972079]
m_5x1024_tp 	= [40.31,4.45573787379823]
m_5x1024_acc	= [75.3255,4.04204820192271]
m_10x1024_tp	= [61.825,14.7938917575239]
m_10x1024_acc	= [72.7753,2.53396282201087]

m_2x2048_tp	= [47.3445,26.1439147859178]
m_2x2048_acc	= [69.1,9.03950588620123]
m_5x2048_tp	= [60.8913,10.5658115124522]
m_5x2048_acc	= [71.8625,3.53452160836513]
m_10x2048_tp	= [18.5275,13.4853213408753]
m_10x2048_acc	= [57.7295,9.67464526481462]

m_2x4096_tp	= [47.3625,13.5890136384753]
m_2x4096_acc	= [67.722,4.74843139011892]

m_2x8192_tp	= [27.1675,20.338091314903]
m_2x8192_acc	= [59.4868,11.0598992272383]


size_512_x = np.array([2,3,4,5,10])
size_512_acc 	= np.array([m_2x512_acc[0],m_3x512_acc[0],m_4x512_acc[0],m_5x512_acc[0],m_10x512_acc[0]])
size_512_acc_err = np.array([m_2x512_acc[1],m_3x512_acc[1],m_4x512_acc[1],m_5x512_acc[1],m_10x512_acc[1]])

size_1024_x = np.array([2,3,5,10])
size_1024_acc 	= np.array([m_2x1024_acc[0],m_3x1024_acc[0],m_5x1024_acc[0],m_10x1024_acc[0]])
size_1024_acc_err = np.array([m_2x1024_acc[1],m_3x1024_acc[1],m_5x1024_acc[1],m_10x1024_acc[1]])

size_2048_x = np.array([2,5,10])
size_2048_acc 	= np.array([m_2x2048_acc[0],m_5x2048_acc[0],m_10x2048_acc[0]])
size_2048_acc_err = np.array([m_2x2048_acc[1],m_5x2048_acc[1],m_10x2048_acc[1]])

size_4096_x = np.array([2])
size_4096_acc 	= np.array([m_2x4096_acc[0]])
size_4096_acc_err = np.array([m_2x4096_acc[1]])

size_8192_x = np.array([2])
size_8192_acc 	= np.array([m_2x8192_acc[0]])
size_8192_acc_err = np.array([m_2x8192_acc[1]])

size_512_x = np.array([2,3,4,5,10])
size_512_tp 	= np.array([m_2x512_tp[0],m_3x512_tp[0],m_4x512_tp[0],m_5x512_tp[0],m_10x512_tp[0]])
size_512_tp_err = np.array([m_2x512_tp[1],m_3x512_tp[1],m_4x512_tp[1],m_5x512_tp[1],m_10x512_tp[1]])

size_1024_x = np.array([2,3,5,10])
size_1024_tp 	= np.array([m_2x1024_tp[0],m_3x1024_tp[0],m_5x1024_tp[0],m_10x1024_tp[0]])
size_1024_tp_err = np.array([m_2x1024_tp[1],m_3x1024_tp[1],m_5x1024_tp[1],m_10x1024_tp[1]])

size_2048_x = np.array([2,5,10])
size_2048_tp 	= np.array([m_2x2048_tp[0],m_5x2048_tp[0],m_10x2048_tp[0]])
size_2048_tp_err = np.array([m_2x2048_tp[1],m_5x2048_tp[1],m_10x2048_tp[1]])

size_4096_x = np.array([2])
size_4096_tp 	= np.array([m_2x4096_tp[0]])
size_4096_tp_err = np.array([m_2x4096_tp[1]])

size_8192_x = np.array([2])
size_8192_tp 	= np.array([m_2x8192_tp[0]])
size_8192_tp_err = np.array([m_2x8192_tp[1]])

plt.figure(1, figsize=(18, 9))
plt.subplot(121)
plt.title('Accuracy Achieved by Histogram Models')
plt.bar(size_512_x-0.3,size_512_acc,width=0.15,color='r')#,yerr=size_512_acc_err)
plt.bar(size_1024_x-0.15,size_1024_acc,width=0.15,color='y')#,yerr=size_1024_acc_err)
plt.bar(size_2048_x,size_2048_acc,width=0.15,color='g')#,yerr=size_2048_acc_err)
plt.bar(size_4096_x+0.15,size_4096_acc,width=0.15,color='b')#,yerr=size_2048_acc_err)
plt.bar(size_8192_x+0.3,size_8192_acc,width=0.15,color='m')#,yerr=size_2048_acc_err)
plt.xlabel('Number of Layers')
plt.ylabel('Accuracy')
plt.axis([1,11,50,80])
plt.legend(['512','1024','2048','4096','8192'])
plt.subplot(122)
plt.title('True Positive Rate Achieved by Histogram Models')
plt.bar(size_512_x-0.3,size_512_tp,width=0.15,color='r')#,yerr=size_512_tp_err)
plt.bar(size_1024_x-0.15,size_1024_tp,width=0.15,color='y')#,yerr=size_1024_tp_err)
plt.bar(size_2048_x,size_2048_tp,width=0.15,color='g')#,yerr=size_2048_tp_err)
plt.bar(size_4096_x+0.15,size_4096_tp,width=0.15,color='b')#,yerr=size_2048_tp_err)
plt.bar(size_8192_x+0.3,size_8192_tp,width=0.15,color='m')#,yerr=size_2048_tp_err)
plt.xlabel('Number of Layers')
plt.ylabel('TP Rate')
plt.axis([1,11,15,70])
plt.legend(['512','1024','2048','4096','8192'])
plt.savefig('hists_graph.pdf')

