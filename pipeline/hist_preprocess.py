'''
' This file contains functions that will extract the HOG/HOF/MBH features from input frames 
'
' The code is adapted from the Matlab code used in the papers:
' - J.R.R. Uijlings, I.C. Duta, E. Sangineto, and N. Sebe 
' "Video Classification with Densely Extracted HOG/HOF/MBH Features: An Evaluation of the Accuracy/Computational Efficiency Trade-off" 
' In International Journal of Multimedia Information Retrieval (IJMIR), 2015.
' - I.C. Duta, J.R.R. Uijlings, T.A. Nguyen, K. Aizawa, A.G. Hauptmann, B. Ionescu, N. Sebe
' "Histograms of Motion Gradients for Real-time Video Classification" 
' In International Workshop on Content-based Multimedia Indexing (CBMI), 2016
'
' As the code is to be used on live videos on a per frame basis, it is slightly altered
' And does not take into account the videos as whole blocks
'''

import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy import einsum
import cv2 as cv
import queue
from scipy.ndimage.filters import convolve as filter2

'''
' This has been hard coded to provide the marix required for the dimensions we are using to save time
'
' Otherwise, it translates a diagonal matrix to a rectangular one, such that all the rows still add to one
' and the columns add to 1/shape[1]
'''
def diag_matrix_linear():
	matrix = np.load('diag_matrix.npy')

	return matrix


'''
' Strides the data, making block as in the paper
' Input:
'	data - data to make into a block
'	A - rectangular diagonal
'	B - rectangular diagonal
' Output:
'	block - the data strided into blocks
'
'''
def make_block(data, A, B):
	block = einsum('ab,bef,eh->fah',A,data,B)

	block = block.T.reshape(64,8)
		
	inds = np.arange(64).reshape(8,8)
	inds = as_strided(inds,(2,2,7,7),(64,8,8,64))
	inds = inds.transpose(0,1,3,2).reshape(-1)
	
	block = block[inds]

	block = block.T.reshape(32,-1).T.reshape(-1)
	 
	return block



'''
' Calculate haar gradients of an image
' Input:
' 	im - image
' Output:
'	gr - haar gradients
'''
def haar_gradients(im):
	ogIm = np.zeros((im.shape[0],im.shape[1],8))

	rowIm = im[:-2, 1:-1] - im[2:,1:-1]
	colIm = im[1:-1,:-2] - im[1:-1,2:]

	magIm = np.sqrt(rowIm * rowIm + colIm * colIm)
	angleIm = np.arctan2(rowIm, colIm)

	binReal = (angleIm) * (8 / (2 * np.pi))
	binLow = np.floor(binReal)
	weightHigh = binReal - binLow
	weightLow = 1 - weightHigh
	binLow = (binLow % 8).astype(int)
	binHigh = (binLow + 1).astype(int)
	binHigh[binHigh == 8] = 0

	[colI, rowI] = np.meshgrid(range(1,im.shape[1]-1), range(1,im.shape[1]-1))
	ogIm[(rowI,colI,binLow)] += magIm * weightLow

	ogIm[(rowI,colI,binHigh)] += magIm * weightHigh


	return ogIm


'''
' Calculate histogram of gradients, repeatedly from a queue
' Input:
'	i - input queue
' Output:
'	o - output queue
'''
def HOG(i, o):
	arrayA = diag_matrix_linear()
	arrayB = arrayA.T
	if True:
		frame = i

		ogIm = haar_gradients(frame)

		hogs = make_block(ogIm,arrayA,arrayB)

		o.put(hogs)


'''
' Calculate horn-schunck optical flow between two images
' Input:
' 	im1 - image
' Output:
'	of - horn-schunck optical flow
'''
def horn_schunck(im1,im2,Niter=10):
	 #set up initial velocities
	U = np.zeros([im1.shape[0],im1.shape[1]])
	V = np.zeros([im1.shape[0],im1.shape[1]])
	
	sobelX = np.array([[-1,-2,-1],[0,0,0],[-1,-2,-1]])
	sobelY = sobelX.T

	Ix = cv.filter2D(im1,-1,sobelX)
	Iy = cv.filter2D(im1,-1,sobelY)

	It = im2 - im1

	avgK = np.array([[0,1,0],[1,0,1],[0,1,0]])


	for i in range(Niter):
		uAvg = cv.filter2D(U,-1,avgK)
		vAvg = cv.filter2D(V,-1,avgK)

		der = (Ix*uAvg + Iy*vAvg + It) / (1 + Ix**2 + Iy**2)

		U = uAvg - Ix * der
		V = vAvg - Iy * der	

	return U, V


def HS(im1, im2, alpha=1, Niter=10):
	"""
	im1: image at t=0
	im2: image at t=1
	alpha: regularization constant
	Niter: number of iteration
	"""
	
	#set up initial velocities
	uInitial = np.zeros([im1.shape[0],im1.shape[1]])
	vInitial = np.zeros([im1.shape[0],im1.shape[1]])
	
	# Set initial value for the flow vectors
	U = uInitial
	V = vInitial
	
	# Estimate derivatives
	[fx, fy, ft] = computeDerivatives(im1, im2)
	
	
	# Averaging kernel
	kernel=np.array([[1/12, 1/6, 1/12],
	                  [1/6,    0, 1/6],
	                  [1/12, 1/6, 1/12]],float)
	
	
	# Iteration to reduce error
	for _ in range(Niter):
	#Compute local averages of the flow vectors
		uAvg = filter2(U,kernel)
		vAvg = filter2(V,kernel)
	#common part of update step
		der = (fx*uAvg + fy*vAvg + ft) / (alpha**2 + fx**2 + fy**2)
	#iterative step
		U = uAvg - fx * der
		V = vAvg - fy * der
	
	return U,V

def computeDerivatives(im1, im2):
#%% build kernels for calculating derivatives
	kernelX = np.array([[-1, 1],[-1, 1]]) * .25 #kernel for computing d/dx
	kernelY = np.array([[-1,-1],[ 1, 1]]) * .25 #kernel for computing d/dy
	kernelT = np.ones((2,2))*.25

	fx = filter2(im1,kernelX) + filter2(im2,kernelX)
	fy = filter2(im1,kernelY) + filter2(im2,kernelY)

	#ft = im2 - im1
	ft = filter2(im1,kernelT) + filter2(im2,-kernelT)

	return fx,fy,ft

'''
' Calculate histogram of optical flow, repeatedly from a queue
' Input:
'	i - input queue
' Output:
'	o - output queue
'''
def HOF(i, o):
	arrayA = diag_matrix_linear()
	arrayB = arrayA.T
	prev_frame = None
	if True:
		n,frame = i.get()

		if prev_frame is None:
			prev_frame = frame
			m = n
			n,frame = i.get()

		v1, v2 = HS(prev_frame,frame)

		mag = np.sqrt(v1**2 + v2**2)
		ang = np.arctan2(v1,v2)

		oM = np.zeros((mag.shape[0],mag.shape[1],8))

		binReal = (ang) * (8 / (2 * np.pi))
		binLow = np.floor(binReal)
		weightHigh = binReal - binLow
		weightLow = 1 - weightHigh
		binLow = (binLow % 8).astype(int)
		binHigh = (binLow + 1).astype(int)
		binHigh[binHigh == 8] = 0
	
		[colI, rowI] = np.meshgrid(range(mag.shape[1]), range(mag.shape[0]))
		oM[(rowI,colI,binLow)] += mag * weightLow
	
		oM[(rowI,colI,binHigh)] += mag * weightHigh

		hof = make_block(oM,arrayA,arrayB)
	
		o.put((m,hof))	

		

def MBHr(i, o):
	arrayA = diag_matrix_linear()
	arrayB = arrayA.T
	prev_frame = None
	if True:
		n,frame = i.get()

		if prev_frame is None:
			prev_frame = frame
			m = n
			n,frame = i.get()

		v1, _ = HS(prev_frame,frame)

		ogIm = haar_gradients(v1)

		hogs = make_block(ogIm,arrayA,arrayB)

		o.put((m,hogs))
		
	
def MBHc(i, o):
	arrayA = diag_matrix_linear()
	arrayB = arrayA.T
	prev_frame = None
	if True:
		n,frame = i.get()

		if prev_frame is None:
			prev_frame = frame
			m = n
			n,frame = i.get()

		_, v2 = HS(prev_frame,frame)
	
		ogIm = haar_gradients(v2)

		hogs = make_block(ogIm,arrayA,arrayB)

		o.put((m,hogs))
