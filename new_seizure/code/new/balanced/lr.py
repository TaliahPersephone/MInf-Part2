import keras
import numpy as np
import math

def step_decay(epoch):
	initial_lrate = 0.00001
	return initial_lrate/math.sqrt(epoch+1)

	


