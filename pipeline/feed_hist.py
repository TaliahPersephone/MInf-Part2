import scipy
import tables
import numpy as np


def feed_hist(i,hists,o):
	hist = np.load(hists)

	while True:
		frame = i.get()

		for q in o:
			q.put(hist[frame,:])
