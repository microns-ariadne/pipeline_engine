
import sys
import os
from imread import imread_multi
import numpy as np

def stacked_tiff_to_images(stacked_tiff_filename):
	
	im_arr = imread_multi(stacked_tiff_filename)
	print 'tiff images in stack: %d' % (len(im_arr),)
	
	images = []
	for image in im_arr:
		im = np.array(image) # (x, y)
		images.append(im)
	
	images = np.array(images) # (num,x,y)
	
	return images
