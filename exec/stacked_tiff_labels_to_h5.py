
import sys
import os
import h5py
from imread import imread_multi
import numpy as np
from stacked_tiff_to_images import stacked_tiff_to_images

def adjust_labels(images):
	
	print 'Convert: (num,x,y) => (y,x,num)'
	print 'Old shape: %r' % (images.shape,)
	
	images = images.transpose((2,1,0)) # (num,x,y) => (y,x,num)
	
	print 'New shape: %r' % (images.shape,)
	
	return images
	

def execute(stacked_tiff_filename, output_filename):
	
	images = stacked_tiff_to_images(stacked_tiff_filename)
	
	images = adjust_labels(images)
	
	f = h5py.File(output_filename, "w")
	dset = f.create_dataset("stack", data=images, compression="gzip")
	
	print 'H5 (stack) dset.shape = %r' % (dset.shape,)
	
	f.close()
	

if '__main__' == __name__:
	try:
		prog_name, stacked_tiff_filename, output_filename = sys.argv[:3]
		
	except ValueError, e:
		sys.exit('USAGE: %s [stacked_tiff_filename] [output_filename] ' % (sys.argv[0],))

	execute(stacked_tiff_filename, output_filename)

