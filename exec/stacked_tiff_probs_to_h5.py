
import sys
import os
import h5py
from imread import imread_multi
import numpy as np
from stacked_tiff_to_images import stacked_tiff_to_images


def adjust_probs(images):
    
    print 'Normalizing pixels from [0..255] to [0..1]'  
    
    scaled_images = []
    for image in images:
        image = image.astype(np.float)
        #image -= image.min()
        image /= image.max()
        scaled_images.append(image)
        
    images = np.array(scaled_images)
        
    print 'Convert: (num,x,y) => (num,x,y,ch)'
    print 'Old shape: %r' % (images.shape,)
        
    images = np.array([images, images]) # (ch,num,x,y)
    images = images.transpose((1,2,3,0)) # (num,x,y,ch)
    
    print 'New shape: %r' % (images.shape,)
    
    return images
    
def execute(stacked_tiff_filename, output_filename):
    
    images = stacked_tiff_to_images(stacked_tiff_filename)
    
    images = adjust_probs(images)
    
    f = h5py.File(output_filename, "w")
    grp = f.create_group("volume")
    dset = grp.create_dataset("predictions", data=images, compression="gzip")
    
    print 'H5 (volume/predictions) dset.shape = %r' % (dset.shape,)
    
    f.close()
    

if '__main__' == __name__:
    try:
        prog_name, stacked_tiff_filename, output_filename = sys.argv[:3]
        
    except ValueError, e:
        sys.exit('USAGE: %s [stacked_tiff_filename] [output_filename] ' % (sys.argv[0],))

    execute(stacked_tiff_filename, output_filename)

