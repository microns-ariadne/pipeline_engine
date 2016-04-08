
import sys
import os
import h5py
import numpy as np
from PIL import Image

def np_adjust_probs(images):
    
    print 'Normalizing pixels from [0..255] to [0..1]'  
    
    scaled_images = []
    for i, image in enumerate(images):
        print ' -- Scale image %d' % (i,)
        image = image.astype(np.float)
        #image -= image.min()
        image /= image.max()
        scaled_images.append(image)
        
    images = scaled_images
    
    n_rows = images[0].shape[0]
    n_cols = images[0].shape[1]
    n_images = len(images)
    
    #volume = np.zeros((n_images, n_rows, n_cols, 2), dtype = np.float)
    volume = np.zeros((n_images, n_rows, n_cols), dtype = np.float)
    
    print 'Generate NP shape: %r' % (volume.shape,)
    
    print ' -- start'
    
    for i, image in enumerate(images):
        print ' -- Set image %d' % (i,)
        #volume[i,:,:,0] = image
        #volume[i,:,:,1] = image
        volume[i,:,:] = image
    
    print ' -- done'
    
    return volume
    

def execute(probs_dir, output_filename):
    
    image_paths = [os.path.join(probs_dir, x) for x in os.listdir(probs_dir)]
    image_paths.sort()
    
    images = []
    for i, image_path in enumerate(image_paths):
        print ' -- Read [%d]: %s' % (i, image_path)
        im = np.array(Image.open(image_path))
        images.append(im)
    
    images = np_adjust_probs(images)
    
    f = h5py.File(output_filename, "w")
    grp = f.create_group("volume")
    dset = grp.create_dataset("predictions", data=images, compression="gzip")
    
    print 'H5 (volume/predictions) dset.shape = %r' % (dset.shape,)
    
    f.close()
    

if '__main__' == __name__:
    try:
        prog_name, probs_dir, output_filename = sys.argv[:3]
        
    except ValueError, e:
        sys.exit('USAGE: %s [probs_dir] [output_filename] ' % (sys.argv[0],))

    execute(probs_dir, output_filename)

