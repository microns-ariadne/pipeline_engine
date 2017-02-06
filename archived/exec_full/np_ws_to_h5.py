
import sys
import os
import h5py
import numpy as np
from imread import imread
import cv2

def execute(input_dir, output_filename):
    
    image_paths = [os.path.join(input_dir, x) for x in os.listdir(input_dir)]
    
    image_paths.sort()
    
    ws_volume = None
    for im_id, image_path in enumerate(image_paths):
             
        im = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        print 'Processing image %d' % (im_id,)
        
        assert(im.dtype == np.uint8) 
        assert(im.shape[2] == 4)
        
        if ws_volume == None:
            ws_volume = np.zeros((im.shape[1], im.shape[0], len(image_paths)), dtype=np.uint32)
            print 'Create WS volume of shape = %r' % (ws_volume.shape,)
            
        im = im.astype(np.uint32)
        
        #im[:,:,0] = (255-im[:,:,0]) * (1<<24)
        im[:,:,2] *= (1<<16)
        im[:,:,1] *= (1<<8)
        im[:,:,0] = im[:,:,0] + im[:,:,1] + im[:,:,2]# + im[:,:,0]
        
        ws_volume[:,:,im_id] = im[:,:,0].transpose((1,0))
    
    print 'ws_volume.unique: %r' % (np.unique(ws_volume),)
    
    f = h5py.File(output_filename, "w")
    f.create_dataset("stack", data = ws_volume, compression="gzip")
    f.close()
    
    print 'Generated: %s' % (output_filename,)
    

if '__main__' == __name__:
    try:
        prog_name, input_dir, output_filename = sys.argv[:3]
        
    except ValueError, e:
        sys.exit('USAGE: %s [input_dir] [output_filename] ' % (sys.argv[0],))


    execute(input_dir, output_filename)

