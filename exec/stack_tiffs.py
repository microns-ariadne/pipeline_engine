
import sys
import os
from imread import imsave
import numpy as np


def execute(input_dir, output_filename):
    
    image_paths = [os.path.join(input_dir, x) for x in os.listdir(input_dir)]
    image_paths.sort()
    
    input_param = ''
    
    for i, image_path in enumerate(image_paths):
        print '[%d] %s' % (i, image_path)
        input_param += ' "%s"' % (image_path,)
    
    print '==> Sorted %d files' % (len(image_paths),)
    
    tiff_cmd = 'tiffcp -,=# %s %s' % (input_param, output_filename)
    #print tiff_cmd 
    os.system(tiff_cmd)
    
    print '==> tiff created: %s' % (output_filename,)
    

    
if '__main__' == __name__:
    try:
        prog_name, input_dir, output_filename = sys.argv[:3]
        
    except ValueError, e:
        sys.exit('USAGE: %s [input_dir] [output_filename] ' % (sys.argv[0],))


    execute(input_dir, output_filename)

