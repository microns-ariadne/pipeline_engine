
import os
import sys
import time

import cv2
import numpy as np

from pipeline_common import *

def execute():
    
    merge_dirs = []
    for data_dir in DATA_DIR_LIST:
        merge_dirs.append(os.path.join(data_dir, MERGE_DIR))
    
    maxID_filepaths = []
    for merge_dir in merge_dirs:
        maxID_filepaths += [os.path.join(merge_dir, x) 
                           for x in os.listdir(merge_dir) if x.find('_maxID.txt') != -1]
    
    maxID_filepaths.sort()
    
    print '--------------------------------------------------------------------'
    print ' -- maxID_filepaths: found %d filepaths' % (len(maxID_filepaths),)
    print '--------------------------------------------------------------------'
    print ' -- Start fixing'
    print '--------------------------------------------------------------------'
    
    t_maxID = 0
    maxIDs_fixed = []
    
    for i, maxID_filepath in enumerate(maxID_filepaths):
        
        print ' -- [%d] Read: %s' % (i, maxID_filepath)
        
        f = open(maxID_filepath, 'rb')
        cur_maxID = int(f.readline())
        f.close()
        
        print '   -- cur_maxID: %d' % (cur_maxID,)
        
        out_dir = os.path.dirname(maxID_filepath)
        out_name = os.path.basename(maxID_filepath).split('.txt')[0] + '_fixed.txt'
        
        out_path = os.path.join(out_dir, out_name)
        
        print '   -- Write[t_maxID = %d]: %s' % (t_maxID, out_path,)
        
        f = open(out_path, 'wb')
        f.write('%d\n' % (t_maxID,))
        f.close()
         
        t_maxID += (cur_maxID + 1)
    
        
if '__main__' == __name__:
    try:
        (prog_name,) = sys.argv[:4]
                
    except ValueError, e:
        sys.exit('USAGE: %s' % (sys.argv[0],))
    
    start_time_secs = time.time()
    
    execute()
    
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'ELAPSED TIME: %s seconds' % (elapsed_time_secs,)
    
    print PROC_SUCCESS_STR
