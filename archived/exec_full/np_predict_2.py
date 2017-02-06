
import sys
import os
import time

def compress_h5_stack(h5_filepath):
    h5_filepath_gzip = h5_filepath + '_gzip'
    
    exec_cmd('python compress_h5_stack.py %s %s' % (h5_filepath, h5_filepath_gzip))
    os.remove(h5_filepath)
    os.rename(h5_filepath_gzip, h5_filepath)

def exec_cmd(cmd):
    print 'EXEC START: %s' % (cmd,)

    start_time_secs = time.time()
    
    os.system(cmd)
    
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'EXEC FINISH: Execution time is %s seconds' % (elapsed_time_secs,)    

def execute(np_path, np_classifier_filepath, block_path_np, block_name):
        
    np_probs_h5_filename = os.path.join(block_path_np, '%s_np_probs.h5' % (block_name,))
    np_ws_h5_filename = os.path.join(block_path_np, '%s_np_ws.h5' % (block_name,))
    np_seg_prefix_filename = os.path.join(block_path_np, '%s_np_seg_' % (block_name,))
    
    print ' -- Execute NP predict'
    exec_cmd('%s %s %s %s --output-file=%s' % 
        (np_path,
         np_ws_h5_filename, # (y,x,num) 
         np_probs_h5_filename, # (num,x,y,2)
         np_classifier_filepath,
         np_seg_prefix_filename))
    
    print ' -- Compress h5 output file (stack)'
    compress_h5_stack(np_seg_h5_filename)
        

if '__main__' == __name__:
    try:
        prog_name, np_path, np_classifier_filepath, block_path_np, block_name = sys.argv[:5]
        
    except ValueError, e:
        sys.exit('USAGE: %s [np_path] [np_classifier_filepath] [block_path_np] [block_name] ' % (sys.argv[0],))


    execute(np_path, np_classifier_filepath, block_path_np, block_name)

