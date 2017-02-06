
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

def execute(is_overwrite, 
            block_path_probs, 
            block_path_ws, 
            block_path_np,
            block_name):
    
    np_probs_h5_filename = os.path.join(block_path_np, '%s_np_probs.h5' % (block_name,))
    
    if (os.path.exists(np_probs_h5_filename)) and (not is_overwrite):
        print 'np_probs_to_h5: already exists: %s' % (np_probs_h5_filename,) 
        print '  -- skipping'
    else:
        exec_cmd('python np_probs_to_h5.py %s %s' % 
            (block_path_probs, np_probs_h5_filename))
    
    np_ws_h5_filename = os.path.join(block_path_np, '%s_np_ws.h5' % (block_name,))
    
    if (os.path.exists(np_ws_h5_filename)) and (not is_overwrite):
        print 'np_ws_to_h5: already exists: %s' % (np_ws_h5_filename,) 
        print '  -- skipping'
    else:
        exec_cmd('python np_ws_to_h5.py %s %s' % 
            (block_path_ws, np_ws_h5_filename))
        

if '__main__' == __name__:
    try:
        prog_name, is_overwrite, block_path_probs, block_path_ws, block_path_np, block_name = sys.argv[:6]
        
        is_overwrite = int(is_overwrite)
        
    except ValueError, e:
        sys.exit('USAGE: %s [is_overwrite] [block_path_probs] [block_path_ws] [block_path_np] [block_name]' % (sys.argv[0],))


    execute(is_overwrite,
            block_path_probs, 
            block_path_ws, 
            block_path_np,
            block_name)

