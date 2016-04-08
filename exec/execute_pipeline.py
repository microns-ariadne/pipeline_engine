
import sys
import os
import argparse
import shutil
import time
from imread import imread,imsave
import numpy as np

IS_PROBS_FIX = 1
IS_NEW_NP = 1

NP_THRESHOLD = 0.2

WS_GALA = 'ws-gala'
WS_NEW = 'ws-new'

WS_ALG = WS_NEW

WATERSHED_SEED_SIZE = 5

WATERSHED_SEED_VAL = 0.0

RESULTS_DIR = None

TIFF_STACK_WS_PROBS_FILEPATH = None
TIFF_STACK_NP_PROBS_FILEPATH = None
TIFF_STACK_LABELS_FILEPATH = None

H5_WS_PROBS_FILEPATH = None
H5_NP_PROBS_FILEPATH = None
H5_LABELS_FILEPATH = None

H5_WS_FILEPATH = None

WS_TMP_DIR = None

NP_CLASSIFIER_FILEPATH = None
H5_NP_FILEPATH = None

ARGS = None

IS_ASK = 0

def verify_dirs_match(dir1, dir2):
    print 'verify_dirs_match:'
    print ' -- dir1: %s' % (dir1,)
    print ' -- dir2: %s' % (dir2,)
    
    filenames_1 = os.listdir(dir1)
    filenames_1.sort()
    filenames_2 = os.listdir(dir2)
    filenames_2.sort()
    
    n_files_1 = len(filenames_1)
    n_files_2 = len(filenames_2)
    if (n_files_1 != n_files_2):
        raise Exception('ERROR: not the same number of files [dir1 = %d] [dir2 = %d]' 
                         % (n_files_1, n_files_2))
    
    print '================================================'
    for i in xrange(n_files_1):
        print '[%d] %s <=> %s' % (i+1, filenames_1[i], filenames_2[i])
    print '================================================'
    
    if IS_ASK:
        in_res = raw_input('Is matching ok ? [YES/NO] ')
    
        if in_res != 'YES':
            raise Exception('ERROR: match not confirmed')
    
def verify_and_get_np_classifier(np_classifier_dir):
    print 'verify_and_get_np_classifier:'
    print ' -- np_classifier_dir: %s' % (np_classifier_dir,)
    
    filenames = os.listdir(np_classifier_dir)
    
    if len(filenames) != 2:
        raise Exception('ERROR: Only 2 files are expected for neuroproof classifier (xml and txt) [got: %d]' %
                        (len(filenames),))
    
    filename_1 = filenames[0]
    filename_2 = filenames[1]
    
    if filename_1.find('_ignore.txt') != -1:
        np_txt_filename = filename_1
    elif filename_2.find('ignore.txt') != -1:
        np_txt_filename = filename_2
    else:
        raise Exception('ERROR: *_ignore.txt neuroproof file was not found')

    if filename_1.find('.xml') != -1:
        np_xml_filename = filename_1
    elif filename_2.find('.xml') != -1:
        np_xml_filename = filename_2
    else:
        raise Exception('ERROR: *.xml neuroproof file was not found')
    
    np_xml_filepath = os.path.join(np_classifier_dir, np_xml_filename)
    
    print ' -- np_classifier_file: %s' % (np_xml_filename,)
    return np_xml_filepath
    

def create_dirs():
    if ARGS.is_overwrite:
        print 'Removing %s' % (RESULTS_DIR,)
        shutil.rmtree(RESULTS_DIR)

    if not os.path.exists(RESULTS_DIR):
        print 'Creating %s' % (RESULTS_DIR,)
        os.makedirs(RESULTS_DIR)
    else:
        print 'Directory %s already exists' % (RESULTS_DIR,)
        print 'Skipping'
        
    if not os.path.exists(WS_TMP_DIR):
        print 'Creating %s' % (WS_TMP_DIR,)
        os.makedirs(WS_TMP_DIR)
    else:
        print 'Clean directory %s' % (WS_TMP_DIR,)
        filepaths = [os.path.join(WS_TMP_DIR, filename) for filename in os.listdir(WS_TMP_DIR)]
        for filepath in filepaths:
            os.remove(filepath)
        print ' -- Removed %d files' % (len(filepaths),)
    
def adjust_probs(probs_dir):
    
    if not IS_PROBS_FIX:
        return
    
    probs_filepaths = [os.path.join(probs_dir, filename) for filename in os.listdir(probs_dir)]
    probs_filepaths.sort()
    
    for prob_filepath in probs_filepaths:
        parts = prob_filepath.split('.tif')
        
        if parts[0].find('__pipeline_fixed__') != -1:
            continue
        
        im = imread(prob_filepath)
        
        im = im.astype(np.float)
        
        if im.min() != 255:
            im -= im.min()
        im /= im.max()
        im *= 255
        
        im = im.astype(np.uint8)
        
        output_filepath = parts[0] + '__pipeline_fixed__.tif'
        imsave(output_filepath, im)
        os.remove(prob_filepath)
        print 'adjust_probs: generated %s' % (output_filepath,)
        
    
def exec_cmd(cmd):
    print 'EXEC START: %s' % (cmd,)

    start_time_secs = time.time()
    
    os.system(cmd)
    
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'EXEC FINISH: Execution time is %s seconds' % (elapsed_time_secs,)    
    
def compress_h5_stack(h5_filepath):
    h5_filepath_gzip = h5_filepath + '_gzip'
    
    exec_cmd('python compress_h5_stack.py %s %s' % (h5_filepath, h5_filepath_gzip))
    os.remove(h5_filepath)
    os.rename(h5_filepath_gzip, h5_filepath)
    

def tifs_to_stack(input_dir, output_filepath):
    if os.path.exists(output_filepath):
        print 'tifs_to_stack: already exists: %s' % (output_filepath,) 
        print '  -- skipping'
        return
    
    print 'tifs_to_stack:'
    print '  -- input : %s' % (input_dir,)
    print '  -- output: %s' % (output_filepath,)
    
    exec_cmd('python stack_tiffs.py %s %s' % (input_dir, output_filepath))
    

def tif_stack_to_h5(script_filename, input_filepath, output_filepath):
    if os.path.exists(output_filepath):
        print 'tif_stack_to_h5: already exists: %s' % (output_filepath,) 
        print '  -- skipping'
        return
    
    print 'tif_stack_to_h5:'
    print '  -- script : %s' % (script_filename,)
    print '  -- input  : %s' % (input_filepath,)
    print '  -- output : %s' % (output_filepath,)
    
    exec_cmd('python %s %s %s' % (script_filename, input_filepath, output_filepath))
    

def process_input_dir(
    input_dir, 
    script_filename,
    tif_stack_filepath, 
    h5_filepath):
    
    tifs_to_stack(input_dir, tif_stack_filepath)
        
    tif_stack_to_h5(script_filename, tif_stack_filepath, h5_filepath)
    

def execute_gala_ws(h5_probs_filepath, h5_output_filepath):
        
    if os.path.exists(h5_output_filepath):
        print 'execute_gala_ws: already exists: %s' % (h5_output_filepath,) 
        print '  -- skipping'
        return
        
    exec_cmd('python ./gala_watershed/gala_overseg.py \
             --pixelprob-file=%s \
             --seed-size=%d \
             --seed-val=%f \
             --supervoxels-name=%s \
             . ' % (h5_probs_filepath, WATERSHED_SEED_SIZE, WATERSHED_SEED_VAL, h5_output_filepath))
                
    print ' -- Compress output h5 file (stack)'
    compress_h5_stack(h5_output_filepath)
 
def execute_ws(ws_probs_dir, h5_output_filepath):

    if os.path.exists(h5_output_filepath):
        print 'execute_ws: already exists: %s' % (h5_output_filepath,) 
        print '  -- skipping'
        return
    
    exec_cmd('./watershed/cpp/build/watershed.x %s %s' % 
             (ws_probs_dir, WS_TMP_DIR))
    
    exec_cmd('python ws_to_h5.py %s %s' % 
             (WS_TMP_DIR, h5_output_filepath))
    
    print ' -- Compress output h5 file (stack)'
    compress_h5_stack(h5_output_filepath)
    

def execute_np_learn(
    h5_ws_filepath,
    h5_np_probs_filepath,
    h5_labels_filepath,
    np_classifier_filepath):
    
    if os.path.exists(np_classifier_filepath):
        print 'execute_np_learn: already exists: %s' % (np_classifier_filepath,) 
        print '  -- skipping'
        return
    
    exec_cmd('./neuroproof_agg/npclean/build/neuroproof_graph_learn_old_4d %s %s %s --classifier-name=%s' % 
             (h5_ws_filepath, # (y,x,num) 
              h5_np_probs_filepath, # (num,x,y,2)
              h5_labels_filepath, # (y,x,num)
              np_classifier_filepath))
     

def execute_np_predict(
    h5_ws_filepath,
    np_probs_dir,
    h5_np_probs_filepath,
    h5_labels_filepath,
    is_learn_np_classifier, 
    np_classifier_filepath, 
    h5_output_filepath):
    
    if os.path.exists(h5_output_filepath):
        print 'execute_np_predict: already exists: %s' % (h5_output_filepath,) 
        print '  -- skipping'
        return
    
    if is_learn_np_classifier:
        print 'execute_np: learn classifier'
        execute_np_learn(
            h5_ws_filepath,
            h5_np_probs_filepath,
            h5_labels_filepath,
            np_classifier_filepath)
    
    print 'execute_np: execute prediction'
    
    if IS_NEW_NP:
        h5_np_probs_filepath_new = h5_np_probs_filepath.split('.h5')[0] + '_new.h5'
        
        if not os.path.exists(h5_np_probs_filepath_new):
            exec_cmd('python ./np_probs_to_h5.py %s %s' % (np_probs_dir, h5_np_probs_filepath_new))
        
        exec_cmd('./neuroproof_agg/npclean/build/neuroproof_graph_predict %s %s %s --output-file=%s --threshold=%.2f' % 
                (h5_ws_filepath, # (y,x,num) 
                h5_np_probs_filepath_new, # (num,x,y)
                np_classifier_filepath,
                h5_output_filepath,
                NP_THRESHOLD))
            
    else:        
        exec_cmd('./neuroproof_agg/npclean/build/neuroproof_graph_predict_old_4d %s %s %s --output-file=%s --threshold=%.2f' % 
                (h5_ws_filepath, # (y,x,num) 
                h5_np_probs_filepath, # (num,x,y,2)
                np_classifier_filepath,
                h5_output_filepath,
                NP_THRESHOLD))
        
    print '    - Compress output h5 file (stack)'
    compress_h5_stack(h5_output_filepath)
        

def execute():
    
    labels_dir = ARGS.labels_dir
    ws_probs_dir = ARGS.ws_probs_dir
    np_probs_dir = ARGS.np_probs_dir
    np_classifier_dir = ARGS.np_classifier_dir
    
    print '=========================================================='
    print '\n *** Setup directories'
    print '=========================================================='
    
    create_dirs()
    
    if ws_probs_dir != np_probs_dir:
        print '=========================================================='
        print '\n *** Verify ws_probs_dir and np_probs_dir match'
        print '=========================================================='
        verify_dirs_match(ws_probs_dir, np_probs_dir)
    
    is_learn_np_classifier = False
    np_classifier_filepath = None
    
    if np_classifier_dir == '':
        if labels_dir == None:
            raise Exception('Labels dir is required when no classifier is supplied for neuroproof')
        
        is_learn_np_classifier = True
        np_classifier_filepath = NP_CLASSIFIER_FILEPATH
        
        if not os.path.exists(H5_LABELS_FILEPATH):
            print '=========================================================='
            print '\n *** Verify labels_dir and ws_probs_dir match'
            print '=========================================================='
            verify_dirs_match(labels_dir, ws_probs_dir)
        
        print '=========================================================='
        print '\n *** Process labels_dir'
        print '=========================================================='
        
        process_input_dir(
            labels_dir,
            'stacked_tiff_labels_to_h5.py',
            TIFF_STACK_LABELS_FILEPATH,
            H5_LABELS_FILEPATH)    
        
    else:
        print '=========================================================='
        print '\n *** Verify and get NP classifier'
        print '=========================================================='
        np_classifier_filepath = verify_and_get_np_classifier(np_classifier_dir)
    
    print '=========================================================='
    print '\n *** Process ws_probs_dir'
    print '=========================================================='
    adjust_probs(ws_probs_dir)
    process_input_dir(
        ws_probs_dir,
        'stacked_tiff_probs_to_h5.py',
        TIFF_STACK_WS_PROBS_FILEPATH,
        H5_WS_PROBS_FILEPATH)

    if np_probs_dir != ws_probs_dir:
        print '=========================================================='
        print '\n *** Process np_probs_dir'
        print '=========================================================='
        adjust_probs(np_probs_dir)
        process_input_dir(
            np_probs_dir,
            'stacked_tiff_probs_to_h5.py',
            TIFF_STACK_NP_PROBS_FILEPATH,
            H5_NP_PROBS_FILEPATH)
    else:
        H5_NP_PROBS_FILEPATH = H5_WS_PROBS_FILEPATH
        
    print '=========================================================='
    print '\n *** Execute watershed'
    print '=========================================================='
    if WS_ALG == WS_GALA:
        execute_gala_ws(H5_WS_PROBS_FILEPATH, H5_WS_FILEPATH)
    elif WS_ALG == WS_NEW:
        execute_ws(ws_probs_dir, H5_WS_FILEPATH)
    else:
        raise Exception('ERROR: WS_ALG is not set')
    
    print '=========================================================='
    print '\n *** Execute agglomeration'
    print '=========================================================='
    execute_np_predict(
        H5_WS_FILEPATH,
        np_probs_dir,
        H5_NP_PROBS_FILEPATH,
        H5_LABELS_FILEPATH,
        is_learn_np_classifier, 
        np_classifier_filepath, 
        H5_NP_FILEPATH)
    
    
if '__main__' == __name__:
    
    parser = argparse.ArgumentParser(description='Execute pipeline.')
    
    parser.add_argument(
        '--overwrite',
        dest = 'is_overwrite',
        action = 'store_true',
        help = 'Overwrite current results')
    
    parser.add_argument(
        '--run-name', 
        dest = 'run_name', 
        type = str, 
        required = True, 
        help = 'String name of the run. Added to each filename as postfix.')
    
    parser.add_argument(
        '--labels-dir', 
        dest = 'labels_dir', 
        type = str, 
        required = True, 
        help = 'Directory for labels (groundtruth) TIFF files')
    
    parser.add_argument(
        '--ws-probs-dir', 
        dest = 'ws_probs_dir', 
        type = str, 
        required = True, 
        help = 'Directory for membrane probability TIFF files (for watershed)')
    
    parser.add_argument(
        '--np-probs-dir', 
        dest = 'np_probs_dir', 
        type = str, 
        required = True, 
        help = 'Directory for membrane probability TIFF files (for neuroproof)')
    
    parser.add_argument(
        '--np-classifier-dir', 
        dest = 'np_classifier_dir', 
        type = str, 
        required = True, 
        help = 'Directory for neuroproof classifier')
    
    ARGS = parser.parse_args()
    
    RESULTS_DIR = './results-%s/' % (ARGS.run_name,)

    _TIFF_STACK_WS_PROBS_FILENAME = 'stacked-ws-probs-%s.tif' % (ARGS.run_name,)
    _TIFF_STACK_NP_PROBS_FILENAME = 'stacked-np-probs-%s.tif' % (ARGS.run_name,)
    _TIFF_STACK_LABELS_FILENAME = 'stacked-labels-%s.tif' % (ARGS.run_name,)
    
    TIFF_STACK_WS_PROBS_FILEPATH = os.path.join(RESULTS_DIR, _TIFF_STACK_WS_PROBS_FILENAME)
    TIFF_STACK_NP_PROBS_FILEPATH = os.path.join(RESULTS_DIR, _TIFF_STACK_NP_PROBS_FILENAME)
    TIFF_STACK_LABELS_FILEPATH = os.path.join(RESULTS_DIR, _TIFF_STACK_LABELS_FILENAME)
    
    _H5_WS_PROBS_FILENAME = 'h5-ws-probs-%s.h5' % (ARGS.run_name,)
    _H5_NP_PROBS_FILENAME = 'h5-np-probs-%s.h5' % (ARGS.run_name,)
    _H5_LABELS_FILENAME = 'h5-labels-%s.h5' % (ARGS.run_name,)
    
    H5_WS_PROBS_FILEPATH = os.path.join(RESULTS_DIR, _H5_WS_PROBS_FILENAME)
    H5_NP_PROBS_FILEPATH = os.path.join(RESULTS_DIR, _H5_NP_PROBS_FILENAME)
    H5_LABELS_FILEPATH = os.path.join(RESULTS_DIR, _H5_LABELS_FILENAME)
    
    _H5_WS_FILENAME = '%s-supervoxels-%s.h5' % (WS_ALG, ARGS.run_name,)
    H5_WS_FILEPATH = os.path.join(RESULTS_DIR, _H5_WS_FILENAME)
    
    _H5_NP_FILENAME = 'np-segmentation-%s.h5' % (ARGS.run_name,)
    H5_NP_FILEPATH = os.path.join(RESULTS_DIR, _H5_NP_FILENAME)
    
    _NP_CLASSIFIER_FILENAME = 'np-classifier-%s.xml' % (ARGS.run_name,)
    NP_CLASSIFIER_FILEPATH = os.path.join(RESULTS_DIR, _NP_CLASSIFIER_FILENAME)
    
    WS_TMP_DIR = os.path.join(RESULTS_DIR, 'ws_tmp_dir')
    
    if (ARGS.is_overwrite):
        print 'Overwrite mode: ON'
    else:
        print 'Overwrite mode: OFF'
    
    execute()

