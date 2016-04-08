#!/usr/bin/env python


import sys
import os
import os.path
import numpy
import shutil
import logging
import json
from skimage import morphology as skmorph
from scipy.ndimage import label
import traceback
import glob
import re
import h5py
import hashlib

IS_3D_WS = 1

DEPTH_2D_3D_WS = 1

# removed agglo
from . import imio, morpho, classify, app_logger, \
    session_manager, pixel

try:
    from gala import stack_np
except ImportError:
    np_installed = False   
else:
    np_installed = True

try:
    import syngeo
except ImportError:
    logging.warning('Could not import syngeo. ' +
                                        'Synapse-aware mode not available.')

# Group where we store predictions in HDF5 file
PREDICTIONS_HDF5_GROUP = '/volume/predictions'


def grab_boundary(prediction, channels, master_logger):
    boundary = None
    master_logger.debug("Grabbing boundary labels: " + str(channels))
    for channel_id in channels:
        if boundary is None:
            boundary = prediction[...,channel_id] 
        else:
            boundary += prediction[...,channel_id]

    return boundary


def gen_supervoxels(options, prediction_file, master_logger):
    master_logger.debug("Generating supervoxels")
    if not os.path.isfile(prediction_file):
        raise Exception("Training file not found: " + prediction_file)

    prediction = imio.read_image_stack(prediction_file, group=PREDICTIONS_HDF5_GROUP)
    master_logger.info("Prediction shape: %s" % str(prediction.shape))
    
    prediction = prediction.transpose((2, 1, 0, 3))
    master_logger.info("Prediction shape transposed: %r" % (prediction.shape,))
        
    boundary = prediction[...,0] 
    
    master_logger.info("Boundary shape: %s" % str(boundary.shape))
    
    master_logger.info("Watershed seed threshold: " + str(options.seed_val))
    
    print 'boundary.shape = %r' % (boundary.shape,)
    
    n_images = boundary.shape[2]
    
    assert (n_images % DEPTH_2D_3D_WS == 0)
    
    n_ws_steps = n_images / DEPTH_2D_3D_WS
    
    seeds_list = []
    num_seeds_list = []
    shift = 0
    if IS_3D_WS:
        print 'Zero locations: %d' % ((boundary <= options.seed_val).sum(),)
        
        seeds, num_seeds = label(boundary <= options.seed_val)
        seeds_list = seeds
        num_seeds_list = num_seeds
    else:    
        for i in xrange(n_ws_steps):
            start_idx = i * DEPTH_2D_3D_WS
            end_idx = start_idx + DEPTH_2D_3D_WS
            seeds, num_seeds = label(boundary[:,:,start_idx:end_idx]<=options.seed_val)
            mask = (seeds != 0).astype(numpy.uint32) * shift
            seeds += mask
            shift += 1000
            seeds_list.append(seeds)
            num_seeds_list.append(num_seeds)
            
            print '[%d] num_seeds %d for shift %d [seeds.shape=%r]' % (i, num_seeds, shift, seeds.shape)
    
        print "2D SEEDS SHAPE: %r" % (seeds_list[0].shape,)
        
    if IS_3D_WS:
        print "TOTAL NUMBER OF 3D SEEDS: %d" % (num_seeds_list,)
    else:
        print "TOTAL NUMBER OF SEEDS: %d" % (sum(num_seeds_list),)
    
    if options.seed_size > 0:
        master_logger.debug("Removing small seeds")
        if IS_3D_WS:
            seeds_list = morpho.remove_small_connected_components(seeds_list, options.seed_size)
        else:
            for i in xrange(n_ws_steps):
                seeds_list[i] = morpho.remove_small_connected_components(seeds_list[i], options.seed_size)
            
        master_logger.debug("Finished removing small seeds")
    
    master_logger.info("Starting watershed")

    boundary_cropped = boundary
    seeds_cropped = seeds_list
    #if options.border_size > 0:
    #    boundary_cropped = boundary[options.border_size:(-1*options.border_size), options.border_size:(-1*options.border_size),options.border_size:(-1*options.border_size)]
    #    seeds_cropped = label(boundary_cropped<=options.seed_val)[0]
    #    if options.seed_size > 0:
    #        seeds_cropped = morpho.remove_small_connected_components(seeds_cropped, options.seed_size)
    
    # Returns a matrix labeled using seeded watershed
    watershed_mask = numpy.ones(boundary_cropped.shape).astype(numpy.uint8)

    # Used to specify region to ignore
    masked_bboxes = []

    if options.mask_file is not None:
        mask_file = open(options.mask_file)
        for line in mask_file:
            br = line.split()
            if len(br) == 6:
                watershed_mask[int(br[2]):(int(br[5])+1),
                int(br[1]):(int(br[4])+1),int(br[0]):(int(br[3])+1)] = 0
                masked_bboxes.append(br)
                mask_file.close()
    
    
    connectivity = None
    #if not IS_3D_WS:
    #    connectivity = numpy.zeros((3,3), dtype=numpy.bool_)
    #    connectivity[0,1] = True
    #    connectivity[1,0] = True
    #    connectivity[2,1] = True
    #    connectivity[1,2] = True
    # 
    print 'boundary_cropped.shape = %r' % (boundary_cropped.shape,)
    print 'boundary_cropped[:,:,0].shape = %r' % (boundary_cropped[:,:,0].shape,)
    print 'seeds_cropped[0].shape = %r' % (seeds_cropped[0].shape,)
    
    n_images = boundary_cropped.shape[2]
    supervoxels_cropped = numpy.zeros(boundary_cropped.shape)
    if IS_3D_WS:
        supervoxels_cropped = skmorph.watershed(boundary_cropped, seeds_cropped, connectivity, None, watershed_mask)    
    else:
        for i in xrange(n_ws_steps):
            start_idx = i * DEPTH_2D_3D_WS
            end_idx = start_idx + DEPTH_2D_3D_WS
            supervoxels_cropped[:,:,start_idx:end_idx] = skmorph.watershed(boundary_cropped[:,:,start_idx:end_idx], seeds_cropped[i], connectivity, None, watershed_mask[:,:,start_idx:end_idx])
            print 'supervoxels_cropped[:,:,%d] done' % (i,)
    
    supervoxels = supervoxels_cropped
    if options.border_size > 0:
        supervoxels = seeds.copy()
        supervoxels.dtype = supervoxels_cropped.dtype
        supervoxels[:,:,:] = 0 
        supervoxels[options.border_size:(-1*options.border_size), 
        options.border_size:(-1*options.border_size),options.border_size:(-1*options.border_size)] = supervoxels_cropped
    
    master_logger.info("Finished watershed")
    
    return supervoxels, prediction


def agglomeration(options, agglom_stack, supervoxels, prediction, 
        image_stack, session_location, sp_outs, master_logger):
    
    seg_thresholds = sorted(options.segmentation_thresholds)
    for threshold in seg_thresholds:
        if threshold != 0 or not options.use_neuroproof:
            master_logger.info("Starting agglomeration to threshold " + str(threshold)
                + " with " + str(agglom_stack.number_of_nodes()))
            agglom_stack.agglomerate(threshold)
            master_logger.info("Finished agglomeration to threshold " + str(threshold)
                + " with " + str(agglom_stack.number_of_nodes()))
            
            if options.inclusion_removal:
                inclusion_removal(agglom_stack, master_logger)

        segmentation = agglom_stack.get_segmentation()     

        if options.h5_output:
            imio.write_image_stack(segmentation,
                session_location+"/agglom-"+str(threshold)+".lzf.h5", compression='lzf')
          
        
        md5hex = hashlib.md5(' '.join(sys.argv)).hexdigest()
        file_base = os.path.abspath(session_location)+"/seg_data/seg-"+str(threshold) + "-" + md5hex + "-"
        transforms = imio.compute_sp_to_body_map(supervoxels, segmentation)
        seg_loc = file_base +"v1.h5"
        if not os.path.exists(session_location+"/seg_data"):
            os.makedirs(session_location+"/seg_data")
        imio.write_mapped_segmentation(supervoxels, transforms, seg_loc)    

        if options.synapse_file is not None:
            h5temp = h5py.File(seg_loc, 'a')
            syn_data = json.load(open((options.synapse_file)))
            meta = syn_data['metadata']
            meta['username'] = "auto"
            syn_data_str = json.dumps(syn_data, indent=4)
            str_type = h5py.new_vlen(str)
            ds = h5temp.create_dataset("synapse-annotations", data=syn_data_str, shape=(1,), dtype=str_type)

        graph_loc = file_base+"graphv1.json"
       
        json_data = {}
        json_data['graph'] = graph_loc
        json_data['border'] = options.border_size  
        subvolume = {}
        subvolume['segmentation-file'] = seg_loc
        subvolume['prediction-file'] = os.path.abspath(session_location) + "/STACKED_prediction.h5"
        
        gray_file_whole = os.path.abspath(glob.glob(options.image_stack)[0])
        gray_path = os.path.dirname(gray_file_whole)
       
        gray_file = os.path.basename(gray_file_whole)
        field_width = len(re.findall(r'\d',gray_file))
        field_rep = "%%0%dd" % field_width
        gray_file = re.sub(r'\d+', field_rep, gray_file)
        
        subvolume['grayscale-files'] = gray_path + "/" + gray_file
        
        # get extant
        x1 = options.border_size
        y1 = options.border_size
        z1 = options.border_size
        z2,y2,x2 = supervoxels.shape
        z2 = z2 - options.border_size - 1
        y2 = y2 - options.border_size - 1
        x2 = x2 - options.border_size - 1
        extant = re.findall(r'\d+-\d+_\d+-\d+_\d+-\d+', gray_path)
        if len(extant) > 0:
            bbox = extant[0]
            x1,x2,y1,y2,z1,z2 = re.findall(r'\d+', bbox)
        subvolume["far-upper-right"] = [int(x2),int(y2),int(z2)]
        subvolume["near-lower-left"] = [int(x1),int(y1),int(z1)]

        json_data['subvolumes'] = [subvolume]
        
        agglom_stack.write_plaza_json(graph_loc, options.synapse_file, (int(z1)-(options.border_size)))
         
        # write out json file
        json_str = json.dumps(json_data, indent=4)
        json_file = session_location + "/seg-" + str(threshold) + "-" + md5hex + "-v1.json"
        jw = open(json_file, 'w')
        jw.write(json_str)

        #if options.raveler_output:
        #    sps_outs = output_raveler(segmentation, supervoxels, image_stack, "agglom-" + str(threshold),
        #        session_location, master_logger)   
        #    master_logger.info("Writing graph.json")
        #    agglom_stack.write_plaza_json(session_location+"/raveler-export/agglom-"+str(threshold)+"/graph.json",
        #                                    options.synapse_file)
        #    if options.synapse_file is not None:
        #        shutil.copyfile(options.synapse_file,
        #                session_location + "/raveler-export/agglom-"+str(threshold)+"/annotations-synapse.json") 
        #    master_logger.info("Finished writing graph.json")


def inclusion_removal(agglom_stack, master_logger):
    master_logger.info("Starting inclusion removal with " + str(agglom_stack.number_of_nodes()) + " nodes")
    agglom_stack.remove_inclusions()
    master_logger.info("Finished inclusion removal with " + str(agglom_stack.number_of_nodes()) + " nodes")


def output_raveler(segmentation, supervoxels, grayscale, name, session_location, master_logger, sps_out=None):
    """Output segmented data to a Raveler formatted directory

    Args:
        segmentation:  ndarray.
        prediction_file:  String.  File name of prediction hdf5 file where predictions
            are assumed to be in group PREDICTIONS_HDF5_GROUP.
        grayscale:  ndarray of grayscale.
        name:  String.  Directory name within raveler-export.
        session_location:  String.  Top-level export directory.

    Returns:
        A 2-tuple of supervoxel and prediction matrices.
    """

    outdir = session_location + "/raveler-export/" + name + "/"
    master_logger.info("Exporting Raveler directory: " + outdir)

    rav = imio.segs_to_raveler(supervoxels, segmentation, 0, do_conn_comp=False, sps_out=sps_out)
    sps_out, dummy1, dummy2 = rav
    
    if os.path.exists(outdir):
        master_logger.warning("Overwriting Raveler directory: " + outdir)
        shutil.rmtree(outdir)
    imio.write_to_raveler(*rav, directory=outdir, gray=grayscale)
    return sps_out



def flow_perform_agglomeration(options, supervoxels, prediction, image_stack,
                                session_location, sps_out, master_logger): 
    # make synapse constraints
    synapse_volume = numpy.array([])
    if not options.use_neuroproof and options.synapse_file is not None:
        pre_post_pairs = syngeo.io.raveler_synapse_annotations_to_coords(
            options.synapse_file)
        synapse_volume = \
            syngeo.io.volume_synapse_view(pre_post_pairs, supervoxels.shape)

     # ?! build RAG (automatically load features if classifier file is available, default to median
    # if no classifier, check if np mode or not, automatically load features in NP as well)

    if options.classifier is not None:
        cl = classify.load_classifier(options.classifier)
        fm_info = json.loads(str(cl.feature_description))

        master_logger.info("Building RAG")
        if fm_info is None or fm_info["neuroproof_features"] is None:
            raise Exception("agglomeration classifier to old to be used") 
        if options.use_neuroproof:
            if not fm_info["neuroproof_features"]:
                raise Exception("random forest created not using neuroproof") 
            agglom_stack = stack_np.Stack(supervoxels, prediction,
                single_channel=False, classifier=cl, feature_info=fm_info, 
                synapse_file=options.synapse_file, master_logger=master_logger) 
        else:
            if fm_info["neuroproof_features"]:
                master_logger.warning("random forest created using neuroproof features -- should still work") 
            # fm = features.io.create_fm(fm_info)
            if options.expected_vi:
                # mpf = agglo.expected_change_vi(fm, cl, beta=options.vi_beta)
                pass
            else:
                # mpf = agglo.classifier_probability(fm, cl)
                pass
            
            ''' agglom_stack = agglo.Rag(supervoxels, prediction, mpf,
                    feature_manager=fm, show_progress=True, nozeros=True, 
                    exclusions=synapse_volume)'''
        master_logger.info("Finished building RAG")
    else:
        master_logger.info("Building RAG")
        boundary = grab_boundary(prediction, options.bound_channels, master_logger)   
        '''agglom_stack = agglo.Rag(supervoxels, boundary, 
                merge_priority_function=agglo.boundary_median,
                show_progress=True, nozeros=True, exclusions=synapse_volume)'''
        master_logger.info("Finished building RAG")


    # remove inclusions 
    if options.inclusion_removal:
        inclusion_removal(agglom_stack, master_logger) 

    # actually perform the agglomeration
    agglomeration(options, agglom_stack, supervoxels, prediction, image_stack,
        session_location, sps_out, master_logger) 





def run_segmentation_pipeline(session_location, options, master_logger): 

    prediction_file  = options.pixelprob_file

    # generate supervoxels
    supervoxels = None
    prediction = None

    print "Generating supervoxels:"

    supervoxels, prediction = gen_supervoxels(options, prediction_file, master_logger) 

    # write superpixels out to hdf5 and/or raveler files
    sps_out = None
    image_stack = None

    if supervoxels is None:
        raise Exception('ERROR: supervoxels is None')

    print "Going to write to h5 format:"
    print "Location: " + session_location + "/" + options.supervoxels_name
    imio.write_image_stack(supervoxels, session_location + "/" + options.supervoxels_name, compression=None)
    print "Compression: None"

                
def prediction_file_verify(options_parser, options, master_logger):
    if options.ilastik_prediction_file and not os.path.isfile(options.ilastik_prediction_file):
        raise Exception("ilastik-prediction-file (%s) specified in parameters does not exist")

def np_verify(options_parser, options, master_logger):
    if options.use_neuroproof and not np_installed:
        raise Exception("NeuroProof not properly installed on your machine.  Install or disable neuroproof")
        
def synapse_file_verify(options_parser, options, master_logger):
    if options.synapse_file:
        if not os.path.exists(options.synapse_file):
            raise Exception("Synapse file " + options.synapse_file + " not found")
        if not options.synapse_file.endswith('.json'):
            raise Exception("Synapse file " + options.synapse_file + " does not end with .json")

def classifier_verify(options_parser, options, master_logger):
    if options.classifier is not None:
        if not os.path.exists(options.classifier):
            raise Exception("Classifier " + options.classifier + " not found")
    # Note -- Classifier could be a variety of extensions (.h5, .joblib, etc) depending
    #  on whether classifier is sklearn or vigra.

def gen_supervoxels_verify(options_parser, options, master_logger):
    if options.gen_supervoxels and not options.gen_pixel and options.pixelprob_file is None:
        raise Exception("Must have a pixel prediction to generate supervoxels")
     

def supervoxels_file_verify(options_parser, options, master_logger):
    if options.supervoxels_file is not None:
        if not os.path.exists(options.supervoxels_file):
            raise Exception("Supervoxel file " + options.supervoxels_file + " does not exist")

def gen_agglomeration_verify(options_parser, options, master_logger):
    if options.gen_agglomeration:
        if not options.gen_supervoxels and options.supervoxels_file is None:
            raise Exception("No supervoxels available for agglomeration")
        if not options.gen_pixel and options.pixelprob_file is None:
            raise Exception("No prediction available for agglomeration")


def create_segmentation_pipeline_options(options_parser):
    pixel.create_pixel_options(options_parser, False)

    options_parser.create_option("ilastik-prediction-file", 
        "Name of prediction file generated by ilastik headless",
        default_val=None, required=False, dtype=str, verify_fn=prediction_file_verify,
        num_args=None, shortcut=None, warning=False, hidden=False)
    
    options_parser.create_option("use-neuroproof", "Use NeuroProof", 
        default_val=False, required=False, dtype=bool, verify_fn=np_verify, num_args=None,
        shortcut='NP', warning=False, hidden=(not np_installed)) 

    options_parser.create_option("supervoxels-name", "Name for the supervoxel segmentation", 
        default_val="supervoxels.h5", required=False, dtype=str, verify_fn=None, num_args=None,
        shortcut=None, warning=False, hidden=True) 

    options_parser.create_option("supervoxels-file", "Supervoxel segmentation file or directory stack", 
        default_val=None, required=False, dtype=str, verify_fn=supervoxels_file_verify, num_args=None,
        shortcut=None, warning=False, hidden=True) 
   
    options_parser.create_option("gen-supervoxels", "Enable supervoxel generation", 
        default_val=False, required=False, dtype=bool, verify_fn=gen_supervoxels_verify, num_args=None,
        shortcut='GS', warning=True, hidden=False) 

    options_parser.create_option("inclusion-removal", "Disable inclusion removal", 
        default_val=True, required=False, dtype=bool, verify_fn=None, num_args=None,
        shortcut='IR', warning=False, hidden=False) 

    options_parser.create_option("seed-val", "Threshold for choosing seeds", 
        default_val=0, required=False, dtype=float, verify_fn=None, num_args=None,
        shortcut=None, warning=False, hidden=True) 

    options_parser.create_option("seed-size", "Threshold for seed size", 
        default_val=0, required=False, dtype=int, verify_fn=None, num_args=None,
        shortcut='SS', warning=False, hidden=False) 

    options_parser.create_option("synapse-file", "Json file containing synapse information", 
        default_val=None, required=False, dtype=str, verify_fn=synapse_file_verify, num_args=None,
        shortcut='SJ', warning=False, hidden=False) 

    options_parser.create_option("mask-file", "Text file specifying a region with no segmentation (left-hand coordinates and should be offset from border", 
        default_val=None, required=False, dtype=str, verify_fn=None, num_args=None,
        shortcut=None, warning=False, hidden=True) 

    options_parser.create_option("segmentation-thresholds", "Segmentation thresholds", 
        default_val=[], required=False, dtype=float, verify_fn=None, num_args='+',
        shortcut='ST', warning=True, hidden=False) 

    options_parser.create_option("gen-agglomeration", "Enable agglomeration", 
        default_val=False, required=False, dtype=bool, verify_fn=gen_agglomeration_verify, num_args=None,
        shortcut='GA', warning=True, hidden=False) 
    
    options_parser.create_option("raveler-output", "Disable Raveler output", 
        default_val=True, required=False, dtype=bool, verify_fn=None, num_args=None,
        shortcut=None, warning=False, hidden=True) 

    options_parser.create_option("h5-output", "Enable h5 output", 
        default_val=False, required=False, dtype=bool, verify_fn=None, num_args=None,
        shortcut=None, warning=False, hidden=True) 

    options_parser.create_option("classifier", "H5 file containing RF", 
        default_val=None, required=False, dtype=str, verify_fn=classifier_verify, num_args=None,
        shortcut='k', warning=False, hidden=False) 

    options_parser.create_option("bound-channels", "Channel numbers designated as boundary", 
        default_val=[0], required=False, dtype=int, verify_fn=None, num_args='+',
        shortcut=None, warning=False, hidden=True) 

    options_parser.create_option("expected-vi", "Enable expected VI during agglomeration", 
        default_val=False, required=False, dtype=bool, verify_fn=None, num_args=None,
        shortcut=None, warning=False, hidden=True) 

    options_parser.create_option("vi-beta", "Relative penalty for false merges in weighted expected VI", 
        default_val=1.0, required=False, dtype=float, verify_fn=None, num_args=None,
        shortcut=None, warning=False, hidden=True) 

    options_parser.create_option("synapse-dilation", "Dilate synapse points by this amount", 
        default_val=1, required=False, dtype=int, verify_fn=None, num_args=None,
        shortcut=None, warning=False, hidden=True) 

    options_parser.create_option("border-size", "Size of the border in pixels", 
        default_val=0, required=False, dtype=int, verify_fn=None, num_args=None,
        shortcut=None, warning=False, hidden=True) 

def entrypoint(argv):
    applogger = app_logger.AppLogger(False, 'seg-pipeline')
    master_logger = applogger.get_logger()

    try: 
        session = session_manager.Session("seg-pipeline", 
            "Segmentation pipeline (featuring boundary prediction, median agglomeration or trained agglomeration, inclusion removal, and raveler exports)", 
            master_logger, applogger, create_segmentation_pipeline_options)    
        master_logger.info("Session location: " + session.session_location)
        run_segmentation_pipeline(session.session_location, session.options, master_logger) 
    except Exception, e:
        master_logger.error(str(traceback.format_exc()))
    except KeyboardInterrupt, err:
        master_logger.error(str(traceback.format_exc()))
 
   
if __name__ == "__main__":
    sys.exit(main(sys.argv))
