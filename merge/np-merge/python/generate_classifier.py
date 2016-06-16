import sys
import os
import argparse

import h5py
import cilk_rw
import numpy as np

import multiprocessing as mp
import time

from util import *

NP_LEARN = '/home/armafire/Pipeline/exec/neuroproof_agg/npclean/build/neuroproof_graph_learn_old_4d'
NP_PREDICT = '/home/armafire/Pipeline/exec/neuroproof_agg/npclean/build/neuroproof_graph_predict'
# WATERSHED = '/mnt/disk1/hayk/oversegmentation-h5-make-0/watershed-3D-seeds-3D-BFS-0-BG-HDF5.x'
# WATERSHED = '/mnt/disk1/hayk/execs/watershed-3D-seeds-3D-BFS-0-BG-RGB.x'
WATERSHED = '/mnt/disk1/hayk/execs/h5_watershed.x --use-h5'

WSHED_TMP_DIR = 'wshed_tmp_slice_%d_half_%d'

SLICE_WIDTH = [32, 128, 128]

PROBS_HALF_TEMP = 'probs_slice_%d_half_%d.h5'
PROBS_4D_HALF_TEMP = 'probs4d_slice_%d_half_%d.h5'
SUPV_HALF_TEMP = 'supv_slice_%d_half_%d.h5'

LABELS_HALF_TEMP = 'labels_slice_%d_half_%d.h5'

SEGM_HALF_TEMP = 'segm_slice_%d_half_%d.h5'

GAP = 4

TOTAL_LABELS = 'all_labels.h5'
TOTAL_SUPV = 'all_supv.h5'
TOTAL_PROB = 'all_probs.h5'

# OVERLAP = [10, 30, 30]
OVERLAP = [0, 0, 0]

def save_halfs(data, middle, width, dset, axis, half_paths):
    ndim = len(data.shape)
    coord = [slice(None)] * ndim
   
    overlap = OVERLAP[axis]
    if not h5py.is_hdf5(half_paths[0]):
        coord[axis] = slice(middle - width, middle + overlap)
        with h5py.File(half_paths[0], 'w') as f:
            f.create_dataset(dset, data=data[coord], compression='gzip')
    else:
        print 'Half already exists: %s' % half_paths[0]

    if not h5py.is_hdf5(half_paths[1]):
        coord[axis] = slice(middle - overlap, middle + width)
        with h5py.File(half_paths[1], 'w') as f:
            f.create_dataset(dset, data=data[coord], compression='gzip')
    else:
        print 'Half already exists: %s' % half_paths[1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Generates classifier")

    parser.add_argument('--probs',
                        dest='probsPath',
                        type=str,
                        required=True,
                        help='Prediction file for ground truth')
    parser.add_argument('--labels',
                        dest='labelsPath',
                        type=str,
                        required=True)
    parser.add_argument('--direction',
                        dest='direction',
                        type=int,
                        required=True,
                        help='0, 1 or 2, index in probability file')
    parser.add_argument('--width',
                        dest='width',
                        type=int,
                        default=-1)
    parser.add_argument('--work-dir',
                        dest='work_dir',
                        type=str,
                        required=True)
    parser.add_argument('--cuts',
                        dest='cuts',
                        type=int,
                        required=True)

    args = parser.parse_args()

    if args.direction != 0 and args.direction != 1 and args.direction != 2:
        print "Error: direction has to be one of %d" % args.direction
        sys.exit(0)

    args.work_dir = os.path.abspath(args.work_dir)

    if os.path.isdir(args.work_dir):
        print "Directory %s exists." % (args.work_dir)
       #sys.exit(0)
    else:
        exec_cmd('mkdir %s' % args.work_dir)

    if args.width == -1:
        args.width = SLICE_WIDTH[args.direction]

    probs = np.array(h5py.File(args.probsPath)['volume/predictions'], dtype='float32')
    probs4d = np.concatenate([probs[..., np.newaxis], probs[..., np.newaxis]], axis=3)
    labels = np.array(h5py.File(args.labelsPath)['stack'], dtype='uint32')

    labels_half_path = [[os.path.join(args.work_dir, LABELS_HALF_TEMP % (cut, half)) for half in [0, 1]] for cut in xrange(args.cuts)]
    probs_half_path = [[os.path.join(args.work_dir, PROBS_HALF_TEMP % (cut, half)) for half in [0, 1]] for cut in xrange(args.cuts)]
    probs4d_half_path = [[os.path.join(args.work_dir, PROBS_4D_HALF_TEMP % (cut, half)) for half in [0, 1]] for cut in xrange(args.cuts)]
    supv_half_path = [[os.path.join(args.work_dir, SUPV_HALF_TEMP % (cut, half)) for half in [0, 1]] for cut in xrange(args.cuts)]
    wshed_tmp_dir = [[os.path.join(args.work_dir, WSHED_TMP_DIR % (cut, half)) for half in [0, 1]] for cut in xrange(args.cuts)] # temporary dir for watershed's work
    segm_half_path = [[os.path.join(args.work_dir, SEGM_HALF_TEMP % (cut, half)) for half in [0, 1]] for cut in xrange(args.cuts)]

    classifier_path = os.path.join(args.work_dir, 'slice-classifier-%d-%d-%d.xml' % (args.direction, args.width, args.cuts))

    size = probs.shape[args.direction]
    cut_locations = [args.width + cut * ((size - 2 * args.width) / (args.cuts - 1)) for cut in xrange(args.cuts)] # cut happens immediately before these.
    
    print "Cut locations: %s" % str(cut_locations)

    print 'Saving halfs'
    for cut in xrange(args.cuts):
        save_halfs(probs, cut_locations[cut], args.width, 'volume/predictions', args.direction, probs_half_path[cut])
        save_halfs(probs4d, cut_locations[cut], args.width, 'volume/predictions', args.direction, probs4d_half_path[cut])
        # save_halfs(labels, cut_locations[cut], args.width, 'stack', 2 - args.direction, labels_half_path[cut])

    print 'Starting watershed'
    start = time.time()
    pool = mp.Pool(args.cuts)

    commands = [ ]
    for cut in xrange(args.cuts):
        for half in [0, 1]:
            if not h5py.is_hdf5(supv_half_path[cut][half]):
                # commands.append('python %s --pixelprob-file %s --seed-size 5 %s' % (WATERSHED,
                #                                                                    probs4d_half_path[cut][half],
                #                                                                    wshed_tmp_dir[cut][half]))

                # commands.append('%s --input-path %s --output-path %s --use-h5;' % (WATERSHED, probs_half_path[cut][half], supv_half_path[cut][half]))
                commands.append('%s %s %s' % (WATERSHED, probs_half_path[cut][half], supv_half_path[cut][half]))

    pool.map(exec_cmd, commands)

    # for cut in xrange(args.cuts):
    #     for half in [0, 1]:
    #         if h5py.is_hdf5(os.path.join(wshed_tmp_dir[cut][half], 'supervoxels.h5')):
    #             exec_cmd('mv %s %s; rm -rf %s' % (os.path.join(wshed_tmp_dir[cut][half], 'supervoxels.h5'),
    #                                                    supv_half_path[cut][half],
    #                                                    wshed_tmp_dir[cut][half]))

    print "Watershed done; time %f" % (time.time() - start)

    print 'Starting segmentation on halfs'
    start = time.time()

    commands = [ ]
    for cut in xrange(args.cuts):
        for half in [0, 1]:
            if not h5py.is_hdf5(segm_half_path[cut][half]):
                commands.append('%s %s %s %s --output-file %s' % (NP_PREDICT,
                                                                  supv_half_path[cut][half],
                                                                  probs_half_path[cut][half],
                                                                  CLASSIFIER,
                                                                  segm_half_path[cut][half]))

    pool.map(exec_cmd, commands)
    print 'Segmentation on halfs is done: time = %f' % (time.time() - start)

    with h5py.File(probs_half_path[0][0]) as f:
        half_slice_shape = np.array(f['volume/predictions'].shape, dtype='int')

    total_probs_shape = np.append(half_slice_shape, 2)
    total_probs_shape[args.direction] = total_probs_shape[args.direction] * 2 * args.cuts + (args.cuts - 1) * GAP
    total_probs = np.ones(shape=total_probs_shape, dtype='float32')

    total_labels_shape = half_slice_shape[::-1]
    total_labels_shape[2 - args.direction] = total_labels_shape[2 - args.direction] * 2 * args.cuts + (args.cuts - 1) * GAP
    total_labels = np.zeros(shape=total_labels_shape, dtype='uint32')
    total_supervoxels = np.zeros(shape=total_labels_shape, dtype='uint32')

    total_probs_coord = [slice(None)] * len(total_probs_shape)
    total_labels_coord = [slice(None)] * len(total_labels_shape)
   
    total_probs_coord[args.direction] = slice(0, 2 * args.width)
    total_labels_coord[2 - args.direction] = slice(0, 2 * args.width)

    print "Total labels shape: %s" % str(total_labels_shape)
    print "Total probs shape: %s" % str(total_probs_shape)

    gt_labels_shift = np.array(0, dtype='uint32')
    supv_labels_shift = np.array(0, dtype='uint32')
    for cut in xrange(args.cuts):
        cut_location = cut_locations[cut]

        print "Cut location: %d" % cut_location
        coord = [slice(None)] * len(total_probs_shape)
        coord[args.direction] = slice(cut_location - args.width, cut_location + args.width)
        print "Assigning %s of original to %s of total probabilities" % (str(coord), str(total_probs_coord))
        total_probs[total_probs_coord] = probs4d[coord]

        coord = [slice(None)] * len(total_labels_shape)
        coord[2 - args.direction] = slice(cut_location - args.width, cut_location + args.width)
        total_labels[total_labels_coord] = labels[coord] + gt_labels_shift
        print "Assigning %s of original to %s of total segmentation" % (str(coord), str(total_labels_coord))
        gt_labels_shift += labels[coord].max() + np.array(1, dtype='uint32')

        with h5py.File(segm_half_path[cut][0]) as f1, h5py.File(segm_half_path[cut][1]) as f2:
            # slice_coord = [slice(None)] * 3

            # slice_coord[2 - args.direction] = slice(0, args.width)
            # a = np.array(f1['stack'], dtype='uint32')[slice_coord]

            # overlap = OVERLAP[args.direction]
            # slice_coord[2 - args.direction] = slice(overlap, args.width + overlap)
            # b = np.array(f2['stack'], dtype='uint32')[slice_coord]

            a = np.array(f1['stack'], dtype='uint32')
            b = np.array(f2['stack'], dtype='uint32')
            supv_data = np.concatenate([a, b + a.max() + np.array(1, dtype='uint32')], axis=2 - args.direction)

            print "Number of objects in slice %d: %d" % (cut, np.unique(supv_data).shape[0])

        print "Assigning %s of original to %s of total supervoxels" % (str(coord), str(total_labels_coord))
        total_supervoxels[total_labels_coord] = supv_data + supv_labels_shift
        supv_labels_shift += supv_data.max() + np.array(1, dtype='uint32')

        total_probs_coord[args.direction] = slice(total_probs_coord[args.direction].start + GAP + 2 * args.width,
                                                  total_probs_coord[args.direction].stop + GAP + 2 * args.width)
        total_labels_coord[2 - args.direction] = slice(total_labels_coord[2 - args.direction].start + GAP + 2 * args.width,
                                                       total_labels_coord[2 - args.direction].stop + GAP + 2 * args.width)

    total_probs_path = os.path.join(args.work_dir, TOTAL_PROB)
    total_supv_path  = os.path.join(args.work_dir, TOTAL_SUPV)
    total_labels_path = os.path.join(args.work_dir, TOTAL_LABELS)

    print 'Saving slice files'
    with h5py.File(total_probs_path) as f:
        f.create_dataset('volume/predictions', data=total_probs, compression='gzip')
    with h5py.File(total_supv_path) as f:
        f.create_dataset('stack', data=total_supervoxels, compression='gzip')
    with h5py.File(total_labels_path) as f:
        f.create_dataset('stack', data=total_labels, compression='gzip')

    command = ('%s %s %s %s --classifier-name %s' % (NP_LEARN,
                                                     total_supv_path,
                                                     total_probs_path,
                                                     total_labels_path,
                                                     classifier_path))

    exec_cmd(command)

