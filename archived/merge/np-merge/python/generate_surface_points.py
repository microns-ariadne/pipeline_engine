import argparse
import multiprocessing as mp
import time
import re
import os
import sys
import fcntl

import h5py
import scipy
import scipy.ndimage
import numpy as np

BLOCK_RGX = "out_segmentation_([0-9]+)_([0-9]+)_([0-9]+).h5"

CLOUD_PTS_EXT = 'pts'

def printLabelPts(label, points, resDir):
    a = label / 1000000
    b = (label % 1000000) / 1000

    cloudDir = os.path.join(resDir, "%03d" % a, "%03d" % b)
    os.system('mkdir -p %s' % cloudDir)

    with open(os.path.join(cloudDir, '%08d.%s' % (label, CLOUD_PTS_EXT)), 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)

        np.savetxt(f, points, fmt='%d')

        fcntl.flock(f, fcntl.LOCK_UN)

def printSurfacePoints((path, block, blockSize, resolution, scale, resultsDir)):
    start = time.time()

    print "STARTING BLOCK: %s" % path

    with h5py.File(path) as f:
        data = np.array(f['stack'])

    data = data[::resolution[0], ::resolution[1], ::resolution[2]]

    boundary = (scipy.ndimage.grey_erosion(data, size=(3, 3, 3)) != scipy.ndimage.grey_dilation(data, size=(3, 3, 3)))

    labels = data[boundary]
    boundaryCoord = np.where(boundary)

    points = np.zeros((len(labels), 3))
    for i in xrange(3):
        points[:, i] = (boundaryCoord[i] * resolution[i] + block[2 - i] * blockSize[i]) * scale[i]

    labels_argsort = labels.argsort()

    labels = labels[labels_argsort]
    points = points[labels_argsort, :]

    first = 0
    nLabels = 1
    for i in xrange(1, len(labels)):
        if labels[i] != labels[i - 1]:
            printLabelPts(labels[first], points[first : i, :], resultsDir)
            first = i

            nLabels += 1

    printLabelPts(labels[first], points[first :, :], resultsDir)

    print "DONE BLOCK %s:\n    Time: %f\n    #surface points: %d\n    #unique labels: %d\n" % (path, time.time() - start, len(labels), nLabels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Generates point clouds for all objects in the given dataset.")

    parser.add_argument("--data-dir",
            dest="dataDir",
            required=True,
            type=str,
            help="Directory containing %s files" % BLOCK_RGX)

    parser.add_argument("--results-dir",
            dest="resultsDir",
            required=True,
            type=str,
            help="Where to output all point clouds")

    parser.add_argument("--block-size",
            nargs=3,
            dest="blockSize",
            default=[100, 1024, 1024],
            type=int,
            help="Size of each block in Z, Y, X dimensions - used to determine coordinates")

    parser.add_argument("--resolution",
            nargs=3,
            dest="resolution",
            default=[1, 5, 5],
            type=int,
            help="Resolution - how many pixels to skip in each dimension (Z, Y, X)")

    parser.add_argument("--z-scale",
            nargs=3,
            dest="scale",
            default=[5, 1, 1],
            type=float,
            help="Multiply coordinates in each direction by corresponding scale. Order is (Z, Y, X)")

    parser.add_argument("--n-proc",
            dest="nProc",
            type=int,
            default=72,
            help="Number of processes to use for computation")

    args = parser.parse_args()

    if not os.path.isdir(args.dataDir):
        print "Data directory does not exist: %s" % args.dataDir
        sys.exit(0)

    if os.path.isdir(args.resultsDir):
        print "Results directory already exists: %s" % args.resultsDir
        sys.exit(0)

    args.scale.reverse()
    args.resolution.reverse()
    args.blockSize.reverse()

    start = time.time()
    print "STARTING: Retreiving surface points from %s" % args.dataDir

    dataDirFiles = os.listdir(args.dataDir)

    processArgs = [ ]
    for fname in dataDirFiles:
        match = re.search(BLOCK_RGX, fname)
        if match is not None:
            ind = [int(c) for c in match.groups()]
            processArgs.append((os.path.join(args.dataDir, fname),
                                ind,
                                args.blockSize,
                                args.resolution,
                                args.scale,
                                args.resultsDir))

    print "Number of segmentation files found: %d" % len(processArgs)

    p = mp.Pool(args.nProc)
    p.map(printSurfacePoints, processArgs)

    print "DONE: Generating surface points from %s" % args.dataDir
    print "    Time: %f" % (time.time() - start)

