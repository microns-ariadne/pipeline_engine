'''Find the shortest path from one point to another through a segmentation'''

import json
import numpy as np
import os
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.ndimage import grey_dilation, grey_erosion

from ..targets.png_volume_target import PngVolumeTarget
from ..tasks.utilities import to_hashable

def make_volume_map(root, seg_name, pattern):
    '''Make a mapping of volume to volume target by scanning directories
    
    :param root: the root directory of the pipeline results
    :param seg_name: the name of the segmentation to fetch, eg "neuroproof"
    :param pattern: the file naming pattern
    
    returns a map of volume to PngVolumeTarget
    '''
    d = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(seg_name+".done"):
                tgt = PngVolumeTarget.from_done_file(
                    os.path.join(dirpath, filename), pattern)
                d[to_hashable(tgt.volume.to_dictionary())] = tgt
    return d
    
def find_path(volume_map, acc_path, x0, y0, z0, x1, y1, z1):
    '''Find the smallest volume of segments connecting two points
    
    :param volume_map: a map of volume to target
    '''
    
    xmin = min(x0, x1)
    xmax = max(x0, x1)
    ymin = min(y0, y1)
    ymax = max(y0, y1)
    zmin = min(z0, z1)
    zmax = max(z0, z1)
    acc_map_dict = json.load(open(acc_path, "r"), object_hook=to_hashable)
    acc_map_by_volume = dict(acc_map_dict["volumes"])
    all_areas = np.zeros(0, int)
    all_seg = []
    all_adj = []
    l0 = None
    l1 = None
    for volume, tgt in volume_map.items():
        if tgt.volume.x + tgt.volume.width < xmin or tgt.volume.x > xmax:
            continue
        if tgt.volume.y + tgt.volume.height < ymin or tgt.volume.y > ymax:
            continue
        if tgt.volume.z + tgt.volume.depth < zmin or tgt.volume.z > zmax:
            continue
        acc_map = np.array(acc_map_by_volume[volume])
        g_labels = np.zeros(np.max(acc_map[:, 0])+1, int)
        g_labels[acc_map[:, 0]] = acc_map[:, 1]
        #
        # Compute the adjacencies
        #
        seg = g_labels[tgt.imread()]
        #
        # Maybe one of the points is within the volume
        #
        if tgt.volume.contains(x0, y0, z0):
            l0 = seg[z0 - tgt.volume.z,
                     y0 - tgt.volume.y,
                     x0 - tgt.volume.x]
        if tgt.volume.contains(x1, y1, z1):
            l1 = seg[z1 - tgt.volume.z,
                     y1 - tgt.volume.y,
                     x1 - tgt.volume.x]
                    
        areas = np.bincount(g_labels.ravel())
        if len(areas) > len(all_areas):
            all_areas, areas = areas, all_areas
        all_areas[:len(areas)] += areas.astype(all_areas.dtype)
        strel = np.array([[[False, False, False],
                           [False, True, False],
                           [False, False, False]],
                          [[False, True, False],
                           [True, True, True],
                           [False, True, False]],
                          [[False, False, False],
                           [False, True, False],
                           [False, False, False]]])
        high = grey_dilation(seg, footprint=strel)
        mask = (seg != 0) & (high > seg)
        seg_adj = seg[mask]
        adj = high[mask]
        #
        # Compile into a matrix once to reduce to single instances
        # of adjacencies
        #
        matrix = coo_matrix((np.ones(len(adj), int),
                             (seg_adj, adj)))
        seg, adj = matrix.nonzero()
        all_seg.append(seg)
        all_adj.append(adj)
        all_seg.append(adj)
        all_adj.append(seg)
    #
    # Throw an exception if we couldn't find a segment with the given coords
    #
    if l0 is None:
        raise ValueError("Could not find segment containing %d, %d, %d" %
                         (x0, y0, z0))
    if l1 is None:
        raise ValueError("Could not find segment containing %d, %d, %d" %
                         (x1, y1, z1))
    #
    # The weight of an edge is the area of the destination segment. That way,
    # we will find the shortest path = volume of the connecting segments
    #
    all_seg = np.hstack(all_seg)
    all_adj = np.hstack(all_adj)
    matrix = coo_matrix((np.ones(len(all_seg)), (all_seg, all_adj))).tocsr()
    all_seg, all_adj = matrix.nonzero()
    weights = all_areas[all_adj]
    matrix = coo_matrix((weights, (all_seg, all_adj)))
    dist_matrix, predecessors = shortest_path(matrix, return_predecessors=True)
    path = [l1]
    l = l1
    while l != l0:
        l = predecessors[l0, l]
        path.insert(0, l)
    return path

    
        
        
        
        
        