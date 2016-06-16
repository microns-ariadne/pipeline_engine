import time

import numpy as np

import voi

def get_merge_pairs(
    slice_supv, 
    slice_segm, 
    shift, 
    width, 
    axis):
    
    print 'START: Calculating merge pairs'
    start = time.time()

    assert (slice_supv.shape == slice_segm.shape), 'slice_supv.shape = %r , slice_segm.shape = %r' % (
        slice_supv.shape,
        slice_segm.shape)
    
    ndim = len(slice_supv.shape)
    coord = [slice(None)] * ndim

    coord[axis] = width - 1
    supv1 = slice_supv[coord]
    segm1 = slice_segm[coord]

    coord[axis] = width
    supv2 = slice_supv[coord]
    segm2 = slice_segm[coord]

    #merge_locations = ((0 < supv1) & (shift < supv2)) & ((segm1 == supv2) | (segm2 == supv1))
    merge_locations = ((supv1 != 0) & (supv2 != 0)) & (segm1 == segm2) & (segm1 != 0)
    pairs = np.vstack([supv1[merge_locations], supv2[merge_locations] - shift])

    if pairs.shape[1] == 0:
        return pairs

    M = pairs[1, ...].max() + 1

    pairs_int = np.unique(pairs[0, ...] * M + pairs[1, ...])

    pairs = np.vstack([pairs_int / M, pairs_int % M])

    print 'FINISH: Time %f' % (time.time() - start)

    return pairs

def get_heuristic_merge_pairs(supv, segm, shift, width, axis, threshold):
    print 'START: Heuristic merge pairs'
    start = time.time()

    ndim = len(segm.shape)
    coord1 = [slice(None)] * ndim
    coord1[axis] = width - 1

    coord2 = [slice(None)] * ndim
    coord2[axis] = width

    # locations = (0 < segm[coord1]) & (segm[coord1] == supv[coord1]) & (shift < segm[coord2]) & (segm[coord2] == supv[coord2])
    locations = (segm[coord1] != 0) & (segm[coord2] != 0) & (segm[coord1] == supv[coord1]) & (segm[coord2] == supv[coord2])
    a = segm[coord1][locations]
    b = segm[coord2][locations] - shift

    assert (b <= 0).sum() == 0

    if len(a) == 0:
        return np.array([], dtype='uint32')

    M = b.max() + 1
    pair_count = np.bincount(a * M + b)
    pairs_int = np.where(pair_count > threshold)[0]

    pairs = np.vstack([pairs_int / M, pairs_int % M, pair_count[pairs_int]])

    print 'FINISH: Time %f' % (time.time() - start)

    return pairs

def get_delta_vi_for_pairs(segm, np_segm, pairs, shift):
    print 'START: VI merge pairs'
    start = time.time()

    v = voi.VariationOfInformation(np_segm, segm)
    merge_voi, split_voi = v.merge_voi, v.split_voi

    res = [ ]
    for pair in pairs:
        [a, b] = pair

        v.merge(a, b + shift)
        new_merge, new_split = v.merge_voi, v.split_voi
        v.backtrack()

        res.append([new_merge - merge_voi, new_split - split_voi])

    print 'FINISH: Time %f' % (time.time() - start)
    return np.array(res)

