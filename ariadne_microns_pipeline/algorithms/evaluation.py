import enum
import hungarian
import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy.sparse import coo_matrix
import fast64counter
import mahotas
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import glob
import os
import cPickle

def thin_boundaries(im, mask):
    im = im.copy()
    assert (np.all(im >= 0)), "Label images must be non-negative"

    # make sure image is not all zero
    if np.sum(im) == 0:
        im[:] = 1.0
        im[0,:] = 2.0

    # repeatedly expand regions by one pixel until the background is gone
    while (im[mask] == 0).sum() > 0:
        zeros = (im == 0)
        im[zeros] = maximum_filter(im, 3)[zeros]

    # make sure image is not constant to avoid zero division
    if len(np.unique(im))==1:
        im[0,:] = 5
    return im

def Rand(pair, gt, pred, alpha):
    '''Parameterized Rand score

    Arguments are pairwise fractions, ground truth fractions, and prediction
    fractions.

    Equation 3 from Arganda-Carreras et al., 2015
    alpha = 0 is Rand-Split, alpha = 1 is Rand-Merge

    '''

    return np.sum(pair ** 2) / (alpha * np.sum(gt ** 2) +
                                (1.0 - alpha) * np.sum(pred ** 2))

def VI(pair, gt, pred, alpha):
    ''' Parameterized VI score

    Arguments are pairwise fractions, ground truth fractions, and prediction
    fractions.

    Equation 6 from Arganda-Carreras et al., 2015
    alpha = 0 is VI-Split, alpha = 1 is VI-Merge
    '''

    pair_entropy = - np.sum(pair * np.log(pair))
    gt_entropy = - np.sum(gt * np.log(gt))
    pred_entropy = - np.sum(pred * np.log(pred))
    mutual_information = gt_entropy + pred_entropy - pair_entropy

    return mutual_information / ((1.0 - alpha) * gt_entropy + alpha * pred_entropy)

def segmentation_metrics(ground_truth, prediction, seq=False, per_object=False):
    '''Computes adjusted FRand and VI between ground_truth and prediction.

    Metrics from: Crowdsourcing the creation of image segmentation algorithms
    for connectomics, Arganda-Carreras, et al., 2015, Frontiers in Neuroanatomy

    ground_truth - correct labels
    prediction - predicted labels

    Boundaries (label == 0) in prediction are thinned until gone, then are
    masked to foreground (label > 0) in ground_truth.

    Return value is ((FRand, FRand_split, FRand_merge), (VI, VI_split, VI_merge)).

    If seq is True, then it is assumed that the ground_truth and prediction are
    sequences that should be processed elementwise.

    '''

    # make non-sequences into sequences to simplify the code below
    if not seq:
        ground_truth = [ground_truth]
        prediction = [prediction]

    counter_pairwise = fast64counter.ValueCountInt64()
    counter_gt = fast64counter.ValueCountInt64()
    counter_pred = fast64counter.ValueCountInt64()

    for gt, pred in zip(ground_truth, prediction):
        mask = (gt > 0)
        pred = thin_boundaries(pred, mask)
        gt = gt[mask].astype(np.int32)
        pred = pred[mask].astype(np.int32)
        counter_pairwise.add_values_pair32(gt, pred)
        counter_gt.add_values_32(gt)
        counter_pred.add_values_32(pred)

    # fetch counts
    frac_pairwise = counter_pairwise.get_counts()[1]
    frac_gt = counter_gt.get_counts()[1]
    frac_pred = counter_pred.get_counts()[1]

    # normalize to probabilities
    frac_pairwise = frac_pairwise.astype(np.double) / frac_pairwise.sum()
    frac_gt = frac_gt.astype(np.double) / frac_gt.sum()
    frac_pred = frac_pred.astype(np.double) / frac_pred.sum()

    alphas = {'F-score': 0.5, 'split': 0.0, 'merge': 1.0}

    Rand_scores = {k: Rand(frac_pairwise, frac_gt, frac_pred, v) for k, v in alphas.items()}
    VI_scores = {k: VI(frac_pairwise, frac_gt, frac_pred, v) for k, v in alphas.items()}
    result = {'Rand': Rand_scores, 'VI': VI_scores}
    if per_object:
        #
        # Compute summary statistics per object
        #
        ij, counts = counter_pairwise.get_counts()
        #
        # The label of predicted objects
        #
        i = ij & 0xffffffff
        #
        # The label of ground truth objects
        #
        j = ij >> 32
        #
        # # of pixels per predicted object
        #
        per_object_counts = np.bincount(i, weights=counts)
        #
        # Fraction of predicted object per ground truth object
        #
        frac = counts.astype(float) / per_object_counts[i]
        #
        # Entropy is - sum(p * log2(p))
        # Entropy = 0 for an exact match
        #
        entropy = -frac * np.log(frac) / np.log(2)
        tot_entropy = np.bincount(i, weights=entropy)
        unique_i = np.unique(i)
        #
        # area
        #
        area = np.bincount(np.hstack([_.flatten() for _ in prediction]))
        result["per_object"] = dict(
            object_id=unique_i.tolist(),
            area = area[unique_i].tolist(),
            overlap_area=per_object_counts[unique_i].tolist(),
            entropy=tot_entropy[unique_i].tolist())
    return result


# Just doing one, so the interface is easier for the network training
# And yes that means I should refactor the function above... when I have time
def quick_Rand(gt, pred, seq=False):
    counter_pairwise = fast64counter.ValueCountInt64()
    counter_gt = fast64counter.ValueCountInt64()
    counter_pred = fast64counter.ValueCountInt64()

    mask = (gt > 0)
    pred = thin_boundaries(pred, mask)
    gt = gt[mask].astype(np.int32)
    pred = pred[mask].astype(np.int32)
    counter_pairwise.add_values_pair32(gt, pred)
    counter_gt.add_values_32(gt)
    counter_pred.add_values_32(pred)

    # fetch counts
    frac_pairwise = counter_pairwise.get_counts()[1]
    frac_gt = counter_gt.get_counts()[1]
    frac_pred = counter_pred.get_counts()[1]

    # normalize to probabilities
    frac_pairwise = frac_pairwise.astype(np.double) / frac_pairwise.sum()
    frac_gt = frac_gt.astype(np.double) / frac_gt.sum()
    frac_pred = frac_pred.astype(np.double) / frac_pred.sum()

    return Rand(frac_pairwise, frac_gt, frac_pred, 0.5)

def Rand_membrane_prob(im_pred, im_gt):
    Rand_score = []
    for thresh in np.arange(0,1,0.05):
        # white regions, black boundaries
        im_seg = im_pred>thresh
        # connected components
        seeds, nr_regions = mahotas.label(im_seg)
        result = quick_Rand(im_gt, seeds)        
        Rand_score.append(result)

    return np.max(Rand_score)

def run_evaluation_boundary_predictions(network_name):
    pathPrefix = './AC4_small/'
    img_gt_search_string = pathPrefix + 'labels/*.tif'
    img_pred_search_string = pathPrefix + 'boundaryProbabilities/'+network_name+'/*.tif'

    img_files_gt = sorted( glob.glob( img_gt_search_string ) )
    img_files_pred = sorted( glob.glob( img_pred_search_string ) )

    allVI = []
    allVI_split = []
    allVI_merge = []

    allRand = []
    allRand_split = []
    allRand_merge = []

    for i in xrange(np.shape(img_files_pred)[0]):
        print img_files_pred[i]
        im_gt = mahotas.imread(img_files_gt[i])
        im_pred = mahotas.imread(img_files_pred[i])
        im_pred = im_pred / 255.0

        VI_score = []
        VI_score_split = []
        VI_score_merge = []

        Rand_score = []
        Rand_score_split = []
        Rand_score_merge = []
    
        start_time = time.clock()

        for thresh in np.arange(0,1,0.05):
            # white regions, black boundaries
            im_seg = im_pred>thresh
            # connected components
            seeds, nr_regions = mahotas.label(im_seg)
            
            result = segmentation_metrics(im_gt, seeds, seq=False)   
            
            VI_score.append(result['VI']['F-score'])
            VI_score_split.append(result['VI']['split'])
            VI_score_merge.append(result['VI']['merge'])

            Rand_score.append(result['Rand']['F-score'])
            Rand_score_split.append(result['Rand']['split'])
            Rand_score_merge.append(result['Rand']['merge'])

        print "This took in seconds: ", time.clock() - start_time

        allVI.append(VI_score)
        allVI_split.append(VI_score_split)
        allVI_merge.append(VI_score_merge)

        allRand.append(Rand_score)
        allRand_split.append(Rand_score_split)
        allRand_merge.append(Rand_score_merge)
        
    with open(pathPrefix+network_name+'.pkl', 'wb') as file:
        cPickle.dump((allVI, allVI_split, allVI_merge, allRand, allRand_split, allRand_merge), file)
    

    # for i in xrange(len(allVI)):
    #     plt.plot(np.arange(0,1,0.05), allVI[i], 'g', alpha=0.5)
    # plt.plot(np.arange(0,1,0.05), np.mean(allVI, axis=0), 'r')
    # plt.show()

    
def run_evaluation_segmentations3D():
    # first test how to convert a great boundary segmentation quickly into 3d objects
    pathPrefix = './AC4/'
    img_gt_search_string = pathPrefix + 'labels/*.tif'
    img_pred_search_string = pathPrefix + 'boundaryProbabilities/IDSIA/*.tif'

    img_files_gt = sorted( glob.glob( img_gt_search_string ) )
    img_files_pred = sorted( glob.glob( img_pred_search_string ) )
    
    s = 100
    img_gt_volume = np.zeros((1024,1024,s))
    img_pred_volume = np.zeros((1024,1024,s))

    for i in xrange(s):
        print img_files_gt[i]
        # read image
        img_gt = mahotas.imread(img_files_gt[i])
        img_gt_volume[:,:,i] = img_gt
        # compute gradient to get perfect segmentation
        img_gt = np.gradient(img_gt)
        img_gt = np.sqrt(img_gt[0]**2 + img_gt[1]**2)
        #img_gt = mahotas.morph.erode(img_gt == 0)
        img_pred_volume[:,:,i] = img_gt == 0


    all_VI = []
    for i in xrange(20):
        print i
        if i>0:
            for j in xrange(s):
                img_pred_volume[:,:,j] = mahotas.morph.erode(img_pred_volume[:,:,j]>0)

    # connected component labeling
    print "labeling"
    seeds, nr_objects = mahotas.label(img_pred_volume)
    # compute scores
    print "computing metric"
    result = segmentation_metrics(img_gt_volume, seeds, seq=False)   
    print result
    all_VI.append(result['VI']['F-score'])
    return seeds

def plot_evaluations():
    pathPrefix = './AC4_small/'
    search_string = pathPrefix + '*.pkl'
    files = sorted( glob.glob( search_string ) )

    for i in xrange(np.shape(files)[0]):
        with open(files[i], 'r') as file:
            allVI, allVI_split, allVI_merge, allRand, allRand_split, allRand_merge = cPickle.load(file)
            # for ii in xrange(len(allVI)):
            #     plt.plot(np.arange(0,1,0.05), allVI[ii], colors[i]+'--', alpha=0.5)
            plt.plot(np.arange(0,1,0.05), np.mean(allRand, axis=0), label=files[i])
            #print "VI: ", files[i], np.max(np.mean(allVI, axis=0))
            print "Rand:", files[i], np.max(np.mean(allRand, axis=0))
    plt.title("Rand_info comparison - higher is better, bounded by 1")
    plt.xlabel("Threshold")
    plt.ylabel("Rand_info")
    plt.legend(loc="upper left")
    plt.show()


    # for i in xrange(np.shape(files)[0]):
    #     with open(files[i], 'r') as file:
    #         allVI, allVI_split, allVI_merge, allRand, allRand_split, allRand_merge = cPickle.load(file)
    #         # for ii in xrange(len(allVI)):
    #         #     plt.plot(allVI_split[ii], allVI_merge[ii], colors[i]+'--', alpha=0.5)
    #         plt.plot(np.mean(allVI_split, axis=0), np.mean(allVI_merge, axis=0), colors[i], label=files[i])
    # plt.xlabel("VI_split")
    # plt.ylabel("VI_merge")
    # #plt.legend()
    # plt.show()


    
def match_synapses_by_overlap(gt, detected, min_overlap_pct):
    '''Determine the best ground truth synapse for a detected synapse by overlap
    
    :param gt: the ground-truth labeling of the volume. 0 = not synapse,
               1+ are the labels for each synapse
    :param detected: the computer-generated labeling of the volume
    :param min_overlap_pct: the percentage of voxels that must overlap
               for the algorithm to consider two objects.
    
    The algorithm tries to maximize the number of overlapping voxels
    globally. It finds the overlap between each pair of gt and detected
    objects. The cost is the number of voxels uncovered by both, given the
    choice.
    
    There must be an alternative cost for each gt matching nothing and
    for each detected matching nothing. This is the area of the thing minus
    the min_overlap_pct so that anything matching less than the min_overlap_pct
    will match against nothing.
    
    Return two vectors. The first vector is the matching label in d for each
    gt label (with zero for "not a match"). The second vector is the matching
    label in gt for each detected label.
    '''
    gt_areas = np.bincount(gt.flatten())
    gt_areas[0] = 0
    #
    # gt_map is a map of the original label #s for the labels that are > 0
    #        We work with the gt_map indices, nto the label #s
    # gt_r_map goes the other way
    #
    gt_map = np.where(gt_areas > 0)[0]
    gt_r_map = np.zeros(len(gt_areas), int)
    n_gt = len(gt_map)
    gt_r_map[gt_map] = np.arange(n_gt)
    #
    # for detected...
    #
    d_areas = np.bincount(detected.flatten())
    d_map = np.where(d_areas > 0)[0]
    d_r_map = np.zeros(len(d_areas), int)
    n_d = len(d_map)
    d_r_map[d_map] = np.arange(n_d)
    #
    # Get the matrix of correspondences.
    #
    z, y, x = np.where((gt > 0) & (detected > 0))
    matrix = coo_matrix((np.ones(len(z), int), 
                         (gt_r_map[gt[z, y, x]], 
                          d_r_map[detected[z, y, x]])),
                        shape=(n_gt, n_d))
    matrix.sum_duplicates()
    matrix = matrix.toarray()
    #
    # Enforce minimum overlap
    #
    d_min_overlap = d_areas * min_overlap_pct / 100
    gt_min_overlap = gt_ares * min_overlap_pct / 100
    bad_gt, bad_d = np.where((matrix < gt_min_overlap[:, np.newaxis]) |
                             (matrix < d_min_overlap[np.newaxis, :]))
    matrix[bad_gt, bad_d] = np.inf
    #
    # The score of each cell is the number of voxels in each cell minus
    # double the overlap - the amount of voxels covered in each map by
    # the overlap.
    #
    matrix = \
        gt_areas[gt_map][:, np.newaxis] +\
        d_areas[d_map][np.newaxis, :] -\
        matrix
    #
    # The alternative is that the thing matches nothing. We augment
    # the matrix with alternatives for each object, for instance:
    #
    # DA3 inf inf x   0    0
    # DA2 inf x   inf 0    0
    # DA1 x   inf inf 0    0
    # G2  y   y   y   inf  x
    # G1  y   y   y   x    inf
    #     D1  D2  D3  GA1  GA2
    #
    # x is the area of the thing * (1 - min_pct_overlap)
    # y is the area of both things - 2x overlap
    #
    big_matrix = np.zeros((n_gt+n_d, n_gt+n_d), np.float32)
    big_matrix[:n_gt, :n_d] = matrix
    big_matrix[n_gt:, :n_d] = np.inf
    big_matrix[:n_gt, n_d:] = np.inf
    big_matrix[n_gt+np.arange(n_d), np.arange(n_d)] = d_areas[d_map]
    big_matrix[np.arange(n_gt), n_d+np.arange(n_gt)] = gt_areas[gt_map]
    #
    # Solve it
    #
    d_match, gt_match = hungarian.lap(big_matrix)
    #
    # Get rid of the augmented results
    #
    d_match = d_match[:n_gt]
    gt_match = gt_match[:n_d]
    #
    # The gt with matches in d have d not in the alternative range
    #
    gt_winners = np.where(d_match < n_d)[0]
    gt_result = np.zeros(len(gt_areas), int)
    gt_result[gt_map[gt_winners]] = d_map[d_match[gt_winners]]
    #
    # Same for d
    #
    d_winners = np.where(gt_match < n_gt)[0]
    d_result = np.zeros(len(d_areas), int)
    d_result[d_map[d_winners]] = gt_map[gt_match[d_winners]]
    
    return gt_result, d_result

def match_synapses_by_distance(gt, detected, xy_nm, z_nm, max_distance):
    '''Match the closest pairs of ground-truth and detected synapses
    
    :param gt: a label volume of the ground-truth synapses
    :param detected: a label volume of the detected synapses
    :param xy_nm: size of voxel in the x/y direction
    :param z_nm: size of voxel in the z direction
    :param max_distance: maximum allowed distance for a match
    
    Centroids are calculated for each object and pairwise distances
    are calculated for each object. These are fed into a global optimization
    which tries to find the matching of gt with detected that results
    in the minimum distance.
    
    An alternative is proposed for each object that is the maximum distance
    and all pairs greater than the maximum distance are given a distance
    of infinity. This enforces the max_distance constraint.
    '''
    z, y, x = np.mgrid[0:gt.shape[0], 0:gt.shape[1], 0:gt.shape[2]]
    areas = np.bincount(gt.flatten())
    areas[0] = 0
    n_gt_orig = len(areas)
    gt_map = np.where(areas > 0)[0]
    n_gt = len(gt_map)
    xc_gt = np.bincount(gt.flatten(), x.flatten())[gt_map] / areas[gt_map]
    yc_gt = np.bincount(gt.flatten(), y.flatten())[gt_map] / areas[gt_map]
    zc_gt = np.bincount(gt.flatten(), z.flatten())[gt_map] / areas[gt_map]
    
    areas = np.bincount(detected.flatten())
    areas[0] = 0
    n_d_orig = len(areas)
    d_map = np.where(areas > 0)[0]
    n_d = len(d_map)
    xc_d = np.bincount(detected.flatten(), x.flatten())[d_map] / areas[d_map]
    yc_d = np.bincount(detected.flatten(), y.flatten())[d_map] / areas[d_map]
    zc_d = np.bincount(detected.flatten(), z.flatten())[d_map] / areas[d_map]
    
    matrix = np.sqrt(
        ((xc_gt[:, np.newaxis] - xc_d[np.newaxis, :]) * xy_nm) ** 2 +
        ((yc_gt[:, np.newaxis] - yc_d[np.newaxis, :]) * xy_nm) ** 2 +
        ((zc_gt[:, np.newaxis] - zc_d[np.newaxis, :]) * z_nm) ** 2)
    matrix[matrix > max_distance] = np.inf
    #
    # The alternative is that the thing matches nothing. We augment
    # the matrix with alternatives for each object, for instance:
    #
    # DA3 inf inf x   0    0
    # DA2 inf x   inf 0    0
    # DA1 x   inf inf 0    0
    # G2  y   y   y   inf  x
    # G1  y   y   y   x    inf
    #     D1  D2  D3  GA1  GA2
    #
    big_matrix = np.zeros((n_gt+n_d, n_gt+n_d), np.float32)
    big_matrix[:n_gt, :n_d] = matrix
    big_matrix[n_gt:, :n_d] = np.inf
    big_matrix[:n_gt, n_d:] = np.inf
    big_matrix[n_gt+np.arange(n_d), np.arange(n_d)] = max_distance
    big_matrix[np.arange(n_gt), n_d+np.arange(n_gt)] = max_distance
    
    #
    # Solve it
    #
    d_match, gt_match = hungarian.lap(big_matrix)
    #
    # Get rid of the augmented results
    #
    d_match = d_match[:n_gt]
    gt_match = gt_match[:n_d]
    #
    # The gt with matches in d have d not in the alternative range
    #
    gt_winners = np.where(d_match < n_d)[0]
    gt_result = np.zeros(n_gt_orig, int)
    gt_result[gt_map[gt_winners]] = d_map[d_match[gt_winners]]
    #
    # Same for d
    #
    d_winners = np.where(gt_match < n_gt)[0]
    d_result = np.zeros(n_d_orig, int)
    d_result[d_map[d_winners]] = gt_map[gt_match[d_winners]]
    
    return gt_result, d_result
