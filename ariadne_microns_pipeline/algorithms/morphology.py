'''Morphological operations'''

import multiprocessing
import numpy as np
import Queue
import threading
import rh_logger
from scipy.ndimage import grey_dilation, grey_erosion, distance_transform_edt

'''A 3d structuring element connecting only left/right top/bottom up/down'''
SIX_CONNECTED = np.array([[[False, False, False],
                           [False, True, False],
                           [False, False, False]],
                          [[False, True, False],
                           [True, True, True],
                           [False, True, False]],
                          [[False, False, False],
                           [False, True, False],
                           [False, False, False]]])

def erode_segmentation(segmentation, strel, in_place=False):
    '''Erode a segmentation using a structuring element
    
    :param segmentation: a labeling of a volume
    :param strel: the structuring element for the erosion. This should be
                  a boolean 3-d array with the voxels to erode marked as True
    :param in_place: True to erode the segmentation volume in-place, False
                     to return a new volume.
    '''
    if not in_place:
        segmentation = segmentation.copy()
    mask =\
        grey_dilation(segmentation, footprint=strel) != \
        grey_erosion(segmentation, footprint=strel)
    segmentation[mask] = 0
    return segmentation

def parallel_distance_transform(volume, xy_nm, z_nm, xy_overlap, z_overlap,
                                xy_block_size, z_block_size, n_cores):
    '''Perform the distance transform in chunks
    
    We overlap the edges of the blocks and take the minimum distance in
    the overlap area.
    
    volume - the binary volume on which to perform the distance transform
    xy_nm - the size of a voxel in the x and y direction in nm
    z_nm - the size of a voxel in the z direction
    xy_overlap - the # of voxels of overlap in the x and y directions
    z_overlap - the # of voxels of overlap in the z direction
    xy_block_size - the size of a block in the x and y directions
    z_block_size - the size of a block in the z direction
    n_cores - # of cores to run
    '''
    n_blocks_x = max(1, 1 + (volume.shape[2] - xy_block_size) / 
                     (xy_block_size - xy_overlap))
    x = np.linspace(0, volume.shape[2], n_blocks_x+1).astype(int)
    sx = x[:-1].copy()
    sx[1:] -= xy_overlap / 2
    ex = x[1:].copy()
    ex[:-1] += xy_overlap / 2
    n_blocks_y = max(1, 1 + (volume.shape[1] - xy_block_size) / 
                     (xy_block_size - xy_overlap))
    y = np.linspace(0, volume.shape[1], n_blocks_y+1).astype(int)
    sy = y[:-1].copy()
    sy[1:] = sy[1:] - xy_overlap / 2
    ey = y[1:].copy()
    ey[:-1] = ey[:-1] + xy_overlap / 2
    n_blocks_z = max(1, 1 + (volume.shape[0] - z_block_size) / 
                     (z_block_size - z_overlap))
    z = np.linspace(0, volume.shape[0], n_blocks_z+1).astype(int)
    sz = z[:-1].copy()
    sz[1:] -= z_overlap / 2
    ez = z[1:].copy()
    ez[:-1] += z_overlap / 2
    #
    # We build the result in this array. It's initialized to the maximum
    # possible distance.
    #
    result = np.ones(volume.shape, np.float32) * \
        min(volume.shape[0] * z_nm, volume.shape[1] * xy_nm, 
            volume.shape[2] * xy_nm)
    #
    # The queue for blocks to write
    #
    queue = Queue.Queue(n_cores)
    #
    # The thread function that writes blocks
    #
    def write(queue=queue, result=result):
        while True:
            dt, x0, x1, y0, y1, z0, z1 = queue.get()
            if dt is None:
                queue.task_done()
                break
            rh_logger.logger.report_event("Received block %d:%d %d:%d %d:%d" %
                                          (x0, x1, y0, y1, z0, z1))
            result[z0:z1, y0:y1, x0:x1] = \
                np.minimum(result[z0:z1, y0:y1, x0:x1], dt.get())
            queue.task_done()
    pool = multiprocessing.Pool(n_cores)
    thread = threading.Thread(target = write)
    thread.start()
    
    sampling = (z_nm, xy_nm, xy_nm)
    try:
        for z0, z1 in zip(sz, ez):
            for y0, y1 in zip(sy, ey):
                for x0, x1 in zip(sx, ex):
                    dt = pool.apply_async(
                        distance_transform_edt,
                        kwds=dict(input=volume[z0:z1, y0:y1, x0:x1],
                                  sampling=sampling))
                    queue.put((dt, x0, x1, y0, y1, z0, z1))
    finally:
        queue.put([None] * 7)
        queue.join()
        thread.join()
        pool.close()
    return result