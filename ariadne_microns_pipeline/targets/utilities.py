import os
import hashlib

def shard(paths, x, y, z):
    '''Shard disk access by choosing one of several paths for a target
    
    The x, y and z coordinates are hashed together and the path chosen is
    the hash, modulo len(paths)
    
    :param paths: a sequence of paths, each of which should be on a different
        spindle
    :param x: the x coordinate of the volume being processed
    :param y: the y coordinate of the volume being processed
    :param z: the z coordinate of the volume being processed
    :returns: one of the paths
    '''
    h = hashlib.md5()
    h.update(",".join([str(_) for _ in x, y, z]))
    idx = (ord(h.digest()[0]) + 255 * ord(h.digest()[1])) % len(paths)
    return paths[idx]