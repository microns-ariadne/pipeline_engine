
'''The globaldict stores expensive structures for universal global access

During a fork operation, the memory-space is cloned, so the import of
boss_pipeline includes the structure at the time of the fork. This can be
used to cache an object that is universally used, but expensive to load.
'''
globaldict = {}

import json

def load_cached_json(path):
    if path not in globaldict:
        globaldict[path] = json.load(open(path))
    return globaldict[path]

all = [load_cached_json]