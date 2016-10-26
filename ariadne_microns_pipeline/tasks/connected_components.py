'''Tasks for finding connected components across blocks'''

import enum
import json
import luigi
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .utilities import RequiresMixin, RunMixin, SingleThreadedMixin
from ..parameters import VolumeParameter, DatasetLocationParameter
from ..parameters import MultiVolumeParameter
from ..targets.factory import TargetFactory
from ..targets.volume_target import VolumeTarget
from .utilities import to_hashable

class LogicalOperation(enum.Enum):
    OR = 1
    AND = 2

class ConnectedComponentsTaskMixin:
    
    volume1 = VolumeParameter(
        description="The volume for the first of the two segmentations")
    location1 = DatasetLocationParameter(
        description="The location of the first of the two segmentations")
    volume2 = VolumeParameter(
        description="The volume for the second of the two segmentations")
    location2 = DatasetLocationParameter(
        description="The location of the second of the two segmentations")
    overlap_volume = VolumeParameter(
        description="Look at the concordance between segmentations "
        "in this volume")
    output_location = luigi.Parameter(
        description=
        "The location for the JSON file containing the concordances")

    def input(self):
        tf = TargetFactory()
        yield tf.get_volume_target(self.location1, self.volume1)
        yield tf.get_volume_target(self.location2, self.volume2)
    
    def output(self):
        return luigi.LocalTarget(self.output_location)
    
class ConnectedComponentsRunMixin:
    
    min_overlap_percent = luigi.FloatParameter(
        default=50,
        description="Minimum amount of percent overlapping voxels when joining "
                    "two segments relative to the areas of each of the "
                    "segments in the overlap volume.")
    operation = luigi.EnumParameter(
        enum=LogicalOperation,
        default=LogicalOperation.OR,
        description="Whether to join if either objects overlap the other "
                    "by the minimum amount (""OR"") or whether they both "
                    "have to overlap the other by the minimum amount (""AND"")")
    
    def ariadne_run(self):
        '''Look within the overlap volume to find the concordances
        
        We load the two volumes and note the unique pairs of labels
        that appear together at the same voxel for the voxels in the
        overlap volume.
        '''
        volume1, volume2 = list(self.input())
        cutout1 = volume1.imread_part(self.overlap_volume.x,
                                      self.overlap_volume.y,
                                      self.overlap_volume.z,
                                      self.overlap_volume.width,
                                      self.overlap_volume.height,
                                      self.overlap_volume.depth)
        
        cutout2 = volume2.imread_part(self.overlap_volume.x,
                                      self.overlap_volume.y,
                                      self.overlap_volume.z,
                                      self.overlap_volume.width,
                                      self.overlap_volume.height,
                                      self.overlap_volume.depth)
        #
        # Order the two cutouts by first segmentation, then second.
        # Sort them using the order and then take only indices i where
        # cutouts[i] != cutouts[i+1]
        #
        cutouts = np.column_stack((cutout1.ravel(), cutout2.ravel()))
        #
        # Remove any pixels labeled with "0"
        #
        cutouts = cutouts[np.all(cutouts != 0, 1)]
        
        if len(cutouts) > 0:
            matrix = coo_matrix((np.ones(cutouts.shape[0], int),
                                 (cutouts[:, 0], cutouts[:, 1])))
            matrix.sum_duplicates()
            a, b = matrix.nonzero()
            counts = matrix.tocsr()[a, b].getA1().astype(int)
            if self.min_overlap_percent > 0:
                frac = float(self.min_overlap_percent) / 100
                cutout_a_area = np.bincount(cutouts[:, 0]).astype(float)
                cutout_b_area = np.bincount(cutouts[:, 1]).astype(float)
                frac_a = counts.astype(float) / cutout_a_area[a]
                frac_b = counts.astype(float) / cutout_b_area[b]
                if self.operation == LogicalOperation.OR:
                    large_enough = \
                        np.where((frac_a >= frac ) | (frac_b >= frac))[0]
                else:
                    large_enough = \
                        np.where((frac_a >= frac ) & (frac_b >= frac))[0]
                    
                counts, a, b = [_[large_enough] for _ in counts, a, b]
            as_list = [ (int(aa), int(bb)) for aa, bb in zip(a, b)]
        else:
            as_list = []
            counts = np.zeros(0, int)
        d = dict(connections=as_list, counts=counts.tolist())
        for volume, name in ((self.volume1, "1"),
                             (self.volume2, "2"),
                             (self.overlap_volume, "overlap")):
            d[name] = dict(x=volume.x,
                           y=volume.y,
                           z=volume.z,
                           width=volume.width,
                           height=volume.height,
                           depth=volume.depth)
        #
        # Compute the areas and find the labels with associated voxels
        #
        areas = np.bincount(volume1.imread().ravel())
        unique = np.where(areas)[0]
        unique = unique[unique != 0]
        areas = areas[unique]
        d["1"]["labels"] = unique.tolist()
        d["1"]["areas"] = areas.tolist()
        areas = np.bincount(volume2.imread().ravel())
        unique = np.where(areas)[0]
        unique = unique[unique != 0]
        d["2"]["labels"] = unique.tolist()
        d["2"]["areas"] = areas.tolist()
        with self.output().open("w") as fd:
            json.dump(d, fd)

class ConnectedComponentsTask(ConnectedComponentsTaskMixin,
                              ConnectedComponentsRunMixin,
                              RequiresMixin, RunMixin,
                              SingleThreadedMixin,
                              luigi.Task):
    '''This task finds the connections between the segmentations of two volumes
    
    Given segmentation #1 and segmentation #2 and an overlapping volume
    look at the labels in segmentation #1 and #2 at each pixel. These are
    the labels that are connected between the volumes. This task finds the
    unique labels between the segmentations and stores them in a JSON file.
    
    The connected components of the whole volume can be found by loading
    all of the JSON files and assigning each block's labels to a global
    label that may be shared between segments.
    '''
    
    task_namespace = 'ariadne_microns_pipeline'


class AllConnectedComponentsTaskMixin:
    
    input_locations = luigi.ListParameter(
        description="The filenames of the output files from the "
        "ConnectedComponentsTask")
    output_location = luigi.Parameter(
        description="The filename of the global assignment")
    
    def input(self):
        for input_location in self.input_locations:
            yield luigi.LocalTarget(input_location)
    
    def output(self):
        return luigi.LocalTarget(self.output_location)

class AllConnectedComponentsRunMixin:
    
    
    def ariadne_run(self):
        connections = []
        mappings = {}
        offset = 0
        #
        # The key here is to give each component its own global #
        # "mappings" is a dictionary with key=block's volume
        #            and a value that is an array mapping each local
        #            component to a global #
        # "connections" is a list of global mappings
        #
        for i, input_target in enumerate(self.input()):
            d = json.load(input_target.open("r"))
            c = np.array(d["connections"])
            l1 = d["1"]["labels"]
            l2 = d["2"]["labels"]
            for k1 in "1", "2":
                for k2 in "labels", "areas":
                    if k2 in d[k1]:
                        del d[k1][k2]
            d["pathname"] = input_target.path
            k1 = to_hashable(d["1"])
            k2 = to_hashable(d["2"])
            for k, l in (k1, l1), (k2, l2):
                if k not in mappings:
                    global_labels = np.arange(offset, offset + len(l))
                    m = np.column_stack((l, global_labels))
                    offset += len(l)
                    mappings[k] = m
                    #
                    # Connect self to self.
                    #
                    connections.append(np.column_stack([global_labels]*2))
                    
            rm1 = np.zeros(np.max(l1)+1, int)
            m1 = mappings[k1]
            rm1[m1[:, 0]] = m1[:, 1]
            rm2 = np.zeros(np.max(l2)+1, int)
            m2 = mappings[k2]
            rm2[m2[:, 0]] = m2[:, 1]
            #
            # Connect backward and forward
            #
            if len(c) > 0:
                connections.append(
                    np.column_stack([rm1[c[:, 0]], rm2[c[:, 1]]]))
                connections.append(
                    np.column_stack([rm2[c[:, 1]], rm1[c[:, 0]]]))
        #
        # Now run connected components on the adjacency graph
        #
        connections = np.vstack(connections)
        g = coo_matrix((np.ones(connections.shape[0], int), 
                        (connections[:, 0], connections[:, 1])))
        n_components, labels = connected_components(g, directed=False)
        #
        # Rebase labels starting at 1, leaving zero = background
        #
        labels += 1
        #
        # Write a json dictionary with the volume keys as keys and
        # a mapping from local to global as values
        #
        d = {}
        d["count"] = n_components
        d["volumes"] = []
        for volume, m in mappings.items():
            gm = [ (int(a), int(b)) for a, b in zip(m[:, 0], labels[m[:, 1]])]
            d["volumes"].append((dict(volume), gm))
        with self.output().open("w") as fd:
            json.dump(d, fd)


class AllConnectedComponentsTask(AllConnectedComponentsTaskMixin,
                                 AllConnectedComponentsRunMixin,
                                 RequiresMixin, RunMixin,
                                 luigi.Task):
    '''Perform all connected components on the component connections
    
    The ConnectedComponentsTask finds the associations between the components
    in adjacent blocks. AllConnectedComponentsTask takes these
    associations and finds a global linking of all connected components,
    arriving at a global label assignment for each connected component.
    
    The output is a dictionary of two tuples. The key of the dictionary
    is the volume and the value is a list of two-tuples where the first
    element of the two-tuple is the local label within that volume and
    the second element is the global label.
    
    In addition, the dictionary has a key of "count" which is the number
    of global labels.
    '''
    
    task_namespace="ariadne_microns_pipeline"

class VolumeRelabelingTaskMixin:
    
    input_volumes = MultiVolumeParameter(
        description="Input volumes to be composited together")
    relabeling_location = luigi.Parameter(
        description=
        "The location of the output file from AllConnectedComponentsTask "
        "that gives the local/global relabeling of the segmentation")
    output_volume = VolumeParameter(
        description="The volume of the output segmentation")
    output_location = DatasetLocationParameter(
        description="The location of the output segmentation")
    
    def input(self):
        '''Return the volumes to be assembled'''
        yield luigi.LocalTarget(self.relabeling_location)
        tf = TargetFactory()
        for d in self.input_volumes:
            yield tf.get_volume_target(d["location"], d["volume"])
    
    def output(self):
        '''Return the volume target that will be written'''
        tf = TargetFactory()
        return tf.get_volume_target(self.output_location, self.output_volume)


class VolumeRelabelingRunMixin:
    
    def ariadne_run(self):
        '''Composite and relabel each of the volumes'''
        output_result = np.zeros((self.output_volume.depth,
                                  self.output_volume.height,
                                  self.output_volume.width),
                                 np.uint32)
        output_volume_target = self.output()
        assert isinstance(output_volume_target, VolumeTarget)
        x0 = output_volume_target.x
        x1 = x0 + output_volume_target.width
        y0 = output_volume_target.y
        y1 = y0 + output_volume_target.height
        z0 = output_volume_target.z
        z1 = z0 + output_volume_target.depth
        generator = self.input()
        #
        # Read the local->global mappings
        #
        with generator.next().open("r") as fd:
            mappings = json.load(fd)
        #
        # Make a dictionary of all the candidate volumes
        #
        volumes = {}
        for volume in generator:
            key = to_hashable(dict(x=volume.x,
                                   y=volume.y,
                                   z=volume.z,
                                   width=volume.width,
                                   height=volume.height,
                                   depth=volume.depth))
            volumes[key] = volume
        #
        # For each volume in the mappings, map local->global 
        for volume, mapping in mappings["volumes"]:
            volume = to_hashable(volume)
            if volume not in volumes:
                continue
            vx0 = max(x0, volume["x"])
            vx1 = min(x1, volume["x"] + volume["width"])
            vy0 = max(y0, volume["y"])
            vy1 = min(y1, volume["y"] + volume["height"])
            vz0 = max(z0, volume["z"])
            vz1 = min(z1, volume["z"] + volume["depth"])
            if vx0 >= vx1 or\
               vy0 >= vy1 or\
               vz0 >= vz1:
                continue
            input_volume_target = volumes[volume]
            mapping_idxs = np.array(mapping, np.uint32)
            mapping_xform = np.zeros(mapping_idxs[:, 0].max()+1, np.uint32)
            mapping_xform[mapping_idxs[:, 0]] = mapping_idxs[:, 1]
            labels = input_volume_target.imread_part(
                vx0, vy0, vz0,
                vx1-vx0, vy1 - vy0, vz1 - vz0)
            labels = mapping_xform[labels]
            output_volume_target.imwrite_part(labels, vx0, vy0, vz0)
        output_volume_target.finish_volume()

class VolumeRelabelingTask(VolumeRelabelingTaskMixin,
                           VolumeRelabelingRunMixin,
                           RequiresMixin, RunMixin,
                           luigi.Task):
    '''Relabel a segmentation volume
    
    The VolumeRelabilingTask uses the output of the AllConnectedComponents
    to map the local segmentations of several input volumes into a single
    unitary output volume.
    '''

class ConnectivityGraph(object):
    '''A wrapper around the .json file generated by AllConnectedComponentsTask
    
    To use:
    
         with tgt.output().open("r") as fd:
             c = ConnectivityGraph.load(fd)
             global_seg = c.convert(local_seg, volume)
    '''
    @staticmethod
    def load(fd):
        '''Load a connectivity graph from JSON
        
        fd: the handle of a .json file produced by AllConnectedComponentsTask
        '''
        self = ConnectivityGraph()
        self.volumes = {}
        mappings = json.load(fd)
        for volume, mapping in mappings["volumes"]:
            self.volumes[to_hashable(volume)] = np.array(mapping)
        return self
    
    def convert(self, segmentation, volume):
        '''Convert a local segmentation to a global one
        
        segmentation: the segmentation with local labels
        volume: the segmentation's global volume coordinates - a Volume object
        '''
        key = to_hashable(volume.to_dictionary())
        mappings = self.volumes[key]
        t = np.zeros(max(mappings[:, 0].max(), segmentation.max())+1,
                     segmentation.dtype)
        t[mappings[:, 0]] = mappings[:, 1]
        return t[segmentation]
        