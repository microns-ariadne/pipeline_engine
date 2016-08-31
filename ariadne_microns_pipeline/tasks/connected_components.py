'''Tasks for finding connected components across blocks'''

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
        order = np.lexsort((cutouts[:, 1], cutouts[:, 0]))
        cutouts = cutouts[order]
        if len(cutouts) > 0:
            first = np.hstack(
                [[True], 
                 np.where(np.any(cutouts[:-1, :] != cutouts[1:, :], 1))[0]])
            unique = cutouts[first]
            as_list = [ (a, b) for a, b in unique.tolist()]
        else:
            as_list = []
        d = dict(connections=as_list)
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
        # TODO: get the # of components more cheaply than looking at every voxel
        #
        unique = np.unique(volume1.imread().ravel())
        unique = unique[unique != 0]
        d["1"]["labels"] = unique.tolist()
        unique = np.unique(volume2.imread().ravel())
        unique = unique[unique != 0]
        d["2"]["labels"] = unique.tolist()
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
            del d["1"]["labels"]
            del d["2"]["labels"]
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
                                 np.uint16)
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
            mapping_idxs = np.array(mapping, np.uint16)
            mapping_xform = np.zeros(mapping_idxs[:, 0].max()+1, np.uint16)
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