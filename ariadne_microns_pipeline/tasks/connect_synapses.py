'''Create triplets of synapses and their axons and dendrites'''

import json
import luigi
import numpy as np
import rh_logger
from scipy.ndimage import grey_dilation, grey_erosion, center_of_mass
from scipy.sparse import coo_matrix

from ..parameters import VolumeParameter, DatasetLocationParameter, \
     EMPTY_DATASET_LOCATION, Volume, is_empty_dataset_location
from ..targets.factory import TargetFactory
from .utilities import RequiresMixin, RunMixin, SingleThreadedMixin
from .connected_components import ConnectivityGraph

class ConnectSynapsesTaskMixin:
    
    volume = VolumeParameter(
        description="The volume to search for connections")
    neuron_seg_location = DatasetLocationParameter(
        description="The location of the segmented neuron dataset")
    synapse_seg_location = DatasetLocationParameter(
        description="The location of the segmented synapses")
    transmitter_probability_map_location = DatasetLocationParameter(
        default=EMPTY_DATASET_LOCATION,
        description="The location of the voxel probabilities of being "
                    "the transmitter side of a synapse")
    receptor_probability_map_location = DatasetLocationParameter(
        default=EMPTY_DATASET_LOCATION,
        description="The location of the voxel probabilities of being "
                    "the receptor side of a synapse")
    output_location = luigi.Parameter(
        description="Where to write the .json file containing the triplets")
    
    def input(self):
        tf = TargetFactory()
        yield tf.get_volume_target(
            location=self.neuron_seg_location,
            volume = self.volume)
        yield tf.get_volume_target(
            location=self.synapse_seg_location,
            volume = self.volume)
        if self.transmitter_probability_map_location != EMPTY_DATASET_LOCATION:
            yield tf.get_volume_target(
                location=self.transmitter_probability_map_location,
                volume=self.volume)
            assert self.receptor_probability_map_location !=\
                   EMPTY_DATASET_LOCATION
            yield tf.get_volume_target(
                location=self.receptor_probability_map_location,
                volume=self.volume)
    
    def output(self):
        return luigi.LocalTarget(self.output_location)

    def estimate_memory_usage(self):
        '''Return an estimate of bytes of memory required by this task'''
        v1 = np.prod([1888, 1416, 70])
        m1 = 4008223 * 1000
        v2 = np.prod([1888, 1416, 42])
        m2 = 2294840 * 1000
        #
        # Model is Ax + B where x is the output volume
        #
        B = (v1 * m2 - v2 * m1) / (v1 - v2)
        A = (float(m1) - B) / v1
        v = np.prod([self.volume.width, 
                     self.volume.height, 
                     self.volume.depth])
        return int(A * v + B)

class ConnectSynapsesRunMixin:
    
    xy_dilation = luigi.IntParameter(
        default=3,
        description="Amount to dilate each synapse in the x/y direction")
    z_dilation = luigi.IntParameter(
        default=0,
        description="Amount to dilate each synapse in the z direction")
    min_contact = luigi.IntParameter(
        default=25,
        description="Minimum acceptable overlap between neurite and synapse "
                    "border.")
    wants_edge_contact = luigi.BoolParameter(
        description="If true, only count pixels along the edge of the "
        "synapse, otherwise consider overlap between the whole synapse "
        "and neurons")
    
    def report_empty_result(self):
        '''Report a result with no synapses.'''
        result = dict(volume=self.volume.to_dictionary(),
                      neuron_1=[],
                      neuron_2=[],
                      synapse=[],
                      synapse_centers=dict(x=[], y=[], z=[]))
        if not is_empty_dataset_location(self.transmitter_probability_map_location):
            result["transmitter_score_1"] = []
            result["transmitter_score_2"] = []
        with self.output().open("w") as fd:
            json.dump(result, fd)
        
    def ariadne_run(self):
        #
        # The heuristic for matching synapses with neurites
        #
        # 0) Dilate the synapses
        # 1) Remove all interior pixels from synapses.
        # 2) Count synapse / neurite overlaps
        # 3) Pick two best neurites and discard synapses with < 2
        #
        # Removing the interior pixels favors neurites with broad and
        # shallow contacts with synapses and disfavors something that
        # intersects a corner heavily.
        #
        # Synapses are sparse - we can perform a naive dilation of them
        # without worrying about running two of them together.
        #
        inputs = self.input()
        neuron_target = inputs.next()
        synapse_target = inputs.next()
        if self.transmitter_probability_map_location == EMPTY_DATASET_LOCATION:
            transmitter_target = None
            receptor_target = None
        else:
            transmitter_target = inputs.next()
            receptor_target = inputs.next()
        synapse = synapse_target.imread()
        #
        # get the centers of the synapses for reference
        #
        n_synapses = np.max(synapse) + 1
        if n_synapses == 1:
            # There are none, return an empty result
            self.report_empty_result()
            return
            
        synapse_centers = np.array(
            center_of_mass(np.ones(synapse.shape, np.uint8),
                           synapse, np.arange(1, n_synapses)),
            np.uint32)
        #
        # There is/may be a bug here if there is only a single synapse in
        # the volume. The result is transposed.
        #
        if synapse_centers.shape[0] != 3:
            synapse_centers=synapse_centers.transpose()
        synapse_centers[0] += self.volume.z
        synapse_centers[1] += self.volume.y
        synapse_centers[2] += self.volume.x
        #
        # Use a rectangular structuring element for speed.
        #
        strel = np.ones((self.z_dilation * 2 + 1,
                         self.xy_dilation * 2 + 1,
                         self.xy_dilation * 2 + 1), bool)
        grey_dilation(synapse, footprint=strel, output=synapse,
                      mode='constant', cval=0)
        if self.wants_edge_contact:
            #
            # Remove the interior (connected to self on 6 sides)
            #
            strel = np.array([[[False, False, False],
                               [False, True, False],
                               [False, False, False]],
                              [[False, True, False],
                               [True, True, True],
                               [False, True, False]],
                              [[False, False, False],
                               [False, True, False],
                               [False, False, False]]])
            mask = \
                grey_dilation(
                    synapse, footprint=strel, mode='constant', cval=0) !=\
                grey_erosion(
                    synapse, footprint=strel, mode='constant', cval=255)
        else:
            mask = True
        #
        # Extract only the overlapping pixels from the neurons and synapses
        #
        neuron = neuron_target.imread()
        volume_mask = (synapse != 0) & (neuron != 0) & mask
        svoxels = synapse[volume_mask]
        nvoxels = neuron[volume_mask]
        if len(nvoxels) > 0:
            #
            # Make a matrix of counts of voxels in both synapses and neurons
            # then extract synapse / neuron matches
            #
            matrix = coo_matrix(
                (np.ones(len(nvoxels), int), (svoxels, nvoxels)))
            matrix.sum_duplicates()
            synapse_labels, neuron_labels = matrix.nonzero()
            counts = matrix.tocsr()[synapse_labels, neuron_labels].getA1()
            #
            # Filter neurons with too little overlap
            #
            mask = counts >= self.min_contact
            counts, neuron_labels, synapse_labels = [
                _[mask] for _ in counts, neuron_labels, synapse_labels]
            #
            # Order by synapse label and -count to get the neurons with
            # the highest count first
            #
            order = np.lexsort((-counts, synapse_labels))
            counts, neuron_labels, synapse_labels = \
                [_[order] for _ in counts, neuron_labels, synapse_labels]
            first = np.hstack(
                [[True], synapse_labels[:-1] != synapse_labels[1:], [True]])
            idx = np.where(first)[0]
            per_synapse_counts = idx[1:] - idx[:-1]
            #
            # Get rid of counts < 2
            #
            mask = per_synapse_counts >= 2
            if not np.any(mask):
                # another way to get nothing.
                self.report_empty_result()
                return
            idx = idx[:-1][mask]
            #
            # pick out the first and second most overlapping neurons and
            # their synapse.
            #
            neuron_1 = neuron_labels[idx]
            synapses = synapse_labels[idx]
            neuron_2 = neuron_labels[idx+1]
            if transmitter_target != None:
                # put transmitters first and receptors second.
                transmitter_probs = transmitter_target.imread()
                receptor_probs = receptor_target.imread()
                #
                # Start by making a matrix to transform the map.
                #
                neuron_mapping = np.hstack(([0], neuron_1, neuron_2))
                matrix = coo_matrix(
                    (np.arange(len(idx)*2) + 1,
                     (np.hstack((neuron_1, neuron_2)),
                      np.hstack((synapses, synapses)))),
                    shape=(np.max(nvoxels)+1, np.max(svoxels) + 1)).tocsr()
                #
                # Convert the neuron / synapse map to the mapping labels
                #
                mapping_labeling = matrix[nvoxels, svoxels]
                #
                # Score each synapse / label overlap on both the transmitter
                # and receptor probabilities
                #
                areas = np.bincount(mapping_labeling.A1)
                transmitter_score = np.bincount(
                    mapping_labeling.A1, transmitter_probs[volume_mask])
                receptor_score = np.bincount(
                    mapping_labeling.A1, receptor_probs[volume_mask])
                total_scores = (transmitter_score - receptor_score) / areas
                score_1 = total_scores[1:len(idx)+1]
                score_2 = total_scores[len(idx)+1:]
                #
                # Flip the scores and neuron assignments if score_2 > score_1
                #
                flippers = score_2 > score_1
                score_1[flippers], score_2[flippers] = \
                    score_2[flippers], score_1[flippers]
                neuron_1[flippers], neuron_2[flippers] = \
                    neuron_2[flippers], neuron_1[flippers]
            #
            # Record the synapse coords. "synapse_centers" goes from 1 to
            # N so that is why we subtract 1 below.
            #
            synapse_center_dict = dict(
                x=synapse_centers[2, synapses-1].tolist(),
                y=synapse_centers[1, synapses-1].tolist(),
                z=synapse_centers[0, synapses-1].tolist())
        else:
            neuron_1 = neuron_2 = synapses = np.zeros(0, int)
            score_1, score_2 = np.zeros(0)
            synapse_center_dict = dict(x=[], y=[], z=[])
        volume = dict(x=self.volume.x,
                      y=self.volume.y,
                      z=self.volume.z,
                      width=self.volume.width,
                      height=self.volume.height,
                      depth=self.volume.depth)
        result = dict(volume=volume,
                      neuron_1=neuron_1.tolist(),
                      neuron_2=neuron_2.tolist(),
                      synapse=synapses.tolist(),
                      synapse_centers=synapse_center_dict)
        if transmitter_target != None:
            result["transmitter_score_1"] = score_1.tolist()
            result["transmitter_score_2"] = score_2.tolist()
        with self.output().open("w") as fd:
            json.dump(result, fd)

class ConnectSynapsesTask(ConnectSynapsesTaskMixin,
                          ConnectSynapsesRunMixin,
                          RequiresMixin,
                          RunMixin,
                          SingleThreadedMixin,
                          luigi.Task):
    '''Connect neurons to synapses'''
    
    task_namespace = "ariadne_microns_pipeline"

class AggregateSynapseConnectionsTaskMixin:
    
    synapse_connection_locations = luigi.ListParameter(
        description="A list of filenames of the synapse connection files "
                    "generated by the ConnectSynapsesTask.")
    connectivity_graph_location = luigi.Parameter(
        description = "The mapping between local IDs and global ones "
                      "produced by the AllConnectedComponentsTask.")
    output_location = luigi.Parameter(
        description = "The location of the aggregate file")
    
    def input(self):
        yield luigi.LocalTarget(self.connectivity_graph_location)
        for location in self.synapse_connection_locations:
            yield luigi.LocalTarget(location)
    
    def output(self):
        return luigi.LocalTarget(self.output_location)

class AggregateSynapseConnectionsRunMixin:
    
    def ariadne_run(self):
        inputs = self.input()
        cg_tgt = inputs.next()
        synapse_tgts = list(inputs)
        #
        # The connectivity graph for mapping neurite IDs
        #
        with cg_tgt.open("r") as fd:
            cg = ConnectivityGraph.load(fd)
        neuron_1 = []
        neuron_2 = []
        synapse_center_x = []
        synapse_center_y = []
        synapse_center_z = []
        for synapse_tgt in synapse_tgts:
            with synapse_tgt.open("r") as fd:
                synapse_dict = json.load(fd)
            volume = Volume(**synapse_dict["volume"])
            if len(synapse_dict["neuron_1"]) == 0:
                rh_logger.logger.report_event(
                    "No synapses found in volume, %d, %d, %d" % 
                    (volume.x, volume.y, volume.z))
                continue
            n1 = cg.convert(np.array(synapse_dict["neuron_1"]), volume)
            n2 = cg.convert(np.array(synapse_dict["neuron_2"]), volume)
            sx = np.array(synapse_dict["synapse_centers"]["x"])+volume.x
            sy = np.array(synapse_dict["synapse_centers"]["y"])+volume.y
            sz = np.array(synapse_dict["synapse_centers"]["z"])+volume.z
            neuron_1.append(n1)
            neuron_2.append(n2)
            synapse_center_x.append(sx)
            synapse_center_y.append(sy)
            synapse_center_z.append(sz)
        neuron_1, neuron_2, synapse_center_x, synapse_center_y, \
            synapse_center_z = [
                np.hstack(_).tolist() for _ in
                neuron_1, neuron_2, synapse_center_x, synapse_center_y,
                synapse_center_z]
        result = dict(
            neuron_1 = neuron_1,
            neuron_2 = neuron_2,
            synapse_center=dict(
                x=synapse_center_x,
                y=synapse_center_y,
                z=synapse_center_z))
        with self.output().open("w") as fd:
            json.dump(result, fd)

class AggregateSynapseConnectionsTask(
    AggregateSynapseConnectionsTaskMixin,
    AggregateSynapseConnectionsRunMixin,
    RequiresMixin,
    RunMixin,
    SingleThreadedMixin,
    luigi.Task):
    '''Aggregate the outputs of SynapseConnectionsTask
    
    This task generates a .json output with the following keys:
    
    neuron_1 - the global ID of the transmitter per synapse
    neuron_2 - the global ID of the receptor per synapse
    synapse_center - with keys of x, y, and z, the coordinates of the
               center of mass per synapse
    '''
    task_namespace="ariadne_microns_pipeline"