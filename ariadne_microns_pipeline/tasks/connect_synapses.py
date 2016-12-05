'''Create triplets of synapses and their axons and dendrites'''

import json
import luigi
import numpy as np
from scipy.ndimage import grey_dilation, grey_erosion
from scipy.sparse import coo_matrix

from ..parameters import VolumeParameter, DatasetLocationParameter, \
     EMPTY_DATASET_LOCATION
from ..targets.factory import TargetFactory
from .utilities import RequiresMixin, RunMixin, SingleThreadedMixin

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
    wants_edge_contact = luigi.BooleanParameter(
        description="If true, only count pixels along the edge of the "
        "synapse, otherwise consider overlap between the whole synapse "
        "and neurons")
    
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
        else:
            neuron_1 = neuron_2 = synapses = np.zeros(0, int)
            score_1, score_2 = np.zeros(0)
        volume = dict(x=self.volume.x,
                      y=self.volume.y,
                      z=self.volume.z,
                      width=self.volume.width,
                      height=self.volume.height,
                      depth=self.volume.depth)
        result = dict(volume=volume,
                      neuron_1=neuron_1.tolist(),
                      neuron_2=neuron_2.tolist(),
                      synapse=synapses.tolist())
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