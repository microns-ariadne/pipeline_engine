'''Create triplets of synapses and their axons and dendrites'''

import json
import luigi
import numpy as np
from scipy.ndimage import grey_dilation, grey_erosion
from scipy.sparse import coo_matrix

from ..parameters import VolumeParameter, DatasetLocationParameter
from ..targets.factory import TargetFactory
from .utilities import RequiresMixin, RunMixin, SingleThreadedMixin

class ConnectSynapsesTaskMixin:
    
    volume = VolumeParameter(
        description="The volume to search for connections")
    neuron_seg_location = DatasetLocationParameter(
        description="The location of the segmented neuron dataset")
    synapse_seg_location = DatasetLocationParameter(
        description="The location of the segmented synapses")
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
        neuron_target, synapse_target = list(self.input())
        synapse = synapse_target.imread()
        #
        # Use a rectangular structuring element for speed.
        #
        strel = np.ones((self.z_dilation * 2 + 1,
                         self.xy_dilation * 2 + 1,
                         self.xy_dilation * 2 + 1), bool)
        grey_dilation(synapse, footprint=strel, output=synapse,
                      mode='constant', cval=0)
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
            grey_dilation(synapse, footprint=strel, mode='constant', cval=0) !=\
            grey_erosion(synapse, footprint=strel, mode='constant', cval=255)
        #
        # Extract only the overlapping pixels from the neurons and synapses
        #
        neuron = neuron_target.imread()
        mask = (synapse != 0) & (neuron != 0) & mask
        svoxels = synapse[mask]
        nvoxels = neuron[mask]
        del synapse
        del neuron
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
        else:
            neuron_1 = neuron_2 = synapses = np.zeros(0, int)
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