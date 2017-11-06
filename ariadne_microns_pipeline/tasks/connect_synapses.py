'''Create triplets of synapses and their axons and dendrites'''

import json
import luigi
import numpy as np
import rh_logger
from scipy.ndimage import grey_dilation, grey_erosion, center_of_mass
from scipy.ndimage import minimum_position
from scipy.sparse import coo_matrix
from scipy.spatial import KDTree
import time

from ..parameters import VolumeParameter, EMPTY_LOCATION, Volume
from ..targets import DestVolumeReader
from .utilities import RequiresMixin, RunMixin, SingleThreadedMixin
from .connected_components import ConnectivityGraph

class ConnectSynapsesTaskMixin:
    
    neuron_seg_load_plan_path = luigi.Parameter(
        description="The load plan for the segmentation of the neuron volume")
    synapse_seg_load_plan_path = luigi.Parameter(
        description="The load plan for the synapse segmentation")
    transmitter_probability_map_load_plan_path = luigi.Parameter(
        default=EMPTY_LOCATION,
        description="The load plan for probability of a voxel being "
                    "the transmitter side of a synapse")
    receptor_probability_map_load_plan_path = luigi.Parameter(
        default=EMPTY_LOCATION,
        description="The load plan for probability of a voxel being "
            "the receptor side of a synapse")
    output_location = luigi.Parameter(
        description="Where to write the .json file containing the triplets")
    
    def input(self):
        load_plans = [self.neuron_seg_load_plan_path,
                      self.synapse_seg_load_plan_path]
        if self.transmitter_probability_map_load_plan_path != EMPTY_LOCATION:
            assert self.receptor_probability_map_load_plan_path != \
                   EMPTY_LOCATION
            load_plans += [self.transmitter_probability_map_load_plan_path,
                           self.receptor_probability_map_load_plan_path]
        else:
            assert self.receptor_probability_map_load_plan_path == \
                   EMPTY_LOCATION
        for load_plan in load_plans:
            for tgt in DestVolumeReader(load_plan).get_source_targets():
                yield tgt
    
    def output(self):
        return luigi.LocalTarget(self.output_location)

    def estimate_memory_usage(self):
        '''Return an estimate of bytes of memory required by this task'''
        v1 = np.prod([1888, 1416, 70])
        m1 = 4008223 * 1000
        v2 = np.prod([1888, 1416, 42])
        m2 = 2294840 * 1000
        volume = DestVolumeReader(self.synapse_seg_load_plan_path).volume
        #
        # Model is Ax + B where x is the output volume
        #
        B = (v1 * m2 - v2 * m1) / (v1 - v2)
        A = (float(m1) - B) / v1
        v = np.prod([volume.width, 
                     volume.height, 
                     volume.depth])
        if self.transmitter_probability_map_load_plan_path != EMPTY_LOCATION:
            v *= 2
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
    x_nm = luigi.FloatParameter(
        default=4.0,
        description="size of a voxel in the x direction")
    y_nm = luigi.FloatParameter(
        default=4.0,
        description="size of a voxel in the y direction")
    z_nm = luigi.FloatParameter(
        default=30.0,
        description="size of a voxel in the z direction")
    distance_from_centroid = luigi.FloatParameter(
        default=70.0,
        description="Ideal distance from centroid marker of markers for "
                    "neuron positiions")
    
    def report_empty_result(self):
        '''Report a result with no synapses.'''
        neuron_target = DestVolumeReader(self.neuron_seg_load_plan_path)
        result = dict(volume=neuron_target.volume.to_dictionary(),
                      neuron_1=[],
                      neuron_2=[],
                      synapse=[],
                      synapse_centers=dict(x=[], y=[], z=[]))
        if self.transmitter_probability_map_load_plan_path != EMPTY_LOCATION:
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
        neuron_target = DestVolumeReader(self.neuron_seg_load_plan_path)
        synapse_target = DestVolumeReader(self.synapse_seg_load_plan_path)
        if self.transmitter_probability_map_load_plan_path == EMPTY_LOCATION:
            transmitter_target = None
            receptor_target = None
        else:
            transmitter_target = DestVolumeReader(
                self.transmitter_probability_map_load_plan_path)
            receptor_target = DestVolumeReader(
                self.receptor_probability_map_load_plan_path)
        synapse = synapse_target.imread()
        n_synapses = np.max(synapse) + 1
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
                tscore_1 = transmitter_score[1:len(idx)+1]
                tscore_2 = transmitter_score[len(idx)+1:]
                rscore_1 = receptor_score[1:len(idx)+1]
                rscore_2 = receptor_score[len(idx)+1:]
                #
                # Flip the scores and neuron assignments if score_2 > score_1
                #
                flippers = score_2 > score_1
                score_1[flippers], score_2[flippers] = \
                    score_2[flippers], score_1[flippers]
                neuron_1[flippers], neuron_2[flippers] = \
                    neuron_2[flippers], neuron_1[flippers]
                #
                # Compute the integrated transmitter score + receptor score
                # per synapse.
                #
                flippers_mult = flippers.astype(tscore_1.dtype)
                synapse_score = \
                    (tscore_1 + rscore_2) * (1 - flippers_mult) + \
                    (tscore_2 + rscore_1) * flippers_mult
            else:
                synapse_score = np.zeros(len(neuron_1))
            #
            # Recompute the centroids of the synapses based on where they
            # intersect the edge of neuron_1. This is closer to what people
            # do when they annotate synapses.
            #
            edge_z, edge_y, edge_x = np.where(
                (synapse != 0) & 
                (grey_dilation(neuron, size=3) != grey_erosion(neuron, size=3)))
            maxsynapses = np.max(synapse_labels)+1
            areas = np.bincount(synapse[edge_z, edge_y, edge_x], 
                                minlength=maxsynapses)
            xs, ys, zs = [
                np.bincount(synapse[edge_z, edge_y, edge_x], _,
                            minlength=maxsynapses)
                for _ in edge_x, edge_y, edge_z]
            xc = xs[synapses] / areas[synapses]
            yc = ys[synapses] / areas[synapses]
            zc = zs[synapses] / areas[synapses]
            #
            # Record the synapse coords. "synapse_centers" goes from 1 to
            # N so that is why we subtract 1 below.
            #
            synapse_center_dict = dict(
                x=xc.tolist(),
                y=yc.tolist(),
                z=zc.tolist())
            #
            # Compute the point in n1 that is closest to the synapse center
            #
            
            n1_per_synapse = np.zeros(maxsynapses, np.uint32)
            n1_per_synapse[synapses] = neuron_1
            idx_per_synapse = np.zeros(maxsynapses, np.uint32)
            idx_per_synapse[synapses] = np.arange(len(synapses))
            n1z, n1y, n1x = np.where(n1_per_synapse[synapse] == neuron)
            n1_idxs = idx_per_synapse[synapse[n1z, n1y, n1x]]
            d = np.sqrt(((n1z - zc[n1_idxs]) * self.z_nm)**2 +
                        ((n1y - yc[n1_idxs]) * self.y_nm)**2 +
                        ((n1x - xc[n1_idxs]) * self.x_nm)**2)
            n1_idx = np.array(
                minimum_position(np.abs(d - self.distance_from_centroid),
                                 synapse[n1z, n1y, n1x], 
                                 synapses)).flatten()
            xn1, yn1, zn1 = n1x[n1_idx], n1y[n1_idx], n1z[n1_idx]
            n1_center_dict = \
                dict(x=xn1.tolist(), y=yn1.tolist(), z=zn1.tolist())
            n2_per_synapse = np.zeros(maxsynapses, np.uint32)
            n2_per_synapse[synapses] = neuron_2
            n2z, n2y, n2x = np.where(n2_per_synapse[synapse] == neuron)
            n2_idxs = idx_per_synapse[synapse[n2z, n2y, n2x]]
            d = np.sqrt(((n2z - zc[n2_idxs]) * self.z_nm)**2 +
                        ((n2y - yc[n2_idxs]) * self.y_nm)**2 +
                        ((n2x - xc[n2_idxs]) * self.x_nm)**2)
            n2_idx = np.array(
                minimum_position(np.abs(d - self.distance_from_centroid), 
                                 synapse[n2z, n2y, n2x], 
                                 synapses)).flatten()
            xn2, yn2, zn2 = n2x[n2_idx], n2y[n2_idx], n2z[n2_idx]
            n2_center_dict = \
                dict(x=xn2.tolist(), y=yn2.tolist(), z=zn2.tolist())
        else:
            synapse_score = np.zeros(0, np.float32)
            neuron_1 = neuron_2 = synapses = np.zeros(0, int)
            score_1 = score_2 = np.zeros(0)
            synapse_center_dict = n1_center_dict = n2_center_dict = \
                dict(x=[], y=[], z=[])
        volume = dict(x=neuron_target.volume.x,
                      y=neuron_target.volume.y,
                      z=neuron_target.volume.z,
                      width=neuron_target.volume.width,
                      height=neuron_target.volume.height,
                      depth=neuron_target.volume.depth)
        result = dict(volume=volume,
                      neuron_1=neuron_1.tolist(),
                      neuron_2=neuron_2.tolist(),
                      synapse=synapses.tolist(),
                      score=synapse_score.tolist(),
                      synapse_centers=synapse_center_dict,
                      neuron_1_centers=n1_center_dict,
                      neuron_2_centers=n2_center_dict)
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
    
    xy_nm = luigi.FloatParameter(
        default=4.0,
        description="Size of a voxel in the X and Y directions")
    z_nm = luigi.FloatParameter(
        default=30.0,
        description="Size of a voxel in the Z direction")
    min_distance_nm = luigi.FloatParameter(
        default=200.0,
        description="Minimum allowable distance between a synapse in one "
        "volume and a synapse in another (otherwise merge them)")
    min_distance_identical_nm = luigi.FloatParameter(
        default=50.0,
        description="If two synapses are within this distance, they are "
        "treated as the same synapse, but in different blocks. They are "
        "eliminated on the basis of their position within the block instead "
        "of on their likelihood of being a synapse")
    
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
        score = []
        synapse_center_x = []
        synapse_center_y = []
        synapse_center_z = []
        neuron_1_center_x = []
        neuron_1_center_y = []
        neuron_1_center_z = []
        neuron_2_center_x = []
        neuron_2_center_y = []
        neuron_2_center_z = []
        volumes = []
        volume_idx = []
        for idx, synapse_tgt in enumerate(synapse_tgts):
            with synapse_tgt.open("r") as fd:
                synapse_dict = json.load(fd)
            volume = Volume(**synapse_dict["volume"])
            volumes.append(volume)
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
            n1x = np.array(synapse_dict["neuron_1_centers"]["x"])+volume.x
            n1y = np.array(synapse_dict["neuron_1_centers"]["y"])+volume.y
            n1z = np.array(synapse_dict["neuron_1_centers"]["z"])+volume.z
            n2x = np.array(synapse_dict["neuron_2_centers"]["x"])+volume.x
            n2y = np.array(synapse_dict["neuron_2_centers"]["y"])+volume.y
            n2z = np.array(synapse_dict["neuron_2_centers"]["z"])+volume.z
            neuron_1.append(n1)
            neuron_2.append(n2)
            score.append(np.array(synapse_dict["score"]))
            synapse_center_x.append(sx)
            synapse_center_y.append(sy)
            synapse_center_z.append(sz)
            neuron_1_center_x.append(n1x)
            neuron_1_center_y.append(n1y)
            neuron_1_center_z.append(n1z)
            neuron_2_center_x.append(n2x)
            neuron_2_center_y.append(n2y)
            neuron_2_center_z.append(n2z)
            volume_idx.append([idx] * len(n1))
        volume_idx, neuron_1, neuron_2, score, \
            synapse_center_x, synapse_center_y, synapse_center_z, \
            neuron_1_center_x, neuron_1_center_y, neuron_1_center_z, \
            neuron_2_center_x, neuron_2_center_y, neuron_2_center_z = \
            map(np.hstack, [
                volume_idx, neuron_1, neuron_2, score, 
                synapse_center_x, synapse_center_y, synapse_center_z,
                neuron_1_center_x, neuron_1_center_y, neuron_1_center_z,
                neuron_2_center_x, neuron_2_center_y, neuron_2_center_z])
        #
        # We pick the synapse farthest from the edge when eliminating.
        # The following code computes the distance to the edge
        #
        vx0, vx1, vy0, vy1, vz0, vz1 = [
            np.array([getattr(volume, _) for volume in volumes])
            for _ in "x", "x1", "y", "y1", "z", "z1"]
        sx, sy, vx0, vx1, vy0, vy1 = \
            [_ * self.xy_nm for _ in 
             synapse_center_x, synapse_center_y, vx0, vx1, vy0, vy1]
        sz, vz0, vz1 = \
            [np.array(_) * self.z_nm for _ in 
                     synapse_center_z, vz0, vz1]
        volume_idx = np.array(volume_idx)
        dx = np.minimum(sx - vx0[volume_idx], vx1[volume_idx] - sx)
        dy = np.minimum(sy - vy0[volume_idx], vy0[volume_idx] - sy)
        dz = np.minimum(sz - vz0[volume_idx], vz1[volume_idx] - sz)
        d_edge = np.sqrt(dx * dx + dy * dy + dz * dz)
        #
        # Create a KDTree, converting coordinates to nm and get pairs
        # closer than the allowed minimum inter-synapse distance.
        #
        t0 = time.time()
        kdtree = KDTree(np.column_stack((
            np.array(synapse_center_x) * self.xy_nm,
            np.array(synapse_center_y) * self.xy_nm,
            np.array(synapse_center_z) * self.z_nm)))
        rh_logger.logger.report_metric(
             "AggregateSynapseConnectionsTask.KDTreeBuildTime", 
             time.time() - t0)
        t0 = time.time()
        pairs = np.array(list(kdtree.query_pairs(self.min_distance_nm)))
        rh_logger.logger.report_metric(
                "AggregateSynapseConnectionsTask.KDTreeQueryPairsTime", 
                 time.time() - t0)
        #
        # Eliminate the duplicates.
        #
        if len(pairs) > 0:
            d_pair = np.sqrt(
                (sx[pairs[:, 0]] - sx[pairs[:, 1]]) ** 2 + 
                (sy[pairs[:, 0]] - sy[pairs[:, 1]]) ** 2 + 
                (sz[pairs[:, 0]] - sz[pairs[:, 1]]) ** 2)
            #
            # Use the edge distance if within min_distance_identical_nm,
            # otherwise, use the synapse score.
            #
            use_edge = d_pair <= self.min_distance_identical_nm
            
            first_is_best = \
                ((d_edge[pairs[:, 0]] > d_edge[pairs[:, 1]]) & use_edge) | \
                ((score[pairs[:, 0]] > score[pairs[:, 1]]) & ~ use_edge)
            to_remove = np.unique(np.hstack(
                 [pairs[first_is_best, 1], pairs[~ first_is_best, 0]]))
            neuron_1, neuron_2, score, \
                synapse_center_x, synapse_center_y, synapse_center_z, \
                neuron_1_center_x, neuron_1_center_y, neuron_1_center_z, \
                neuron_2_center_x, neuron_2_center_y, neuron_2_center_z = \
                [np.delete(_, to_remove) for _ in 
                 neuron_1, neuron_2, score,
                 synapse_center_x, synapse_center_y, synapse_center_z,
                 neuron_1_center_x, neuron_1_center_y, neuron_1_center_z,
                 neuron_2_center_x, neuron_2_center_y, neuron_2_center_z]
        #
        # Make the dictionaries.
        #
        neuron_1, neuron_2, score, \
            synapse_center_x, synapse_center_y, synapse_center_z, \
            neuron_1_center_x, neuron_1_center_y, neuron_1_center_z, \
            neuron_2_center_x, neuron_2_center_y, neuron_2_center_z = [
                _.tolist() for _ in
                neuron_1, neuron_2, score, 
                synapse_center_x, synapse_center_y, synapse_center_z,
                neuron_1_center_x, neuron_1_center_y, neuron_1_center_z,
                neuron_2_center_x, neuron_2_center_y, neuron_2_center_z]
        result = dict(
            neuron_1 = neuron_1,
            neuron_2 = neuron_2,
            synapse_center=dict(
                x=synapse_center_x,
                y=synapse_center_y,
                z=synapse_center_z),
            neuron_1_center=dict(
                x=neuron_1_center_x,
                y=neuron_1_center_y,
                z=neuron_1_center_z),
            neuron_2_center=dict(
                x=neuron_2_center_x,
                y=neuron_2_center_y,
                z=neuron_2_center_z)        )
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