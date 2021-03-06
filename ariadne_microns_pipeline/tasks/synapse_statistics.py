'''Calculate the confusion matrix of pairs of synapses'''

import cPickle
import json
import luigi
import numpy as np
import rh_logger
from scipy.sparse import coo_matrix

from .utilities import RunMixin, RequiresMixin, SingleThreadedMixin
from .utilities import to_hashable

class SynapseStatisticsTaskMixin:
    
    synapse_matches = luigi.ListParameter(
        description="The .json files containing ground-truth/synapse matches.")
    detected_synapse_connections = luigi.ListParameter(
        description="The .json files containing all connections between "
        "detected neurites and synapses")
    neuron_map = luigi.Parameter(
        description="The .json file that maps sub-block neuron IDs to "
                    "global ones")
    gt_neuron_maps = luigi.ListParameter(
        description="The .json file that maps detected neuron labels to "
                    "ground-truth neuron labels")
    gt_synapse_connections = luigi.ListParameter(
        description="The .json files containing connections between ground "
        "truth synapses and neurites")
    output_location = luigi.Parameter(
        description="The location for the confusion map")

    def input(self):
        yield luigi.LocalTarget(self.neuron_map)
        yield luigi.LocalTarget(self.gt_synapse_connections)
        for synapse_match, \
            neuron_match, \
            detected_connection, \
            gt_connection in zip(
                self.synapse_matches, 
                self.gt_neuron_maps,
                self.detected_synapse_connections, 
                self.gt_synapse_connections):
            yield luigi.LocalTarget(synapse_match)
            yield luigi.LocalTarget(neuron_match)
            yield luigi.LocalTarget(detected_connection)
            yield luigi.LocalTarget(gt_connection)
    
    def output(self):
        return luigi.LocalTarget(self.output_location)


class SynapseStatisticsRunMixin:
    
    def ariadne_run(self):
        inputs = self.input()
        neuron_map_tgt = inputs.next()
        gt_synapse_connection_tgt = inputs.next()
        
        #
        # Create all synapse-neuron-synapse triplets in the ground-truth
        #
        #
        # Get the detected synapses, mapping synapses to their gt
        # and neurons to their global labeling
        #
        s1_d = []
        s2_d = []
        n_d = []
        s_gt = []
        n_gt = []
        gt_labels = []
        tp_synapses = 0
        fp_synapses = 0
        fn_synapses = 0
        neuron_map_dict = json.load(neuron_map_tgt.open("r"),
                                    object_hook=to_hashable)
        neuron_map = dict(neuron_map_dict["volumes"])
        synapse_offset = 1
        synapse_map = []
        try:
            while True:
                synapse_matches = json.load(inputs.next().open("r"),
                                            object_hook=to_hashable)
                neuron_matches = json.load(inputs.next().open("r"),
                                           object_hook=to_hashable)
                d_synapse_connections = json.load(inputs.next().open("r"),
                                                  object_hook = to_hashable)
                gt_synapse_connections = json.load(inputs.next().open("r"),
                                                   object_hook = to_hashable)
                #
                # The ground-truth synapse/neuron connections
                #
                gt_neuron = np.hstack([
                    gt_synapse_connections["neuron_1"],
                    gt_synapse_connections["neuron_2"]])
                gt_synapse = np.hstack([gt_synapse_connections["synapse"]]*2)
                #
                # The local label #s of the synapses. Doubled because each
                # synapse connects two neurons
                #
                l_synapse = np.hstack([d_synapse_connections["synapse"]] * 2)
                if len(l_synapse) == 0:
                    continue
                l_neuron = np.hstack([
                    d_synapse_connections["neuron_1"],
                    d_synapse_connections["neuron_2"]])
                volume = synapse_matches["volume"]
                #
                # map of gt synapses to detected
                #
                gt_per_detected = np.array(synapse_matches["gt_per_detected"])
                detected_per_gt = np.array(synapse_matches["detected_per_gt"])
                gtl = np.array(synapse_matches["gt_labels"])
                max_synapse = max(np.max(gt_synapse), len(detected_per_gt))
                global_synapse_labels = np.zeros(max_synapse + 1, int)
                global_synapse_labels[gtl] = \
                    np.arange(len(gtl)) + synapse_offset
                synapse_offset += len(gtl)
                synapse_map.append((
                    dict(x=volume["x"], y=volume["y"], z=volume["z"],
                         width=volume["width"], height=volume["height"], 
                         depth=volume["depth"]), 
                    zip(global_synapse_labels, gtl)))
                #
                # # of false synapses are those without correspondences
                # in the ground truth labels
                #
                fn_synapses += np.sum(detected_per_gt[gtl] == 0)
                d_labels = np.array(synapse_matches["detected_labels"])
                if len(d_labels) == 0:
                    continue
                tp_synapses += np.sum(gt_per_detected[d_labels] != 0)
                fp_synapses += np.sum(gt_per_detected[d_labels] == 0)
                to_keep = np.zeros(
                    max(np.max(d_labels)+1, np.max(l_synapse))+1, bool)
                to_keep[d_labels] = True
                mask = to_keep[l_synapse]
                l_neuron = l_neuron[mask]
                l_synapse = l_synapse[mask]
                #
                # map of local neuron labels to global
                #
                tmp = np.array(neuron_map[volume])
                nm = np.zeros(np.max(tmp[:, 0])+1, int)
                nm[tmp[:, 0]] = tmp[:, 1]
                g_neuron = nm[l_neuron]
                #
                # Convert local synapse label # to the matching gt synapse
                # label #. Zero = no matching gt synapse.
                #
                # Note that there is no gt synapse with a label of zero,
                # so anything that pairs with a false positive synapse will
                # not be found in the true rendering of matching pairs.
                #
                g_synapse = global_synapse_labels[gt_per_detected[l_synapse]]
                #
                # Get the triples
                #
                s1_t, s2_t, n_t = self.get_triplets(g_synapse, g_neuron)
                #
                # Remove false positive detection <-> false positive detection
                #
                mask = (s1_t != 0) | (s2_t != 0)
                s1_t, s2_t, n_t = s1_t[mask], s2_t[mask], n_t[mask]
                s1_d.append(s1_t)
                s2_d.append(s2_t)
                n_d.append(n_t)
                #
                # Now do similar for gt_connections
                #
                gt_synapse = global_synapse_labels[gt_synapse]
                mask = gt_synapse != 0
                gt_synapse, gt_neuron = gt_synapse[mask], gt_neuron[mask]
                s_gt.append(gt_synapse)
                n_gt.append(gt_neuron)
                
        except StopIteration:
            pass
        s_gt, n_gt = np.hstack(s_gt), np.hstack(n_gt)
        s1_d = np.hstack(s1_d)
        s2_d = np.hstack(s2_d)
        #
        # Make the detected matrix
        #
        d_matrix = coo_matrix((np.ones(len(s1_d)*2),
                               (np.hstack((s1_d, s2_d)),
                                np.hstack((s2_d, s1_d)))),
                              shape=(synapse_offset, synapse_offset)).tocsr()
        #
        # Make the gt matrix
        #
        s1_gt, s2_gt, n_gt = self.get_triplets(s_gt, n_gt)
        gt_matrix = coo_matrix((np.ones(len(s1_gt)*2),
                                (np.hstack((s1_gt, s2_gt)),
                                 np.hstack((s2_gt, s1_gt)))),
                               shape=(synapse_offset, synapse_offset)).tocsr()
        #
        # Compile the results 
        #
        if len(s1_gt) == 0:
            true_positives = np.zeros(0, bool)
        else:
            true_positives = (d_matrix[s1_gt, s2_gt] != 0).A1
        tp_doublets = np.column_stack((s1_gt[true_positives],
                                       s2_gt[true_positives]))
        tp_doublets = tp_doublets[tp_doublets[:, 0] < tp_doublets[:, 1]]
        false_negatives = ~true_positives
        fn_doublets = np.column_stack((s1_gt[false_negatives],
                                       s2_gt[false_negatives]))
        fn_doublets = fn_doublets[fn_doublets[:, 0] < fn_doublets[:, 1]]
        if len(s1_d) == 0:
            false_positives = np.zeros(0, bool)
        else:
            false_positives = gt_matrix[s1_d, s2_d].A1 == 0
        fp_doublets = np.column_stack((s1_d[false_positives],
                                       s2_d[false_positives]))
        fp_doublets = fp_doublets[(fp_doublets[:, 0] < fp_doublets[:, 1]) &
                                  (fp_doublets[:, 0] != 0)]
        n_true_positives = np.sum(true_positives)
        n_false_negatives = len(true_positives) - n_true_positives
        n_false_positives = np.sum(false_positives)
        precision = \
            float(n_true_positives) / \
            (n_true_positives + n_false_positives + np.finfo(float).eps)
        recall = \
            float(n_true_positives) / \
            (n_true_positives + n_false_negatives + np.finfo(float).eps)
        #
        # False negatives to add to pairs
        #
        s1_gt_fn = s1_gt[false_negatives]
        s2_gt_fn = s2_gt[false_negatives]
        #
        # Synapse stats
        #
        synapse_precision =\
            float(tp_synapses) / \
            (tp_synapses + fp_synapses + np.finfo(float).eps)
        synapse_recall = float(tp_synapses) /\
            (tp_synapses + fn_synapses + np.finfo(float).eps)
        result = dict(
            precision=precision,
            recall=recall,
            n_true_positives=n_true_positives,
            n_false_positives=n_false_positives,
            n_false_negatives=n_false_negatives,
            n_true_positive_synapses=tp_synapses,
            n_false_positive_synapses=fp_synapses,
            n_false_negative_synapses=fn_synapses,
            synapse_precision=synapse_precision,
            synapse_recall=synapse_recall,
            true_positive_labels=[_.tolist() for _ in tp_doublets],
            false_positive_labels=[_.tolist() for _ in fp_doublets],
            false_negative_labels=[_.tolist() for _ in fn_doublets],
            synapse_map=synapse_map)
        with self.output().open("w") as fd:
            json.dump(result, fd)
        
    def get_triplets(self, synapses, neurons):
        '''Get synapse, neuron, synapse triplets'''
        #
        # Sort the list by neuron-major, synapse-minor
        #
        order = np.lexsort((synapses, neurons))
        synapses, neurons = synapses[order], neurons[order]
        #
        # Get counts of # synapses per neuron
        #
        first = np.where(np.hstack([
            [True], neurons[1:] != neurons[:-1], [True]]))[0]
        counts=first[1:] - first[:-1]
        idx = first[:-1]
        #
        # Now there are (N * (N-1)) / 2 combinations for N synapses
        # attached to each neuron. We make tables for each N that do
        # the enumeration.
        #
        max_count = np.max(counts)
        tbl = np.zeros((max_count+1,
                        ((max_count * (max_count-1))/2),
                        2), int)
        for count in np.unique(counts):
            n = count * (count - 1) / 2
            if n == 0:
                continue
            a, b = np.mgrid[0:count, 0:count]
            a, b = a[a < b], b[a < b]
            tbl[count, :n, 0] = a
            tbl[count, :n, 1] = b
        #
        # The unique neurons:
        #
        first = first[:-1][counts > 1]
        uneurons = neurons[idx[counts > 1]]
        counts = counts[counts > 1]
        #
        # Create indices into above table
        #
        # neuron at index
        n_first = np.cumsum(np.hstack([[0], counts * (counts-1) / 2]))
        n_idx = np.zeros(n_first[-1], int)
        n_idx[n_first[1:-1]] = 1
        n_idx = np.cumsum(n_idx)
        #
        # synapse index
        #
        s_idx = np.arange(len(n_idx)) - n_first[n_idx]
        #
        # Use the combination table to read the offsets of the 
        # first and second synapses relative to the first occurence of
        # the neuron.
        #
        s1 = synapses[first[n_idx] + tbl[counts[n_idx], s_idx, 0]]
        s2 = synapses[first[n_idx] + tbl[counts[n_idx], s_idx, 1]]
        neuron = uneurons[n_idx]
        return s1, s2, neuron

class SynapseStatisticsTask(SynapseStatisticsTaskMixin,
                            SynapseStatisticsRunMixin,
                            RequiresMixin,
                            RunMixin,
                            SingleThreadedMixin,
                            luigi.Task):
    '''Compute precision and recall on synapse-neuron-synapse connections
    
    Create a .json file with the following key/values:
    
    precision: true positives / true_positives + false_positives
    recall: true positives / true positives + false negatives
    n_true_positives: total # of true positives
    n_false_positives: total # of detected synapse pairs not in gt set
    n_false_negatives: total # of gt synapse pairs not detected
    '''

    task_namespace = "ariadne_microns_pipeline"
