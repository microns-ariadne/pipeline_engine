'''Calculate the confusion matrix of pairs of synapses'''

import json
import luigi
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
    gt_synapse_connections = luigi.Parameter(
        description="The .json file that records synapse-neuron connections "
                    "in the ground-truth.")
    output_location = luigi.Parameter(
        description="The location for the confusion map")

    def input(self):
        yield luigi.LocalTarget(self.neuron_map)
        yield luigi.LocalTarget(self.gt_synapse_connections)
        for synapse_match, neuron_match, detected_connection in zip(
            self.synapse_matches, self.gt_neuron_maps,
            self.detected_synapse_connections):
            yield luigi.LocalTarget(synapse_match)
            yield luigi.LocalTarget(neuron_match)
            yield luigi.LocalTarget(detected_connection)
    
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
        gt_synapse_connections = json.load(gt_synapse_connection_tgt.open("r"))
        synapse_map = dict([
            (k, np.array(value))
            for k, v in gt_synapse_connections["synapse_map"].items()])
        synapses = np.array(gt_synapse_connections["synapse"])
        neurons = np.array(gt_synapse_connections["neuron"])
        s1_gt, s2_gt, n_gt = get_triplets(synapses, neurons)
        #
        # Make a table of all synapse pairs
        #
        pair_tbl = coo_matrix((np.ones(len(s1_gt)*2, np.uint8),
                               (np.hstack(s1_gt, s2_gt),
                                np.hstack(s2_gt, s1_gt)))).tocsr()
        #
        # Get the detected synapses, mapping synapses to their gt
        # and neurons to their global labeling
        #
        s1_d = []
        s2_d = []
        n_d = []
        bad_synapses = 0
        neuron_map = json.load(neuron_map_tgt.open("r"),
                               object_hook=to_hashable)
        try:
            while True:
                synapse_matches = json.load(inputs.next().open("r"),
                                            object_hook=to_hashable)
                neuron_matches = json.load(inputs.next().open("r"),
                                           object_hook=to_hashable)
                d_synapse_connections = json.load(inputs.next().open("r"),
                                                  object_hook = to_hashable)
                #
                # The local label #s of the synapses. Doubled because each
                # synapse connects two neurons
                #
                l_synapse = np.hstack([d_synapse_connections["synapse"]] * 2)
                l_neuron = np.hstack([
                    d_synapse_connections["neuron_1"],
                    d_synapse_connections["neuron_2"]])
                volume = synapse_matches["volume"]
                #
                # map of local neuron labels to global
                #
                nm = neuron_map[volume]
                g_neuron = nm[l_neuron]
                #
                # map of gt synapses to detected
                #
                gt_per_detected = np.array(synapse_matches["gt_per_detected"])
                sm = np.array(synapse_map[volume])
                #
                # Convert local synapse label # to the matching gt synapse
                # label #. Zero = no matching gt synapse.
                #
                # Note that there is no gt synapse with a label of zero,
                # so anything that pairs with a false positive synapse will
                # not be found in the true rendering of matching pairs.
                #
                g_synapse = sm[gt_per_detected[l_synapse]]
                #
                # Get the triples
                #
                s1_t, s2_t, n_t = get_triplets(g_synapse, g_neuron)
                s1_d.append(s1_t)
                s2_d.append(s2_t)
                n_d.append(n_t)
        except StopIteration:
            pass
        #
        # Make the detected matrix
        #
        s1_d = np.hstack(s1_d)
        s2_d = np.hstack(s2_d)
        max_label = np.max([np.max(_) for _ in s1_d, s2_d, s1_gt, s2_gt])
        d_matrix = coo_matrix((np.ones(len(s1_d)*2),
                               (np.hstack((s1_d, s2_d)),
                                np.hstack((s2_d, s1_d)))),
                              shape=(max_label+1, max_label+1)).tocsr()
        gt_matrix = coo_matrix((np.ones(len(s1_gt)*2),
                                (np.hstack((s1_gt, s2_gt),
                                           (s2_gt, s1_gt)))),
                               shape=(max_label+1, max_label+1)).tocsr()
        #
        # Compile the results 
        #
        true_positives = d_matrix[s1_gt, s2_gt].A1 != 0
        false_negatives = ~true_positives
        false_positives = gt_matrix[s1_d, s2_d].A1 == 0
        n_true_positives = np.sum(true_positives)
        n_false_negatives = len(true_positives) - n_true_positives
        n_false_positives = np.sum(false_positives)
        precision = \
            float(n_true_positives) / (n_true_positives + n_false_positives)
        recall = \
            float(n_true_positives) / (n_true_positives + n_false_negatives)
        #
        # False negatives to add to pairs
        #
        s1_gt_fn = s1_gt[false_negatives]
        s2_gt_fn = s2_gt[false_negatives]
        result = dict(
            precision=precision,
            recall=recall,
            n_true_positives=n_true_positives,
            n_false_positives=n_false_positives,
            n_false_negatives=n_false_negatives)
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
            [True], neurons[1:] != neurons[:-1], [True]]))
        counts=first[1:] - first[:-1]
        idx = first[:-1]
        #
        # Now there are (N * (N-1)) / 2 combinations for N synapses
        # attached to each neuron. We make tables for each N that do
        # the enumeration.
        #
        max_count = np.max_counts
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
        uneurons = neurons[first[counts > 1]]
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
        return neuron, s1, s2

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