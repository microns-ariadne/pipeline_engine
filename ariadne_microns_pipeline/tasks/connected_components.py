'''Tasks for finding connected components across blocks'''

import copy
import bisect
import enum
import fast64counter
import json
import logging
import luigi
import numpy as np
import os
import rh_logger
import time
import tifffile
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .utilities import RequiresMixin, RunMixin, SingleThreadedMixin, \
     DatasetMixin
from ..targets import DestVolumeReader, SrcVolumeTarget
from ..parameters import Volume
from ..parameters import VolumeParameter
from .utilities import to_hashable

class LogicalOperation(enum.Enum):
    OR = 1
    AND = 2

class Direction(enum.Enum):
    X = 0
    Y = 1
    Z = 2

class JoiningMethod(enum.Enum):
    
    '''Join blocks using a simple minimum overlap criterion'''
    SIMPLE_OVERLAP = 1
    '''Join blocks using the pairwise multimatch marriage algorithm'''
    PAIRWISE_MULTIMATCH = 2

class ConnectedComponentsTaskMixin:

    volume1 = VolumeParameter(
        description="The voxel volume for the whole of segmentation #1")
    cutout_loading_plan1_path = luigi.Parameter(
        description="The file path to the loading plan for cutout #1")
    segmentation_loading_plan1_path = luigi.Parameter(
        description="The file path for the entire segmentation #1")
    volume2 = VolumeParameter(
        description="The voxel volume for the whole of segmentation #2")
    cutout_loading_plan2_path = luigi.Parameter(
        description="The file path to the loading plan for cutout #2")
    segmentation_loading_plan2_path = luigi.Parameter(
        description="The file path for the entire segmentation #2")
    output_location = luigi.Parameter(
        description=
        "The location for the JSON file containing the concordances")

    def input(self):
        for path in self.cutout_loading_plan1_path, \
            self.cutout_loading_plan2_path:
            reader = DestVolumeReader(path)
            for tgt in reader.get_source_targets():
                yield(tgt)
    
    def output(self):
        return luigi.LocalTarget(self.output_location)
    
    def estimate_memory_usage(self):
        '''Return an estimate of the number of bytes required to run'''
        v1 = np.prod([1888, 1888, 70])
        m1 = 3459457 * 1000
        v2 = np.prod([1888, 1416, 42])
        m2 = 1868989 * 1000
        #
        # Model is Ax + B where x is volume in voxels. We assume the major
        # memory cost is loading the big volumes and that they are the same size
        #
        B = (v1 * m2 - v2 * m1) / (v1 - v2)
        A = (float(m1) - B) / v1
        v = np.prod([self.volume1.width, self.volume1.height, self.volume1.depth])
        return int(A * v + B)
        
    
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
    #
    # Parameters for the pairwise multimatch
    #
    joining_method = luigi.EnumParameter(
        enum=JoiningMethod,
        default=JoiningMethod.SIMPLE_OVERLAP,
        description="Algorithm to use to join segmentations across blocks")
    min_overlap_volume = luigi.IntParameter(
        default=1000,
        description="The minimum acceptable volume in voxels of overlap "
        "between segments needed to join them.")
    max_poly_matches = luigi.IntParameter(
        default=1)
    dont_join_orphans = luigi.BoolParameter()
    orphan_min_overlap_ratio = luigi.FloatParameter(
        default=0.9)
    orphan_min_overlap_volume = luigi.IntParameter(
        default=1000,
        description="The minimum acceptable volume in voxels of overlap "
                    "needed to join an orphan segment.")
    
    def ariadne_run(self):
        '''Look within the overlap volume to find the concordances
        
        We load the two volumes and note the unique pairs of labels
        that appear together at the same voxel for the voxels in the
        overlap volume.
        '''
        cutout1_tgt = DestVolumeReader(self.cutout_loading_plan1_path)
        cutout2_tgt = DestVolumeReader(self.cutout_loading_plan2_path)
        cutout1, cutout2 = [_.imread() for _ in cutout1_tgt, cutout2_tgt]
        
        if self.joining_method == JoiningMethod.PAIRWISE_MULTIMATCH:
            connections, counts = self.pairwise_multimatch(cutout1, cutout2)
        else:
            connections, counts = self.overlap_match(cutout1, cutout2)
        d = dict(connections=connections, counts=counts)
        for volume, name in ((self.volume1, "1"),
                             (self.volume2, "2"),
                             (cutout1_tgt.volume, "overlap")):
            d[name] = dict(x=volume.x,
                           y=volume.y,
                           z=volume.z,
                           width=volume.width,
                           height=volume.height,
                           depth=volume.depth)
        #
        # Get the statistics from the source target.
        #
        for i, src_tgt, location in zip(
            ["1", "2"], 
            self.input(),
            [self.segmentation_loading_plan1_path,
             self.segmentation_loading_plan2_path]):
            src_json = json.load(open(src_tgt.path))
            d[i]["labels"] = src_json["labels"]
            d[i]["areas"] = src_json["areas"]
            d[i]["location"] = location
        with self.output().open("w") as fd:
            json.dump(d, fd)

    def overlap_match(self, cutout1, cutout2):
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
        return as_list, counts.tolist()
    
    def pairwise_multimatch(self, cutout1, cutout2):
        '''Match the segments in two cutouts using pairwise marriage
        
        :param cutout1: The area to examine for overlapping segmentation from
                        the first volume.
        :param cutout2: The area to examine for overlapping segmentation from
                        the second volume.
        
        The code in this routine is adapted from Seymour Knowles-Barley's
        pairwise_multimatch: https://github.com/Rhoana/rhoana/blob/29526687202921e7173b33ec909fcd6e5b9e18bf/PairwiseMatching/pairwise_multimatch.py
        '''
        counter = fast64counter.ValueCountInt64()
        counter.add_values_pair32(cutout1.astype(np.int32).ravel(), 
                                  cutout2.astype(np.int32).ravel())
        overlap_labels1, overlap_labels2, overlap_areas = \
            counter.get_counts_pair32()
        
        areacounter = fast64counter.ValueCountInt64()
        areacounter.add_values(np.int64(cutout1.ravel()))
        areacounter.add_values(np.int64(cutout2.ravel()))
        areas = dict(zip(*areacounter.get_counts()))
        # Merge with stable marrige matches best match = greatest overlap
        to_merge = []
        to_merge_overlap_areas = []
    
        m_preference = {}
        w_preference = {}
    
        # Generate preference lists
        for l1, l2, overlap_area in zip(
            overlap_labels1, overlap_labels2, overlap_areas):
    
            if l1 != 0 and l2 != 0 and\
               overlap_area >= self.min_overlap_volume:
                if l1 not in m_preference:
                    m_preference[l1] = [(l2, overlap_area)]
                else:
                    m_preference[l1].append((l2, overlap_area))
                if l2 not in w_preference:
                    w_preference[l2] = [(l1, overlap_area)]
                else:
                    w_preference[l2].append((l1, overlap_area))
    
        def get_area(l1, l2):
            return [_ for _ in m_preference[l1] if _[0] == l2][0][1]
                    
        # Sort preference lists
        for mk in m_preference.keys():
            m_preference[mk] = sorted(m_preference[mk], 
                                      key=lambda x:x[1], reverse=True)
    
        for wk in w_preference.keys():
            w_preference[wk] = sorted(w_preference[wk], 
                                      key=lambda x:x[1], reverse=True)
    
        # Prep for proposals
        mlist = sorted(m_preference.keys())
        wlist = sorted(w_preference.keys())
    
        mfree = mlist[:] * self.max_poly_matches
        engaged  = {}
        mprefers2 = copy.deepcopy(m_preference)
        wprefers2 = copy.deepcopy(w_preference)
    
        # Stable marriage loop
        rh_logger.logger.report_event("Entering stable marriage loop")
        t0 = time.time()
        while mfree:
            m = mfree.pop(0)
            mlist = mprefers2[m]
            if mlist:
                w = mlist.pop(0)[0]
                fiance = engaged.get(w)
                if not fiance:
                    # She's free
                    engaged[w] = [m]
                    rh_logger.logger.report_event(
                        "  {0} and {1} engaged".format(w, m),
                    log_level=logging.DEBUG)
                elif len(fiance) < self.max_poly_matches and m not in fiance:
                    # Allow polygamy
                    engaged[w].append(m)
                    rh_logger.logger.report_event(
                        "  {0} and {1} engaged".format(w, m),
                    log_level=logging.DEBUG)
                else:
                    # m proposes w
                    wlist = list(x[0] for x in wprefers2[w])
                    dumped = False
                    for current_match in fiance:
                        if wlist.index(current_match) > wlist.index(m):
                            # w prefers new m
                            engaged[w].remove(current_match)
                            engaged[w].append(m)
                            dumped = True
                            rh_logger.logger.report_event(
                                "  {0} dumped {1} for {2}"
                                .format(w, current_match, m),
                                log_level=logging.DEBUG)
                            if mprefers2[current_match]:
                                # current_match has more w to try
                                mfree.append(current_match)
                            break
                    if not dumped and mlist:
                        # She is faithful to old fiance - look again
                        mfree.append(m)
        rh_logger.logger.report_metric("Stable marriage loop time (sec)",
                                       time.time() - t0)
    
        # m_can_adopt = copy.deepcopy(overlap_labels1)
        # w_can_adopt = copy.deepcopy(overlap_labels1)
        m_partner = {}
        w_partner = {}
        t0 = time.time()
        for l2 in engaged.keys():
            for l1 in engaged[l2]:
    
                rh_logger.logger.report_event(
                    "Merging segments {1} and {0}.".format(l1, l2),
                    log_level=logging.DEBUG)
                to_merge.append((l1, l2))
                to_merge_overlap_areas.append(get_area(l1, l2))
    
                # Track partners
                if l1 in m_partner:
                    m_partner[l1].append(l2)
                else:
                    m_partner[l1] = [l2]
                if l2 in w_partner:
                    w_partner[l2].append(l1)
                else:
                    w_partner[l2] = [l1]
        rh_logger.logger.report_metric("Pairwise multimatch merge time (sec)",
                                       time.time() - t0)
        # Join all orphans that fit overlap proportion critera (no limit)
        if not self.dont_join_orphans:
            t0 = time.time()
            for l1 in m_preference.keys():
    
                # ignore any labels with a match
                # if l1 in m_partner.keys():
                #     continue
    
                l2, overlap_area = m_preference[l1][0]
    
                # ignore if this pair is already matched
                if l1 in m_partner.keys() and l2 in m_partner[l1]:
                    continue
    
                overlap_ratio = overlap_area / np.float32(areas[l1])
    
                if overlap_ratio >= self.orphan_min_overlap_ratio and \
                   overlap_area >= self.orphan_min_overlap_volume:
                    rh_logger.logger.report_event(
                        "Merging orphan segment {0} to {1} ({2} voxel overlap = {3:0.2f}%)."
                        .format(l1, l2, overlap_area, overlap_ratio * 100),
                        log_level=logging.DEBUG)
                    to_merge.append((l1, l2))
                    to_merge_overlap_areas.append(get_area(l1, l2))
    
            for l2 in w_preference.keys():
    
                # ignore any labels with a match
                # if l2 in w_partner.keys():
                #     continue
    
                l1, overlap_area = w_preference[l2][0]
    
                # ignore if this pair is already matched
                if l2 in w_partner.keys() and l1 in w_partner[l2]:
                    continue
    
                overlap_ratio = overlap_area / np.float32(areas[l2])
    
                if overlap_ratio >= self.orphan_min_overlap_ratio and \
                   overlap_area >= self.orphan_min_overlap_volume:
                    rh_logger.logger.report_event(
                        "Merging orphan segment {0} to {1} ({2} voxel overlap = {3:0.2f}%)."
                        .format(l2, l1, overlap_area, overlap_ratio * 100),
                        log_level=logging.DEBUG)
                    to_merge.append((l1, l2))
                    to_merge_overlap_areas.append(get_area(l1,l2))
            rh_logger.logger.report_metric(
                "Pairwise multimatch orphan joining time", 
                time.time() - t0)            
        #
        # Convert from np.uint32 or whatever to int to make JSON serializable
        #
        to_merge = [(int(a), int(b)) for a, b in to_merge]
        to_merge_overlap_areas = map(int, to_merge_overlap_areas)
        return to_merge, to_merge_overlap_areas
        
class ConnectedComponentsTask(ConnectedComponentsTaskMixin,
                              ConnectedComponentsRunMixin,
                              RequiresMixin, RunMixin,
                              SingleThreadedMixin,
                              luigi.Task):
    '''This task finds the connections between the segmentations of two volumes
    
    Given segmentation #1 and segmentation #2 and an overlapping volume,
    look at the labels in segmentation #1 and #2 at each pixel. These are
    the labels that are connected between the volumes. This task finds the
    unique labels between the segmentations and stores them in a JSON file.
    
    The connected components of the whole volume can be found by loading
    all of the JSON files and assigning each block's labels to a global
    label that may be shared between segments.
    
    To use the ConnectedComponentsTask, create loading plans for an overlapping
    region in each of the two segmentations.
    '''
    
    task_namespace = 'ariadne_microns_pipeline'


class AllConnectedComponentsTaskMixin:
    
    input_locations = luigi.ListParameter(
        description="The filenames of the output files from the "
        "ConnectedComponentsTask")
    output_location = luigi.Parameter(
        description="The filename of the global assignment")
    max_connections = luigi.Parameter(
        default=0,
        description="Reject a component if it makes more than this many "
                    "connections outside of its volume.")
    additional_loading_plans=luigi.ListParameter(
        default=[],
        description="Additional loading-plan files to be written into the "
        "locations dictionary, e.g. for use by a stitching pipeline.")
    
    def input(self):
        for input_location in self.input_locations:
            yield luigi.LocalTarget(input_location)
    
    def output(self):
        return luigi.LocalTarget(self.output_location)

class AllConnectedComponentsRunMixin:
    
    
    def ariadne_run(self):
        connections = []
        mappings = {}
        locations = {}
        joins = {}
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
            loc1 = d["1"]["location"]
            loc2 = d["2"]["location"]
            for k1 in "1", "2":
                for k2 in "labels", "areas", "location":
                    if k2 in d[k1]:
                        del d[k1][k2]
            d["pathname"] = input_target.path
            k1 = to_hashable(d["1"])
            k2 = to_hashable(d["2"])
            joins[k1, k2] = input_target.path
            for k, l, loc in (k1, l1, loc1), (k2, l2, loc2):
                if k not in locations:
                    locations[k] = loc
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
        connections = np.vstack(connections)
        #
        # Filter for too many connections
        if self.max_connections != 0:
            n_connections = np.bincount(connections[:, 0])
            too_many = np.where(n_connections > self.max_connections)[0]
            cmap = np.ones(len(n_connections), bool)
            cmap[too_many] = False
            mask = cmap[connections[:, 0]] & cmap[connections[:, 1]]
            connections = np.vstack(
                (connections[mask], np.column_stack((too_many, too_many))))
        #
        # Now run connected components on the adjacency graph
        #
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
        d["additional_locations"] = []
        for volume, m in mappings.items():
            gm = [ (int(a), int(b)) for a, b in zip(m[:, 0], labels[m[:, 1]])]
            d["volumes"].append((dict(volume), gm))
        d["locations"] = []
        for volume, loc in locations.items():
            d["locations"].append((dict(volume), loc))
        for volume, loading_plan in self.additional_loading_plans:
            d["additional_locations"].append((volume, loading_plan))
        d["joins"] = []
        for (k1, k2), path in joins.items():
            d["joins"].append((dict(k1), dict(k2), path))
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
    
class FakeAllConnectedComponentsTaskMixin:
    
    volume = VolumeParameter(
        description="The volume of the segmentation")
    loading_plan = luigi.Parameter(
        description="The location of the segmentation")
    output_location = luigi.Parameter(
        description="The location of the connectivity graph")
    
    def input(self):
        for tgt in DestVolumeReader(self.loading_plan).get_source_targets():
            yield tgt
    
    def output(self):
        return luigi.LocalTarget(self.output_location)

class FakeAllConnectedComponentsRunMixin:
    
    def ariadne_run(self):
        '''Create a connection graph for a single volume'''
        
        seg = DestVolumeReader(self.loading_plan).imread()
        components = np.where(np.bincount(seg.flatten()) != 0)[0]
        if components[0] == 0:
            components = components[1:]
        volumes = [[ self.volume.to_dictionary(), 
                     [[_, _] for _ in components]]]
        locations = [[ self.volume.to_dictionary(), self.loading_plan]]
        d = dict(count=len(components),
                 volumes=volumes,
                 locations=locations,
                 joins=[])
        with self.output().open("w") as fd:
            json.dump(d, fd)

class FakeAllConnectedComponentsTask(FakeAllConnectedComponentsTaskMixin,
                                     FakeAllConnectedComponentsRunMixin,
                                     RequiresMixin,
                                     RunMixin,
                                     luigi.Task):
    '''A task to write out the connectivity graph if we have a single volume
    
    The FakeAllConnectedComponentsTask creates a fake connectivity graph
    JSON file (see AllConnectedComponentsTask for details on this file) for
    the case where the pipeline consists of only one volume.
    '''
    
    task_namespace = "ariadne_microns_pipeline"

class VolumeRelabelingTaskMixin(DatasetMixin):
    
    input_volumes = luigi.ListParameter(
        description="Input volumes to be composited together")
    relabeling_location = luigi.Parameter(
        description=
        "The location of the output file from AllConnectedComponentsTask "
        "that gives the local/global relabeling of the segmentation")
    
    def input(self):
        '''Return the volumes to be assembled'''
        yield luigi.LocalTarget(self.relabeling_location)
        for loading_plan in self.input_volumes:
            for tgt in DestVolumeReader(loading_plan).get_source_targets():
                yield tgt
    
class VolumeRelabelingRunMixin:
    
    def ariadne_run(self):
        '''Composite and relabel each of the volumes'''
        output_volume_target = self.output()
        output_result = np.zeros((output_volume_target.volume.depth,
                                  output_volume_target.volume.height,
                                  output_volume_target.volume.width),
                                 np.uint32)
        x0 = output_volume_target.volume.x
        x1 = output_volume_target.volume.x1
        y0 = output_volume_target.volume.y
        y1 = output_volume_target.volume.y1
        z0 = output_volume_target.volume.z
        z1 = output_volume_target.volume.z1
        #
        # Read the local->global mappings
        #
        with open(self.relabeling_location, "r") as fd:
            mappings = json.load(fd)
        #
        # Make a dictionary of all the candidate volumes
        #
        volumes = {}
        for loading_plan in self.input_volumes:
            tgt = DestVolumeReader(loading_plan)
            key = to_hashable(dict(x=tgt.volume.x,
                                   y=tgt.volume.y,
                                   z=tgt.volume.z,
                                   width=tgt.volume.width,
                                   height=tgt.volume.height,
                                   depth=tgt.volume.depth))
            volumes[key] = tgt
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
            volume = Volume(**volume)
            mapping_idxs = np.array(mapping, np.uint32)
            mapping_xform = np.zeros(mapping_idxs[:, 0].max()+1, np.uint32)
            mapping_xform[mapping_idxs[:, 0]] = mapping_idxs[:, 1]
            labels = input_volume_target.imread()[vz0-volume.z:vz1-volume.z,
                                                  vy0-volume.y:vy1-volume.y,
                                                  vx0-volume.x:vx1-volume.x]
            labels = mapping_xform[labels]
            output_result[vz0-z0:vz1-z0,
                          vy0-y0:vy1-y0,
                          vx0-x0:vx1-x0] = labels
        output_volume_target.imwrite(output_result)

class VolumeRelabelingTask(VolumeRelabelingTaskMixin,
                           VolumeRelabelingRunMixin,
                           RequiresMixin, RunMixin,
                           luigi.Task):
    '''Relabel a segmentation volume
    
    The VolumeRelabilingTask uses the output of the AllConnectedComponents
    to map the local segmentations of several input volumes into a single
    unitary output volume.
    '''
    task_namespace = "ariadne_microns_pipeline"

class StoragePlanRelabelingTask(
    DatasetMixin,
    RequiresMixin,
    RunMixin,
    luigi.Task):
    '''Relabel and rewrite a segmentation storage plan'''
    connectivity_graph_path = luigi.Parameter(
        description="Path to the connectivity graph describing the volume "
        "to be uploaded to the Boss")
    src_loading_plan_path = luigi.Parameter(
        description="A loading plan that's named in the connectivity graph")
    
    def input(self):
        loading_plan = DestVolumeReader(self.src_loading_plan_path)
        for tgt in loading_plan.get_source_targets():
            yield tgt
    
    def ariadne_run(self):
        src_target = DestVolumeReader(self.src_loading_plan_path)
        cg = ConnectivityGraph.load(open(self.connectivity_graph_path))
        src_volume = src_target.volume
        data = cg.convert(src_target.imread(), src_volume)
        dest_tgt = self.output()
    
        dest_volume = dest_tgt.volume
        assert dest_volume.x >= src_volume.x
        assert dest_volume.x1 <= src_volume.x1
        assert dest_volume.y >= src_volume.y
        assert dest_volume.y1 <= src_volume.y1
        assert dest_volume.z >= src_volume.z
        assert dest_volume.z1 <= src_volume.z1
        
        self.output().imwrite(
            data[dest_volume.z - src_volume.z:dest_volume.z1 - src_volume.z,
                 dest_volume.y - src_volume.y:dest_volume.y1 - src_volume.y,
                 dest_volume.x - src_volume.x:dest_volume.x1 - src_volume.x])
        

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
        self.locations = {}
        if "locations" in mappings:
            for volume, location in mappings["locations"]:
                self.locations[to_hashable(volume)] = location
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
    
    def get_tgt(self, volume):
        '''Return a DestVolumeReader covering the volume passed
        
        '''
        location = DestVolumeReader(
            **self.locations[to_hashable(volume.to_dictionary())])
        return location
        