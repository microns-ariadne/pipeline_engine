import json
import luigi
import os

from ..parameters import Volume
from ..parameters import DatasetLocation
from ..targets.factory import TargetFactory
from ..tasks.factory import AMTaskFactory
from ..tasks.connected_components import LogicalOperation, JoiningMethod
from ..tasks.utilities import to_hashable

class StitchPipelineTask(luigi.Task):
    task_namespace = "ariadne_microns_pipeline"
    
    component_graph_1 = luigi.Parameter(
        description="The location of the component graph file written by "
                    "AllConnectedComponentsTask for volume #1")
    component_graph_2 = luigi.Parameter(
        description="The location of the component graph file written by "
                    "AllConnectedComponentsTask for volume #2")
    output_location = luigi.Parameter(
        description="The location of the component graph file written by "
                    "AllConnectedComponentsTask encompassing both volumes")
    #
    # Parameters for the connected components task
    #
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
    partner_min_total_area_ratio = luigi.FloatParameter(
        default=0.001)
    max_poly_matches = luigi.IntParameter(
        default=1)
    dont_join_orphans = luigi.BoolParameter()
    orphan_min_overlap_ratio = luigi.FloatParameter(
        default=0.9)
    orphan_min_total_area_ratio = luigi.FloatParameter(
        default=0.001)
    halo_size_xy = luigi.IntParameter(
        default=5,
        description="The number of pixels on either side of the origin to "
                    "use as context when extracting the slice to be joined, "
                    "joining slices in the x and y directions")
    halo_size_z = luigi.IntParameter(
        default=1,
        description="The number of pixels on either side of the origin to "
                    "use as context when extracting the slice to be joined, "
                    "joining slices in the z direction")

    
    def inputs(self):
        yield luigi.LocalTarget(self.component_graph_1)
        yield luigi.LocalTarget(self.component_graph_2)
        
    def output(self):
        return luigi.LocalTarget(self.output_location+".done")

    @staticmethod
    def _overlaps(v1, v2):
        '''Return true if volume 1 overlaps volume 2'''
        for origin_key, size_key in (("x", "width"),
                                     ("y", "height"),
                                     ("z", "depth")):
            v1_0 = v1[origin_key]
            v1_1 = v1_0 + v1[size_key]
            v2_0 = v2[origin_key]
            v2_1 = v2_0 + v2[size_key]
            if v1_0 >= v2_1 or v2_0 >= v1_1:
                return False
        return True
    
    def _find_overlapping_volume(self, v1, v2):
        '''Figure out how to configure our overlap plane
        
        The assumption here is that the dimension with the lowest percent
        overlap defines the overlap plane.
        '''
        min_overlap_score = 2
        for origin_key, size_key in (("x", "width"),
                                     ("y", "height"),
                                     ("z", "depth")):
            v1_0 = v1[origin_key]
            v1_1 = v1_0 + v1[size_key]
            v2_0 = v2[origin_key]
            v2_1 = v2_0 + v2[size_key]
            #
            # compute the overlap / maximum extent
            overlap_score = \
                float(min(v1_1, v2_1) - max(v1_0, v2_0)) / \
                float(max(v1_1, v2_1) - min(v1_0, v2_0))
            if overlap_score < min_overlap_score:
                overlap_identity = origin_key
                min_overlap_score = overlap_score
        #
        # For each dimension other than the overlap_identity, take the
        # total overlapping extent. For the overlapping identity dimension
        # take a single plane plus the halo size.
        #
        overlap_volume = {}
        for origin_key, size_key, halo_size in (
            ("x", "width", self.halo_size_xy),
            ("y", "height", self.halo_size_xy),
            ("z", "depth", self.halo_size_z)):
            v1_0 = v1[origin_key]
            v1_1 = v1_0 + v1[size_key]
            v2_0 = v2[origin_key]
            v2_1 = v2_0 + v2[size_key]
            if origin_key != overlap_identity:
                overlap_volume[origin_key] = max(v1_0, v2_0)
                overlap_volume[size_key] = min(v1_1, v2_1) - max(v1_0, v2_0)
            else:
                midpoint = int((max(v1_0, v2_0) + min(v1_1, v2_1))/2)
                overlap_volume[origin_key] = midpoint - halo_size
                overlap_volume[size_key] = halo_size * 2 + 1
        return Volume(
            overlap_volume["x"], overlap_volume["y"], overlap_volume["z"],
            overlap_volume["width"], overlap_volume["height"], 
            overlap_volume["depth"])
        
    def compute_requirements(self):
        '''Compute the requirements and dependencies for the pipeline
        
        The AllConnectedComponentsTasks must have been run at this point.
        '''
        self.factory = AMTaskFactory()
        cg1 = json.load(open(self.component_graph_1, "r"))
        cg2 = json.load(open(self.component_graph_2, "r"))
        #
        # These are the block joins done by the individual pipelines. They
        # are re-fed into AllConnectedComponents for the next round.
        #
        joins_done = cg1["joins"] + cg2["joins"]
        joins_to_do = []
        #
        # Build a database of locations in volume # 1
        #
        d_locs = {}
        for volume, location in cg1["locations"]:
            d_locs[to_hashable(volume)] = location
        #
        # Find overlapping volumes by comparing volume #2 against volume # 1
        #
        for volume1, location1 in cg2["locations"]:
            for volume2, location2 in d_locs.items():
                if StitchPipelineTask._overlaps(volume1, volume2):
                    v1 = Volume(**volume1)
                    v2 = Volume(**volume2)
                    l1 = DatasetLocation(**location1)
                    l2 = DatasetLocation(**location2)
                    filename = "connected-components-%d-%d-%d_%d-%d-%d.json" % (
                        v1.x, v1.y, v1.z, v2.x, v2.y, v2.z)
                    output_location = os.path.join(l1.roots[0], filename)
                    overlap_volume = self._find_overlapping_volume(
                        volume1, volume2)
                    task = self.factory.gen_connected_components_task(
                        v1, l1, v2, l2, overlap_volume, output_location)
                    joins_to_do.append(task)
        #
        # Make the ultra-stupendous AllConnectedComponentsTask
        #
        all_join_files = [_[2] for _ in joins_done]
        all_join_files += [task.output().path for task in joins_to_do]
        self.all_connected_components_task = \
            self.factory.gen_all_connected_components_task(
                all_join_files, self.output_location)
        for task in joins_to_do:
            self.all_connected_components_task.set_requirement(task)

    def requires(self):
        if not hasattr(self, "all_connected_components_task"):
            self.compute_requirements()
        yield self.all_connected_components_task

    def run(self):
        with self.output().open("w") as fd:
            fd.write("Done")
