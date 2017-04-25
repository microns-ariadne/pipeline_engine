import json
import luigi
import os

from .pipeline import NP_DATASET
from ..parameters import Volume
from ..tasks.connected_components import \
     LogicalOperation, JoiningMethod, Direction, ConnectedComponentsTask, \
     AllConnectedComponentsTask
from ..tasks.utilities import to_hashable
from ..targets.volume_target import DestVolumeReader

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
    root_dir = luigi.Parameter(
        description="Directory for storing intermediate files")
    join_direction = luigi.EnumParameter(
        enum=Direction,
        description="The plane in which to join the components")
    min_block_overlap_area = luigi.IntParameter(
        description="Minimum overlap in the joining plane for blocks to be "
                    "considered.")
    min_block_overlap = luigi.IntParameter(
        description="Minimum overlap in the joining direction for blocks")
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
        return luigi.LocalTarget(self.output_location+".pipeline.done")

    def _overlaps(self, v1, v2):
        '''Return true if volume 1 overlaps volume 2'''
        overlap = 1
        for origin_key, size_key, enum_key in (
            ("x", "width", Direction.X),
            ("y", "height", Direction.Y),
            ("z", "depth", Direction.Z)):
            v1_0 = v1[origin_key]
            v1_1 = v1_0 + v1[size_key]
            v2_0 = v2[origin_key]
            v2_1 = v2_0 + v2[size_key]
            if v1_0 >= v2_1 or v2_0 >= v1_1:
                return False
            if enum_key != self.join_direction:
                overlap *= min(v1_1, v2_1) - max(v1_0, v2_0)
            elif min(v1_1, v2_1) - max(v1_0, v2_0) < self.min_block_overlap:
                return False
        return overlap >= self.min_block_overlap_area
    
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
        cg1 = json.load(open(self.component_graph_1, "r"))
        cg2 = json.load(open(self.component_graph_2, "r"))
        #
        # These are the block joins done by the individual pipelines. They
        # are re-fed into AllConnectedComponents for the next round.
        #
        joins_done = cg1["joins"] + cg2["joins"]
        joins_to_do = []
        #
        # Get the additional loading plans in volume 1. These were put there
        # in anticipation of us needing edge loading plans
        #
        d_locs = {}
        for volume, location in cg1["additional_locations"]:
            #
            # The overlap volume is the actual volume of the loading plan,
            # not that of the volume it's in. Get it from the plan itself
            #
            overlap_volume = DestVolumeReader(location).volume.to_dictionary()
            d_locs[to_hashable(overlap_volume)] = (location, volume)
        additional_locations_1 = list(cg1["additional_locations"])
        additional_locations_2 = []
        #
        # Find overlapping volumes by comparing volume #2 against volume # 1
        #
        for volume2, location2 in cg2["additional_locations"]:
            overlap2 = DestVolumeReader(location2).volume.to_dictionary()
            for overlap1, (location1, volume1) in d_locs.items():
                if self._overlaps(overlap1, overlap2):
                    v1 = Volume(**volume1)
                    v2 = Volume(**volume2)
                    filename = "connected-components-%d-%d-%d_%d-%d-%d.json" % (
                        v1.x, v1.y, v1.z, v2.x, v2.y, v2.z)
                    output_location = os.path.join(
                        self.root_dir, filename)
                    for vmatch, seg_location1 in cg1["locations"]:
                        if vmatch == volume1:
                            break
                    else:
                        raise Exception("No matching location for cutout")
                    for vmatch, seg_location2 in cg2["locations"]:
                        if vmatch == volume2:
                            break
                    else:
                        raise Exception("No matching location for cutout")
                    task = ConnectedComponentsTask(
                        volume1=v1,
                        cutout_loading_plan1_path=location1,
                        segmentation_loading_plan1_path=seg_location1,
                        volume2=v2,
                        cutout_loading_plan2_path=location2,
                        segmentation_loading_plan2_path=seg_location2,
                        output_location=output_location)
                    joins_to_do.append(task)
                    #
                    # Remove the location from the list of additional locations.
                    # It's inside the combined volume
                    #
                    for i, (volume1a, location1a) \
                        in enumerate(additional_locations_1):
                        if volume1a == volume1 and location1a == location1:
                            del additional_locations_1[i]
                            break
                    break
            else:
                #
                # If the location didn't match anything, put it on the list
                # for inclusion in the output
                #
                additional_locations_2.append((volume2, location2))
        #
        # Make the ultra-stupendous AllConnectedComponentsTask
        #
        all_join_files = [_[2] for _ in joins_done]
        all_join_files += [task.output().path for task in joins_to_do]
        self.all_connected_components_task = \
            AllConnectedComponentsTask(
                input_locations=all_join_files, 
                output_location=self.output_location,
                additional_loading_plans = 
                additional_locations_1 + additional_locations_2)
        for task in joins_to_do:
            self.all_connected_components_task.set_requirement(task)

    def requires(self):
        if not hasattr(self, "all_connected_components_task"):
            self.compute_requirements()
        yield self.all_connected_components_task

    def run(self):
        with self.output().open("w") as fd:
            fd.write("Done")
