import json
import luigi
import numpy as np
import os
import rh_config

from .pipeline import NP_DATASET, CHIMERA_INPUT_DATASET, CHIMERA_OUTPUT_DATASET
from .pipeline import MEMBRANE_DATASET
from ..parameters import Volume, EMPTY_LOCATION
from ..tasks.connected_components import \
     LogicalOperation, JoiningMethod, Direction, ConnectedComponentsTask, \
     AllConnectedComponentsTask, AdditionalLocationDirection, \
     AdditionalLocationType, ConnectivityGraph
from ..tasks.copytasks import ChimericSegmentationTask, CopyLoadingPlansTask
from ..tasks.neuroproof import NeuroproofTask, NeuroproofVersion
from ..tasks.utilities import to_hashable
from ..targets.volume_target import DestVolumeReader, \
     write_simple_loading_plan, write_simple_storage_plan, \
     write_compound_storage_plan

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
    direction1 = luigi.EnumParameter(
        enum=AdditionalLocationDirection,
        description="The side of volume #1 to join, e.g. X1 for the left.")
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
    exclude1 = luigi.ListParameter(
        default=[],
        description="The segment IDs to exclude from block # 1. These should be global "
        "IDs in the block's component graph")
    exclude2 = luigi.ListParameter(
        default=[],
        description="The segment IDs to exclude from block # 2. These should be global "
        "IDs in the block's component graph")
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
    partner_min_total_area_ratio = luigi.FloatParameter(
        default=0.001)
    max_poly_matches = luigi.IntParameter(
        default=1)
    dont_join_orphans = luigi.BoolParameter()
    orphan_min_overlap_ratio = luigi.FloatParameter(
        default=0.9)
    orphan_min_overlap_volume = luigi.IntParameter(
        default=1000,
        description="The minimum acceptable volume in voxels of overlap "
                    "needed to join an orphan segment.")
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
    neuroproof_classifier = luigi.Parameter(
        default=EMPTY_LOCATION,
        description="The classifier to use for neuroproofing in the ABUT "
        "join method.")
    membrane_neuroproof_channel = luigi.Parameter(
        default=MEMBRANE_DATASET,
        description="The name of the primary neuroproof probability "
        "channel")
    additional_neuroproof_channels = luigi.ListParameter(
        default=[],
        description="The names of additional classifier classes "
                    "that are fed into Neuroproof as channels")
    neuroproof_version = luigi.EnumParameter(
        enum=NeuroproofVersion,
        default=NeuroproofVersion.FAST,
        description="The command-line convention to be used to run the "
        "Neuroproof binary")
    np_threshold = luigi.FloatParameter(
        default=.2,
        description="The probability threshold for merging in Neuroproof "
        "(range = 0-1).")
    np_cores = luigi.IntParameter(
        description="The number of cores used by a Neuroproof process",
        default=1)
    prune_feature = luigi.BoolParameter(
        description="Automatically prune useless features")
    use_mito = luigi.BoolParameter(
        description="Set delayed mito agglomeration")
    
    def inputs(self):
        yield luigi.LocalTarget(self.component_graph_1)
        yield luigi.LocalTarget(self.component_graph_2)
        
    def output(self):
        return luigi.LocalTarget(self.output_location+".pipeline.done")

    def find_local_exclusions(self, exclusions, volume, cg):
        '''Convert global to local exclusions
        
        :param exclusions: The exclusions using global IDs
        :param volume: the volume to be joined
        :param cg: the connectivity graph of the volume to be joined
        :returns: a list of local IDs for the exclusions
        '''
        if len(exclusions) == 0:
            return []
        result = []
        connections = cg.volumes.get(to_hashable(volume.to_dictionary()))
        if connections is None:
            return result
        for exclusion in exclusions:
            idxs = np.where(connections[:, 1] == exclusion)[0]
            if len(idxs) > 0:
                result.append(connections[idxs[0], 0])
        return result
    
    def validate(self, cg1, cg2):
        '''Make sure that cg1 and cg2 are compatible
        
        Check that their paddings are the same.
        Check that the volumes overlap properly
        '''
        for key in "np_x_pad", "np_y_pad", "np_z_pad":
            assert cg1.metadata[key] == cg2.metadata[key], \
                   "%s differs: 1=%s, 2=%s" % (
                       key, cg1.metadata[key], cg2.metadata[key])
        
    def compute_requirements(self):
        '''Compute the requirements and dependencies for the pipeline
        
        The AllConnectedComponentsTasks must have been run at this point.
        '''
        self.direction2 = self.direction1.opposite()
        cg1 = json.load(open(self.component_graph_1, "r"))
        if len(self.exclude1) > 0:
            connectivity_graph1 = \
                ConnectivityGraph.load(open(self.component_graph_1))
        else:
            connectivity_graph1 = None
        cg2 = json.load(open(self.component_graph_2, "r"))
        if len(self.exclude2) > 0:
            connectivity_graph2 = \
                ConnectivityGraph.load(open(self.component_graph_2))
        else:
            connectivity_graph2 = None
        self.validate(cg1, cg2)
        #
        # These are the block joins done by the individual pipelines. They
        # are re-fed into AllConnectedComponents for the next round.
        #
        joins_done = cg1["joins"] + cg2["joins"]
        joins_to_do = []
        #
        # Sort the additional locations by their direction. X0 matches with X1,
        # Y0 with Y1 and Z0 with Z1.
        #
        # The dictionary keys are the direction and location type enums
        # and the values are dictionaries that are the x, y and z of the
        # extents of their loading plans.
        #
        #
        # For abutting, use the ABUTTING location type, otherwise use the
        # OVERLAPPING type for matching. 
        #
        if self.joining_method == JoiningMethod.ABUT:
            tgt_location_type = AdditionalLocationType.ABUTTING
        else:
            tgt_location_type = AdditionalLocationType.OVERLAPPING
        al1_by_direction = {}
        al2_by_direction = {}
        for d, cg in ((al1_by_direction, cg1), 
                      (al2_by_direction, cg2)):
            for al in cg["additional_locations"]:
                direction = AdditionalLocationDirection[al["direction"]]
                location_type = AdditionalLocationType[al["location_type"]]
                if location_type != tgt_location_type:
                    continue
                dlkey = direction
                if dlkey not in d:
                    d[dlkey] = {}
                extent = Volume(**al["extent"])
                if self.joining_method == JoiningMethod.ABUT:
                    if direction == AdditionalLocationDirection.X1:
                        xyzkey = (extent.x1, extent.y, extent.z)
                    elif direction == AdditionalLocationDirection.Y1:
                        xyzkey = (extent.x, extent.y1, extent.z)
                    elif direction == AdditionalLocationDirection.Z1:
                        xyzkey = (extent.x, extent.y, extent.z1)
                    else:
                        xyzkey = (extent.x, extent.y, extent.z)
                else:
                    xyzkey = (extent.x, extent.y, extent.z)
                d[dlkey][xyzkey] = al
        #
        # Make a dictionary of direction and volume to the MATCHING
        # loading plan if using the abutting method. Key is the direction +
        # x, y, z of the parent volume.
        #
        if self.joining_method == JoiningMethod.ABUT:
            al1_by_volume = {}
            al2_by_volume = {}
            channels_by_volume = {}
            for d, cg in ((al1_by_volume, cg1),
                          (al2_by_volume, cg2)):
                for al in cg["additional_locations"]:
                    direction = AdditionalLocationDirection[al["direction"]]
                    location_type = AdditionalLocationType[al["location_type"]]
                    volume = Volume(**al["volume"])
                    key = (direction, volume.x, volume.y, volume.z)
                    if location_type == AdditionalLocationType.MATCHING:
                        d[key] = al
            for al in cg1["additional_locations"]:
                direction = AdditionalLocationDirection[al["direction"]]
                location_type = AdditionalLocationType[al["location_type"]]
                volume = Volume(**al["volume"])
                key = (direction, volume.x, volume.y, volume.z)
                if location_type == AdditionalLocationType.CHANNEL:
                    if key not in channels_by_volume:
                        channels_by_volume[key] = {}
                    channels_by_volume[key][al["channel"]] = al
        #
        # Make a dictionary of the parent segmentations
        # The key is the x, y, z of the block
        #
        seg_loading_plans1 = {}
        seg_loading_plans2 = {}
        for seg_loading_plans, cg in ((seg_loading_plans1, cg1),
                                      (seg_loading_plans2, cg2)):
            for volume, loading_plan in cg["locations"]:
                volume = Volume(**volume)
                seg_loading_plans[volume.x, volume.y, volume.z] = loading_plan

        unused_additional_locations = [] 
        #
        # For abut, we need to get the neuroproof config
        #
        if self.joining_method == JoiningMethod.ABUT:
            config = rh_config.config["neuroproof"]
            neuroproof = config["neuroproof_graph_predict"]
            ld_library_path = os.pathsep.join(config.get("ld_library_path", []))
            
        #
        # Find matching volumes between datasets
        #
        d1 = self.direction1
        d2 = self.direction2
        ald1 = al1_by_direction[d1]
        ald2 = al2_by_direction[d2]
        for xyzkey, al1 in ald1.items():
            if xyzkey not in ald2:
                unused_additional_locations.append(al1)
            else:
                al2 = ald2[xyzkey]
                del ald2[xyzkey]
                v1 = Volume(**al1["volume"])
                v2 = Volume(**al2["volume"])
                e1 = Volume(**al1["extent"])
                e2 = Volume(**al2["extent"])
                cutout_loading_plan1_path = al1["loading_plan"]
                cutout_loading_plan2_path = al2["loading_plan"]
                segmentation_loading_plan1_path = \
                    seg_loading_plans1[v1.x, v1.y, v1.z]
                segmentation_loading_plan2_path = \
                    seg_loading_plans2[v2.x, v2.y, v2.z]
                filename = "connected-components-%d-%d-%d_%d-%d-%d.json" % (
                    v1.x, v1.y, v1.z, v2.x, v2.y, v2.z)
                output_location = os.path.join(
                    self.root_dir, filename)
                exclude1 = self.find_local_exclusions(
                    self.exclude1, v1, connectivity_graph1)
                exclude2 = self.find_local_exclusions(
                    self.exclude2, v2, connectivity_graph2)
                if self.joining_method != JoiningMethod.ABUT:
                    task = ConnectedComponentsTask(
                        volume1=v1,
                        cutout_loading_plan1_path=
                        cutout_loading_plan1_path,
                        segmentation_loading_plan1_path=
                        segmentation_loading_plan1_path,
                        volume2=v2,
                        cutout_loading_plan2_path=
                        cutout_loading_plan2_path,
                        segmentation_loading_plan2_path=
                        segmentation_loading_plan2_path,
                        exclude1=exclude1,
                        exclude2=exclude2,
                        min_overlap_percent=self.min_overlap_percent,
                        operation=self.operation,
                        joining_method=self.joining_method,
                        min_overlap_volume=self.min_overlap_volume,
                        max_poly_matches=self.max_poly_matches,
                        dont_join_orphans=self.dont_join_orphans,
                        orphan_min_overlap_ratio=self.orphan_min_overlap_ratio,
                        orphan_min_overlap_volume=self.orphan_min_overlap_volume,
                        output_location=output_location)
                else:
                    #
                    # The slivers to be matched
                    #
                    sliverd1 = al1_by_volume[d1, v1.x, v1.y, v1.z]
                    sliverd2 = al2_by_volume[d2, v2.x, v2.y, v2.z]
                    slivere1 = Volume(**sliverd1["extent"])
                    slivere2 = Volume(**sliverd1["extent"])
                    sliverlp1 = sliverd1["loading_plan"]
                    sliverlp2 = sliverd2["loading_plan"]
                    #
                    # Pipeline is:
                    #
                    # Create the chimeric segmentation
                    # Copy the membrane and other prediction channels
                    #      some from one volume and some from the other.
                    # Neuroproof it
                    # Run connected components using the abut method
                    #
                    cvolume = e1.get_union_region(e2)
                    #
                    # The storage plan for the chimera segmentation
                    #
                    chimeric_storage_plan_path = os.path.join(
                        self.root_dir, 
                        str(cvolume.x), str(cvolume.y), str(cvolume.z),
                        "%s_%s_%d-%d_%d-%d_%d-%d.storage.plan" % (
                            CHIMERA_INPUT_DATASET, d1,
                            cvolume.x, cvolume.x1,
                            cvolume.y, cvolume.y1,
                            cvolume.z, cvolume.z1))
                    chimeric_tif_path = os.path.join(
                        self.root_dir, 
                        str(cvolume.x), str(cvolume.y), str(cvolume.z),
                        "%s_%s_%d-%d_%d-%d_%d-%d.tif" % (
                            CHIMERA_INPUT_DATASET, d1,
                            cvolume.x, cvolume.x1,
                            cvolume.y, cvolume.y1,
                            cvolume.z, cvolume.z1))
                    dir_path = os.path.dirname(chimeric_storage_plan_path)
                    if not os.path.isdir(dir_path):
                        os.makedirs(dir_path)
                    write_simple_storage_plan(
                        chimeric_storage_plan_path,
                        chimeric_tif_path,
                        cvolume, CHIMERA_INPUT_DATASET,
                        "uint32")
                    #
                    # The plan for loading it in Neuroproof
                    #
                    chimeric_load_plan_path = os.path.join(
                        self.root_dir, 
                        str(cvolume.x), str(cvolume.y), str(cvolume.z),
                        "%s_%s_%d-%d_%d-%d_%d-%d.loading.plan" % (
                            CHIMERA_INPUT_DATASET, d1,
                            cvolume.x, cvolume.x1,
                            cvolume.y, cvolume.y1,
                            cvolume.z, cvolume.z1))
                    write_simple_loading_plan(
                        chimeric_load_plan_path,
                        chimeric_tif_path,
                        cvolume,
                        CHIMERA_INPUT_DATASET,
                        "uint32")
                    #
                    # The plan for storing the slivers of the Neuroproof
                    # segmentation.
                    #
                    # First calculate the three volumes making up the
                    # plan: left, sliver and right. Left and right are
                    # not used and only for troubleshooting the entire
                    # neuroproofing.
                    #
                    npvolume = slivere1.get_union_region(slivere2)
                    if AdditionalLocationDirection.X0 in (d1, d2):
                        left_volume = Volume(
                            cvolume.x, cvolume.y, cvolume.z,
                            npvolume.x-cvolume.x, 
                            cvolume.height, cvolume.depth)
                        right_volume = Volume(
                            npvolume.x1, cvolume.y, cvolume.z,
                            cvolume.x1 - npvolume.x1, 
                            cvolume.height, cvolume.depth)
                    elif AdditionalLocationDirection.Y0 in (d1, d2):
                        left_volume = Volume(
                            cvolume.x, cvolume.y, cvolume.z,
                            cvolume.width,
                            npvolume.y-cvolume.y, 
                            cvolume.depth)
                        right_volume = Volume(
                            cvolume.x, npvolume.y1, cvolume.z,
                            cvolume.width,
                            cvolume.y1 - npvolume.y1, 
                            cvolume.depth)
                    else:
                        left_volume = Volume(
                            cvolume.x, cvolume.y, cvolume.z,
                            cvolume.width, cvolume.height,
                            npvolume.z - cvolume.z)
                        right_volume = Volume(
                            cvolume.x, cvolume.y, npvolume.z1,
                            cvolume.width, cvolume.height,
                            cvolume.z1 - npvolume.z1)
                    #
                    # Then build a complex storage plan and a loading
                    # plan for the slivers
                    #
                    np_storage_plan_path = os.path.join(
                        self.root_dir, 
                        str(cvolume.x), str(cvolume.y), str(cvolume.z),
                        "%s_%s_%d-%d_%d-%d_%d-%d.storage.plan" % (
                            CHIMERA_OUTPUT_DATASET, d1,
                            cvolume.x, cvolume.x1,
                            cvolume.y, cvolume.y1,
                            cvolume.z, cvolume.z1))
                    np_loading_plan_path = os.path.join(
                        self.root_dir, 
                        str(cvolume.x), str(cvolume.y), str(cvolume.z),
                        "%s_%s_%d-%d_%d-%d_%d-%d.loading.plan" % (
                            CHIMERA_OUTPUT_DATASET, d1,
                            npvolume.x, npvolume.x1,
                            npvolume.y, npvolume.y1,
                            npvolume.z, npvolume.z1))
                    left_tif_path = os.path.join(
                        self.root_dir, 
                        str(left_volume.x), str(left_volume.y), 
                        str(left_volume.z),
                        "%s_%s_%d-%d_%d-%d_%d-%d.tif" % (
                            CHIMERA_OUTPUT_DATASET, d1,
                            left_volume.x, left_volume.x1,
                            left_volume.y, left_volume.y1,
                            left_volume.z, left_volume.z1))
                    right_tif_path = os.path.join(
                        self.root_dir, 
                        str(right_volume.x), str(right_volume.y), 
                        str(right_volume.z),
                        "%s_%s_%d-%d_%d-%d_%d-%d.tif" % (
                            CHIMERA_OUTPUT_DATASET, d1,
                            right_volume.x, right_volume.x1,
                            right_volume.y, right_volume.y1,
                            right_volume.z, right_volume.z1))
                    np_tif_path = os.path.join(
                        self.root_dir, 
                        str(npvolume.x), str(npvolume.y), 
                        str(npvolume.z),
                        "%s_%s_%d-%d_%d-%d_%d-%d.tif" % (
                            CHIMERA_OUTPUT_DATASET, d1,
                            npvolume.x, npvolume.x1,
                            npvolume.y, npvolume.y1,
                            npvolume.z, npvolume.z1))
                    write_compound_storage_plan(
                        np_storage_plan_path,
                        [left_tif_path, np_tif_path, right_tif_path],
                        [left_volume, npvolume, right_volume],
                        cvolume, CHIMERA_OUTPUT_DATASET, "uint32")
                    write_simple_loading_plan(
                        np_loading_plan_path,
                        np_tif_path,
                        npvolume, CHIMERA_OUTPUT_DATASET, "uint32")
                    #
                    # The chimera-building task
                    #
                    ctask = ChimericSegmentationTask(
                        loading_plan1_path=sliverlp1,
                        loading_plan2_path=sliverlp2,
                        storage_plan=chimeric_storage_plan_path)
                    #
                    # The neuroproof task
                    #
                    d = channels_by_volume[d1, volume.x, volume.y, volume.z]
                    membrane_lp = \
                        d[self.membrane_neuroproof_channel]["loading_plan"]
                    additional_loading_plans = [
                        d[_]["loading_plan"]
                        for _ in self.additional_neuroproof_channels]
                    nptask = NeuroproofTask(
                        storage_plan=np_storage_plan_path,
                        prob_loading_plan_path=membrane_lp,
                        additional_loading_plan_paths=
                            additional_loading_plans,
                        input_seg_loading_plan_path=chimeric_load_plan_path,
                        neuroproof=neuroproof,
                        neuroproof_ld_library_path=ld_library_path,
                        classifier_filename=self.neuroproof_classifier,
                        threshold=self.np_threshold,
                        neuroproof_version=self.neuroproof_version,
                        cpu_count=self.np_cores)
                    nptask.set_requirement(ctask)
                    #
                    # The connected components task
                    #
                    task = ConnectedComponentsTask(
                        volume1=v1,
                        cutout_loading_plan1_path=sliverlp1,
                        segmentation_loading_plan1_path=
                        segmentation_loading_plan1_path,
                        volume2=v2,
                        cutout_loading_plan2_path=sliverlp2,
                        segmentation_loading_plan2_path=
                        segmentation_loading_plan2_path,
                        neuroproof_segmentation=np_loading_plan_path,
                        output_location=output_location,
                        joining_method=JoiningMethod.ABUT)
                    
                joins_to_do.append(task)
        #
        # Filter the additional locations
        #
        additional_locations = filter(
            lambda _:AdditionalLocationDirection[_["direction"]] != d1,
            cg1["additional_locations"]) + filter(
            lambda _:AdditionalLocationDirection[_["direction"]] != d2,
            cg2["additional_locations"])
        #
        # Make the ultra-stupendous AllConnectedComponentsTask
        #
        metadata = {}
        metadata["1"] = cg1.metadata
        metadata["2"] = cg2.metadata
        # merge identical metadata from left & right
        for key, value in cg1.metadata.values():
            if key in cg2.metadata and cg2.metadata[key] == value:
                metadata[key] == value
        
        all_join_files = [_[2] for _ in joins_done]
        all_join_files += [task.output().path for task in joins_to_do]
        self.all_connected_components_task = \
            AllConnectedComponentsTask(
                input_locations=all_join_files, 
                output_location=self.output_location,
                additional_loading_plans = additional_locations)
        self.all_connected_components_task.metadata = metadata
        for task in joins_to_do:
            self.all_connected_components_task.set_requirement(task)
    
    def abuts(self, a, b):
        '''Return the abutting direction if volume A abuts volume B, else None
        
        '''
        x_same = a.x == b.x and a.x1 == b.x1
        y_same = a.y == b.y and a.y1 == b.y1
        z_same = a.z == b.z and a.z1 == b.z1
        if (a.x1 == b.x or a.x == b.x1) and y_same and z_same:
            return Direction.X
        if (a.y1 == b.y or a.y == b.y1) and x_same and z_same:
            return Direction.Y
        if (a.z1 == b.z or a.z == b.z1) and x_same and y_same:
            return Direction.Z
        
    def compute_requirements_abut(self):
        '''Compute requirements for the abutting method'''
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
        # A location abuts another if 

    def requires(self):
        if not hasattr(self, "all_connected_components_task"):
            self.compute_requirements()
        yield self.all_connected_components_task

    def run(self):
        with self.output().open("w") as fd:
            fd.write("Done")
