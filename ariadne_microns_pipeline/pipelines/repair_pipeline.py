import json
import luigi
import os
import rh_logger

from ..tasks.repair import RepairSegmentationTask
from ..tasks.utilities import to_hashable
from ..tasks.connected_components import \
     ConnectivityGraph, FakeConnectedComponentsTask, AllConnectedComponentsTask
from ..parameters import Volume, DEFAULT_LOCATION, EMPTY_LOCATION
from ..targets.volume_target import SrcVolumeTarget

class RepairPipeline(luigi.Task):
    '''Repair a segmentation
    
    This pipeline takes a downsampled repair volume and a list of
    segment IDs to repair. It writes global segmentation volumes for each
    local volume in the connectivity graph and also writes connected components
    files that link the segments in adjacent volumes. Finally, it rewrites
    the connected components file, mapping global IDs to themselves.
    '''
    task_namespace="ariadne_microns_pipeline"
    
    src_connectivity_graph = luigi.Parameter(
        description="The path to the original connectivity graph")
    dest_connectivity_graph = luigi.Parameter(
        description="The path to the new connectivity graph")
    destination = luigi.Parameter(
        description="This is the base of the file hierarchy of the new "
        "segmentation volumes")
    repair_file = luigi.Parameter(
        description="The name of the hdf5 file containing the downsampled "
        "repair volume.")
    repair_file_dataset_name = luigi.Parameter(
        default="stack",
        description="The name of the dataset within the hdf5 file")
    blood_vessel_file = luigi.Parameter(
        default = EMPTY_LOCATION,
        description="The file of areas to mask out (blood vessels)")
    blood_vessel_dataset_name = luigi.Parameter(
        default="stack",
        description="The name of the dataset within the blood vessel file")
    x_offset = luigi.IntParameter(
        default=0,
        description="Offset of the repair and blood vessel volumes in the "
        "x direction")
    y_offset = luigi.IntParameter(
        default=0,
        description="Offset of the repair and blood vessel volumes in the "
        "y direction")
    z_offset = luigi.IntParameter(
        default=0,
        description="Offset of the repair and blood vessel volumes in the "
        "z direction")
    x_upsampling=luigi.IntParameter(
        default=8,
        description="The amount of upsampling to apply to the repair "
        "segmentation in the X direction")
    y_upsampling=luigi.IntParameter(
        default=8,
        description="The amount of upsampling to apply to the repair "
        "segmentation in the Y direction")
    z_upsampling=luigi.IntParameter(
        default=2,
        description="The amount of upsampling to apply to the repair "
        "segmentation in the X direction")
    segments_to_repair=luigi.ListParameter(
        description="The global IDs of segments that should be repaired")
    repair_segments_to_exclude=luigi.ListParameter(
        default=[],
        description="Mask out these segments from the repair volume")
    done_file=luigi.Parameter(
        default=DEFAULT_LOCATION,
        description="The name of touchfile indicating that the task is done")

    def output(self):
        if self.done_file == DEFAULT_LOCATION:
            done_file = os.path.join(self.destination, "repair.done")
        else:
            done_file = self.done_file
        return luigi.LocalTarget(done_file)
    
    def requires(self):
        if not hasattr(self, "requirements"):
            try:
                rh_logger.logger.start_process(
                    "Repair pipeline", "starting", [])
            except:
                pass
            self.compute_requirements()
        return self.requirements
    
    def compute_requirements(self):
        self.idx = 1
        self.known_dirs = set()
        self.cg = ConnectivityGraph.load(open(self.src_connectivity_graph))
        self.jcg = json.load(open(self.src_connectivity_graph))
        rh_logger.logger.report_event("Writing loading plans")
        self.write_loading_plans()
        rh_logger.logger.report_event("Writing additional loading plans")
        self.write_additional_loading_plans()
        rh_logger.logger.report_event("Writing storage plans")
        self.write_storage_plans()
        rh_logger.logger.report_event("Creating repair tasks")
        self.create_repair_tasks()
        rh_logger.logger.report_event("Creating fake connected components tasks")
        self.create_cc_tasks()
    
    def to_dir(self, volume):
        '''Create a path to an appropriate directory for this volume'''
        path = os.path.join(
            self.destination,
            str(volume.x),
            str(volume.y),
            str(volume.z))
        if (path not in self.known_dirs) and not os.path.isdir(path):
            os.makedirs(path)
        self.known_dirs.add(path)
        return path

    def copy_loading_plan(self, lp, volume):
        location = os.path.join(
            self.to_dir(volume),
            "neuroproof_%d-%d_%d-%d_%d-%d_%d.loading.plan" %
            (volume.x, volume.x1,
             volume.y, volume.y1,
             volume.z, volume.z1, self.idx))
        jlp = json.load(open(lp))
        dataset_done_files = []
        assert len(jlp["dataset_done_files"]) == 1
        for sp in jlp["dataset_done_files"]:
            self.src_storage_plans[volume] = sp
            dataset_done_files.append(self.get_repair_storage_plan_path(
                volume))
        jlp["dataset_done_files"] = dataset_done_files
        blocks = []
        for tif_path, v in jlp["blocks"]:
            subvolume = Volume(**v)
            tif_path = os.path.join(self.to_dir(subvolume),
                                    os.path.split(tif_path)[1])
            blocks.append((tif_path, v))
        jlp["blocks"] = blocks
        json.dump(jlp, open(location, "w"))
        return location
        
    def write_loading_plans(self):
        self.src_loading_plans = {}
        self.src_storage_plans = {}
        self.loading_plans = {}
        for dvolume, lp in self.jcg["locations"]:
            volume = Volume(**dvolume)
            self.src_loading_plans[volume] = lp
            self.loading_plans[volume] = self.copy_loading_plan(lp, volume)
            self.idx += 1
        
    def write_additional_loading_plans(self):
        self.additional_locations = []
        for d in self.jcg["additional_locations"]:
            lp = d["loading_plan"]
            volume = Volume(**d["volume"])
            location = self.copy_loading_plan(lp, volume)
            d["loading_plan"] = location
            self.additional_locations.append(d)
    
    def get_repair_storage_plan_path(self, volume):
        filename = "neuroproof_%d-%d_%d-%d_%d-%d.storage.plan" % (
            volume.x, volume.x1, volume.y, volume.y1, volume.z, volume.z1)
        return os.path.join(self.to_dir(volume), filename)
    
    def write_storage_plans(self):
        self.storage_plans = {}
        for volume, sp_path in self.src_storage_plans.items():
            sp = json.load(open(sp_path))
            location = self.get_repair_storage_plan_path(volume)
            blocks = []
            for v, tif_path in sp["blocks"]:
                subvolume = Volume(**v)
                tif_path = os.path.join(self.to_dir(subvolume),
                                        os.path.split(tif_path)[1])
                blocks.append((v, tif_path))
            sp["blocks"] = blocks
            self.storage_plans[volume] = location
            json.dump(sp, open(location, "w"))
        
    def create_repair_tasks(self):
        self.repair_tasks = {}
        for volume, lp in self.src_loading_plans.items():
            sp = self.storage_plans[volume]
            mapping = self.cg.volumes[to_hashable(volume)]
            mapping_file = os.path.join(self.to_dir(volume), "mappings.json")
            d = { "local":mapping[:, 0].tolist(),
                  "global":mapping[:, 1].tolist() }
            json.dump(d, open(mapping_file, "w"))
            task = RepairSegmentationTask(
                storage_plan=sp,
                segmentation_loading_plan_path=lp,
                repair_file=self.repair_file,
                repair_file_dataset_name=self.repair_file_dataset_name,
                blood_vessel_file=self.blood_vessel_file,
                blood_vessel_dataset_name=self.blood_vessel_dataset_name,
                x_offset=self.x_offset,
                y_offset=self.y_offset,
                z_offset=self.z_offset,
                x_upsampling=self.x_upsampling,
                y_upsampling=self.y_upsampling,
                z_upsampling=self.z_upsampling,
                mapping_file=mapping_file,
                segments_to_repair=self.segments_to_repair,
                repair_segments_to_exclude=self.repair_segments_to_exclude)
            self.repair_tasks[volume] = task
        
    def create_cc_tasks(self):
        self.requirements = []
        self.cc_files = []
        self.cc_tasks = []
        for dvolume1, dvolume2, src_cc_path in self.jcg["joins"]:
            volume1 = Volume(**dvolume1)
            volume2 = Volume(**dvolume2)
            lp1 = self.loading_plans[volume1]
            lp2 = self.loading_plans[volume2]
            rtask1 = self.repair_tasks[volume1]
            rtask2 = self.repair_tasks[volume2]
            cc_path = os.path.join(
                self.to_dir(volume1),
                "connected-components_%d-%d-%d_%d-%d-%d.json" %
                (volume1.x, volume1.y, volume1.z, 
                 volume2.x, volume2.y, volume2.z))
            self.cc_files.append(cc_path)
            task = FakeConnectedComponentsTask(
                segmentation_loading_plan1_path=lp1,
                segmentation_loading_plan2_path=lp2,
                output_location=cc_path)
            task.set_requirement(rtask1)
            task.set_requirement(rtask2)
            self.cc_tasks.append(task)
        task = AllConnectedComponentsTask(
            input_locations = self.cc_files,
            output_location=self.dest_connectivity_graph,
            additional_loading_plans=self.additional_locations,
            metadata=self.jcg["metadata"])
        for cc_task in self.cc_tasks:
            task.set_requirement(cc_task)
        self.requirements.append(task)
    
    def run(self):
        with self.output().open("w") as fd:
            fd.write("Repair finished successfully")

from .pipeline import \
     SYNAPSE_DATASET, SYNAPSE_RECEPTOR_DATASET, SYNAPSE_TRANSMITTER_DATASET
from ..tasks.connect_synapses \
     import ConnectSynapsesTask, AggregateSynapseConnectionsTask
import glob

class SynapseRepairPipeline(luigi.Task):
    '''Rebase the synapses to a new segmentation
    
    After repairing, the segmentation is completely different. Fix the
    synapses by running ConnectSynapsesTask on the blocks again.
    '''
    task_namespace="ariadne_microns_pipeline"
    
    src_connectivity_graph=luigi.Parameter(
        description="Connectivity graph for the original segmentation")
    dest_connectivity_graph=luigi.Parameter(
        description="Connectivity graph for the new segmentation")
    destination=luigi.Parameter(
        description="Root directory for the synapse connectivity files")
    synapse_connections_file=luigi.Parameter(
        description="The destination for the global synapse connections file")
    done_file=luigi.Parameter(
        default=DEFAULT_LOCATION,
        description="Location of the touchfile indicating pipeline is done")
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

    
    def output(self):
        if self.done_file == DEFAULT_LOCATION:
            done_file = os.path.join(self.destination, "repair.done")
        else:
            done_file = self.done_file
        return luigi.LocalTarget(done_file)

    def to_dir(self, volume):
        '''Create a path to an appropriate directory for this volume'''
        path = os.path.join(
            self.destination,
            str(volume.x),
            str(volume.y),
            str(volume.z))
        if (path not in self.known_dirs) and not os.path.isdir(path):
            os.makedirs(path)
        self.known_dirs.add(path)
        return path
   
    def requires(self):
        if not hasattr(self, "requirements"):
            try:
                rh_logger.logger.start_process(
                    "Repair pipeline", "starting", [])
            except:
                pass
            try:
                self.compute_requirements()
            except:
                rh_logger.logger.report_exception()
                raise
        return self.requirements
    
    def compute_requirements(self):
        self.known_dirs = set()
        self.src_cg = ConnectivityGraph.load(open(self.src_connectivity_graph))
        rh_logger.logger.report_event("Loaded %s" % self.src_connectivity_graph)
        self.dest_cg = ConnectivityGraph.load(
            open(self.dest_connectivity_graph))
        rh_logger.logger.report_event("Finding synapse loading plans")
        self.find_loading_plans()
        rh_logger.logger.report_event("Making tasks")
        self.make_tasks()
        rh_logger.logger.report_event("Finished with task setup")
        
    def find_loading_plans(self):
        self.synapse_loading_plans = {}
        self.transmitter_loading_plans = {}
        self.receptor_loading_plans = {}
        for volume, location in self.src_cg.locations.items():
            loading_plans = glob.glob(
                os.path.join(os.path.dirname(location), "[str]*.loading.plan"))
            for lp in loading_plans:
                lpfile = os.path.split(lp)[1]
                if lpfile.startswith(SYNAPSE_DATASET):
                    self.synapse_loading_plans[volume] = lp
                elif lpfile.startswith(SYNAPSE_TRANSMITTER_DATASET):
                    self.transmitter_loading_plans[volume] = lp
                elif lpfile.startswith(SYNAPSE_RECEPTOR_DATASET):
                    self.receptor_loading_plans[volume] = lp
    
    def make_tasks(self):
        self.cs_tasks = []
        self.cs_locs = []
        for volume, location in self.src_cg.locations.items():
            dest = os.path.join(self.to_dir(volume), "synapse-connections.json")
            self.cs_locs.append(dest)
            task = ConnectSynapsesTask(
                neuron_seg_load_plan_path=location,
                synapse_seg_load_plan_path=self.synapse_loading_plans[volume],
                transmitter_probability_map_load_plan_path=
                    self.transmitter_loading_plans[volume],
                receptor_probability_map_load_plan_path=
                    self.receptor_loading_plans[volume],
                output_location=dest,
                xy_dilation=self.xy_dilation,
                z_dilation=self.z_dilation,
                min_contact=self.min_contact,
                wants_edge_contact=self.wants_edge_contact,
                x_nm=self.x_nm,
                y_nm=self.y_nm,
                z_nm=self.z_nm,
                distance_from_centroid=self.distance_from_centroid)
            self.cs_tasks.append(task)
        
        aggtask = AggregateSynapseConnectionsTask(
            synapse_connection_locations=self.cs_locs,
            connectivity_graph_location=self.dest_connectivity_graph,
            output_location=self.synapse_connections_file,
            xy_nm=self.x_nm,
            z_nm=self.z_nm,
            min_distance_nm=self.min_distance_nm,
            min_distnace_identical_nm=self.min_distance_identical_nm)
        for task in self.cs_tasks:
            aggtask.set_requirement(task)
        self.requirements = [aggtask]
        
    def run(self):
        with self.output().open("w") as fd:
            fd.write("Repair finished successfully")
