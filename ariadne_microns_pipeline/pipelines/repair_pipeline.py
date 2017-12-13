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
                rh_logger.logger.start_logging(
                    "Repair pipeline", "starting", [])
            except:
                pass
            self.compute_requirements()
        return self.requirements
    
    def compute_requirements(self):
        self.idx = 1
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
        if not os.path.isdir(path):
            os.makedirs(path)
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
                local_mapping=mapping[:, 0].tolist(),
                global_mapping=mapping[:, 1].tolist(),
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