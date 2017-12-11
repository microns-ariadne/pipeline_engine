# Merge predictions, e.g. affinity, into membrane
#
import glob
import json
import luigi
import numpy as np
import os
import rh_logger

from ..tasks.merge_predictions import MergeOperation, MergePredictionsTask
from ..tasks.connected_components import ConnectivityGraph
from ..targets.volume_target import DestVolumeReader, SrcVolumeTarget
from ..targets.volume_target import write_loading_plan, write_storage_plan
from ..tasks.utilities import to_hashable, to_json_serializable

class MergePredictionsPipeline(luigi.Task):
    task_namespace="ariadne_microns_pipeline"
    
    operation = luigi.EnumParameter(
        enum=MergeOperation,
        default=MergeOperation.Average,
        description="The operation to perform")
    invert = luigi.BoolParameter(
        description="Subtract the result from the maximum allowed value")
    connectivity_graph_path = luigi.Parameter(
        description="The location of the connectivity graph")
    input_dataset_names = luigi.ListParameter(
        description="The dataset names of the inputs to be merged.")
    output_dataset_name = luigi.Parameter(
        description="The dataset name of the outputs to be generated.")
    index_file_name = luigi.Parameter(
        description="The name of the index file containing the ouput "
        "dataset's loading and storage plans")
    
    def output(self):
        return luigi.LocalTarget(self.index_file_name)
    
    def requires(self):
        if not hasattr(self, "requirements"):
            try:
                rh_logger.logger.start_process(
                    "MergePredictions", "starting", [])
            except:
                pass
            self.compute_requirements()
        return self.requirements
    
    def compute_requirements(self):
        self.cg = ConnectivityGraph.load(open(self.connectivity_graph_path))
        #
        # Find the loading plans of the input channels
        #
        rh_logger.logger.report_event("Finding input channel loading plans")
        self.find_input_channel_loading_plans()
        #
        # Find the storage plans of the input channels. These get used
        # to write loading plans that match the storage plans and to
        # write storage plans for the output channel.
        #
        rh_logger.logger.report_event("Finding input channel storage plans")
        self.find_input_channel_storage_plans()
        #
        # Write the loading plans for the input channels
        #
        rh_logger.logger.report_event("Writing input channel loading plans")
        self.write_input_channel_loading_plans()
        #
        # Write the storage plans for the output channel
        #
        rh_logger.logger.report_event("Writing output channel storage plans")
        self.write_output_channel_storage_plans()
        #
        # Write the loading plans for the output channel
        #
        rh_logger.logger.report_event("Write output channel loading plans")
        self.write_output_loading_plans()
        #
        # Make the needed tasks
        #
        self.requirements = self.make_merge_tasks()
    
    def find_input_channel_loading_plans(self):
        self.input_channel_lps = dict(
            [(channel, {}) for channel in self.input_dataset_names])
        #
        # We get the input channel loading plans from the 
        # Neuroproof loading plans by hacking their names
        #
        for volume, location in self.cg.locations.items():
            location_dir = os.path.dirname(location)
            paths = glob.glob(os.path.join(location_dir, "*.loading.plan"))
            for channel in self.input_dataset_names:
                for path in paths:
                    if os.path.split(path)[1].startswith(channel):
                        self.input_channel_lps[channel][volume] = path
    
    def find_input_channel_storage_plans(self):
        self.input_channel_sps = dict(
            [(channel, {}) for channel in self.input_dataset_names])
        #
        # We enumerate all the storage plans in each loading plan
        #
        for channel in self.input_dataset_names:
            d = self.input_channel_sps[channel]
            for volume, lp in self.input_channel_lps[channel].items():
                for sp in DestVolumeReader(lp).get_source_targets():
                    d[to_hashable(sp.volume)] = sp.storage_plan_path
                    
        
    def write_input_channel_loading_plans(self):
        '''Write loading plans that mirror the input channel storage plans'''
        self.input_channel_block_lps = dict(
            [(channel, {}) for channel in self.input_dataset_names])
        for channel in self.input_dataset_names:
            d = self.input_channel_block_lps[channel]
            for volume, sp in self.input_channel_sps[channel].items():
                sp_dir = os.path.dirname(sp)
                lp_path = os.path.join(
                    sp_dir,
                    "%s_%d-%d_%d-%d_%d-%d.loading_plan" % 
                     (channel, volume.x, volume.x1, volume.y, volume.y1,
                      volume.z, volume.z1))
                d[volume] = lp_path
                storage_plan = SrcVolumeTarget(sp)
                storage_plan.write_loading_plan(lp_path)
    
    def write_output_channel_storage_plans(self):
        '''Write a storage plan for each block to be merged'''
        self.output_channel_storage_plans = {}
        #
        # Copy channel 0's storage plan
        #
        ch0 = self.input_dataset_names[0]
        for volume, sp in self.input_channel_sps[ch0].items():
            spd = json.load(open(sp))
            sp_dir, sp0_file = os.path.split(sp)
            sp_file = "%s_%d-%d_%d-%d_%d-%d.storage.plan" % (
                self.output_dataset_name, 
                spd["x"], spd["x"] + spd["dimensions"][2],
                spd["y"], spd["y"] + spd["dimensions"][1],
                spd["z"], spd["z"] + spd["dimensions"][0])
            sp_path = os.path.join(sp_dir, sp_file)
            spd["dataset_name"] = self.output_dataset_name
            blocks = spd["blocks"]
            spd["blocks"] = []
            for v, tif_path in blocks:
                tif_file = "%s_%d-%d_%d-%d_%d-%d.tif" % (
                    self.output_dataset_name, 
                    v["x"], v["x"] + v["width"],
                    v["y"], v["y"] + v["height"],
                    v["z"], v["z"] + v["depth"])
                tif_path = os.path.join(os.path.dirname(tif_path), tif_file)
                spd["blocks"].append((v, tif_path))
            json.dump(spd, open(sp_path, "w"))
            self.output_channel_storage_plans[volume] = sp_path
    
    def write_output_loading_plans(self):
        '''Write loading plans for the output channel based on the input lps'''
        
        self.output_channel_loading_plans = {}
        #
        # Copy channel 0's loading plans
        #
        ch0 = self.input_dataset_names[0]
        for volume, lp in self.input_channel_lps[ch0].items():
            lpd = json.load(open(lp))
            lp_dir, lp0_file = os.path.split(lp)
            lp_file = "%s_%d-%d_%d-%d_%d-%d.loading.plan" % (
                self.output_dataset_name, 
                lpd["x"], lpd["x"] + lpd["dimensions"][2],
                lpd["y"], lpd["y"] + lpd["dimensions"][1],
                lpd["z"], lpd["z"] + lpd["dimensions"][0])
            lp_path = os.path.join(lp_dir, lp_file)
            lpd["dataset_name"] = self.output_dataset_name
            blocks = lpd["blocks"]
            lpd["blocks"] = []
            for tif_path, v in blocks:
                tif_file = "%s_%d-%d_%d-%d_%d-%d.tif" % (
                    self.output_dataset_name, 
                    v["x"], v["x"] + v["width"],
                    v["y"], v["y"] + v["height"],
                    v["z"], v["z"] + v["depth"])
                tif_path = os.path.join(os.path.dirname(tif_path), tif_file)
                lpd["blocks"].append((tif_path, v))
            json.dump(lpd, open(lp_path, "w"))
            self.output_channel_loading_plans[volume] = lp_path
        
    def make_merge_tasks(self):
        '''Make one merge task per block'''
        tasks = []
        for volume, sp in self.output_channel_storage_plans.items():
            lps = [self.input_channel_block_lps[channel][volume]
                   for channel in self.input_dataset_names]
            task = MergePredictionsTask(
               storage_plan=sp,
               loading_plans=lps,
               operation=self.operation,
               invert=self.invert)
            tasks.append(task)
        return tasks
    
    def run(self):
        '''Make an index file with the details of the run'''
        d = dict(output_loading_plans=[], output_storage_plans=[])
        lists = []
        for channel in self.input_dataset_names:
            cd = d[channel] = {}
            for name in ("input_channel_loading_plans", 
                         "input_channel_block_loading_plans",
                         "input_channel_storage_plans"):
                d1 = cd[name] = []
                d2 = getattr(self, name)[channel]
                lists.append((d1, d2))
        lists.append(d["output_channel_loading_plans"], 
                     self.output_channel_loading_plans)
        lists.append(d["output_channel_storage_plans"],
                     self.output_channel_storage_plans)
        for l1, d2 in lists:
            for v, path in d2.items():
                l1.append((to_json_serializable(v), path))
        with self.output().open("w") as fd:
            json.dump(d, fd)
            
        
            
        
                