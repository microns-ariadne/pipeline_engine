import h5py
import json
import luigi
import numpy as np
from scipy.ndimage import zoom
import os
import rh_logger

from .synapse_gt_pipeline import SynapseGtTask
from ..parameters import Volume, DatasetLocation, VolumeParameter, DatasetLocationParameter
from ..targets.png_volume_target import PngVolumeTarget
from ..targets.factory import TargetFactory
from ..tasks.factory import AMTaskFactory
from ..tasks.match_synapses import MatchMethod
from ..tasks.utilities import RequiresMixin

PATTERN = "{x:09d}_{y:09d}_{z:09d}"

class SynapseCutoutPipelineTask(luigi.Task):
    
    root = luigi.Parameter(default="/n/coxfs01/leek/results/red_cylinder_pipeline/kasthuri11/neocortex/sem/raw")
    x_start = luigi.IntParameter(default=8192)
    x_end = luigi.IntParameter(default=8192+2048*3)
    x_step = luigi.IntParameter(default=2048)
    y_start = luigi.IntParameter(default=15616)
    y_end = luigi.IntParameter(default=15616 + 2048*2)
    y_step = luigi.IntParameter(default=2048)
    z_start = luigi.IntParameter(default=1017)
    z_end = luigi.IntParameter(default=1283)
    z_step = luigi.ListParameter(
        default=[88, 89, 89])
    
    def get_input_volume(self, x, y, z, dataset_name):
        pattern = PATTERN+"_"+dataset_name
        done_file = os.path.join(self.root, str(x), str(y), str(z),
                                 pattern.format(locals())+".done")
        tgt = PngVolumeTarget.from_done_file(done_file, pattern)
        return tgt
    
    def get_output_volume(self, x, y, z, width, height, depth, dataset_name):
        pattern = "{x:09d}_{y:09d}_{z:09d}_"+dataset_name
        tgt = PngVolumeTarget(
            [os.path.join(self.root, str(x), str(y), str(z))],
            dataset_name,
            pattern, x, y, z, width, height, depth)
        return tgt
    
    def input(self):
        yield self.get_input_volume(self.x_start, self.y_start, self.z_start, 
                                    "image")
    
    def output(self):
        return luigi.LocalTarget(os.path.join(self.root, "cutout.done"))
        
    def requires(self):
        if hasattr(self, "requirements"):
            for requirement in self.requirements:
                yield requirement
            return
        try:
            rh_logger.logger.start_process("Synapse cutouts", "starting", [])
        except:
            pass
        nx = int((self.x_end - self.x_start + self.x_step - 1) / self.x_step)
        ny = int((self.y_end - self.y_start + self.y_step - 1) / self.y_step)
        nz = len(self.z_step)
        self.requirements = []
        
        for x in range(self.x_start, self.x_end, self.x_step):
            for y in range(self.y_start, self.y_end, self.y_step):
                z = self.z_start
                for depth in self.z_step:
                    depth = int(depth)
                    requirement = self.do_block(x, y, z, depth)
                    self.requirements.append(requirement)
                    yield requirement
                    z += depth
    
    def do_block(self, x, y, z, depth):
        rh_logger.logger.report_event("Processing block %d, %d, %d" % (x, y, z))
        root_in = os.path.join(self.root, str(x), str(y), str(z))
        path = os.path.join(self.root, str(x), str(y), str(z),
                            "synapse-Unet-caffe",
                            "cylinder-syn-pred.h5")
        with h5py.File(path, "r") as h5fd:
            ds = h5fd["main"]
        
            wout = ds.shape[3] * 2
            x_pad = (self.x_step - wout) / 2
            hout = ds.shape[2] * 2
            y_pad = (self.y_step - hout) / 2
            dout = ds.shape[1]
            z_pad = (depth - dout) / 2
            
            xout = x+x_pad
            yout = y+y_pad
            zout = z+z_pad
            root_out = os.path.join(self.root, str(xout), str(yout), str(zout))
            tgt = self.get_output_volume(xout, yout, zout, wout, hout, dout, 
                                         "synapse")
            if not tgt.exists():
                synapse = (ds[0] * 255).astype(np.uint8)
                rh_logger.logger.report_event("    Loaded synapse probs")
                synapse = np.array([zoom(_, 2) for _ in synapse])
                rh_logger.logger.report_event("    Resized prob volume")
                tgt.imwrite(synapse)
                rh_logger.logger.report_event("Wrote synapse probabilities")
        
        factory = AMTaskFactory()
        input_volume = Volume(x, y, z, self.x_step, self.y_step, depth)
        output_volume = Volume(xout, yout, zout, wout, hout, dout)
        b_neuroproof, b_gt_mask, b_gt, b_synapse_gt_segmentation =\
            [factory.gen_block_task(
                output_volume, 
                DatasetLocation([root_out], dataset_name, PATTERN+"_"+dataset_name),
                [dict(volume=input_volume,
                      location=DatasetLocation([root_in], dataset_name, PATTERN+"_"+dataset_name))])
             for dataset_name in "neuroproof", "gt-mask", "gt", "synapse-gt-segmentation"]
        synapse_task = factory.gen_find_synapses_task(
            volume=output_volume,
            syn_location=tgt.dataset_location,
            neuron_segmentation=b_neuroproof.output().dataset_location,
            output_location=DatasetLocation([root_out], "synapse-segmentation", PATTERN+"_synapse-segmentation"),
            threshold=256/4,
            erosion_xy=4,
            erosion_z=1,
            sigma_xy=8,
            sigma_z=.5,
            min_size_2d=100,
            max_size_2d=40000,
            min_size_3d=2000,
            min_slice=3)
        synapse_task.set_requirement(b_neuroproof)
        conn_task = factory.gen_connect_synapses_task(
            volume=output_volume,
            synapse_location=synapse_task.output().dataset_location,
            neuron_location=b_neuroproof.output().dataset_location,
            output_location=os.path.join(root_out, "synapse_connectivity.json"),
            xy_dilation=3,
            z_dilation=0,
            min_contact=500)
        conn_task.set_requirement(synapse_task)
        conn_task.set_requirement(b_neuroproof)
        match_task = factory.gen_match_synapses_task(
            volume=output_volume,
            gt_location=b_synapse_gt_segmentation.output().dataset_location, 
            detected_location = synapse_task.output().dataset_location, 
            output_location=os.path.join(root_out, "synapse-matches.json"),
            method=MatchMethod.overlap)
        match_task.min_overlap_pct = 10.0
        match_task.mask_location = b_gt_mask.output().dataset_location
        match_task.set_requirement(synapse_task)
        match_task.set_requirement(b_synapse_gt_segmentation)
        match_task.set_requirement(b_gt_mask)
        neuron_match_task=factory.gen_match_neurons_task(
            volume=output_volume,
            gt_location=b_gt.output().dataset_location,
            detected_location=b_neuroproof.output().dataset_location,
            output_location=os.path.join(root_out, "neuron-matches.json"))
        neuron_match_task.set_requirement(b_gt)
        neuron_match_task.set_requirement(b_neuroproof)
        gt_neuron_synapse_match_task = factory.gen_connected_components_task(
            volume1=output_volume,
            location1=b_gt.output().dataset_location,
            volume2=output_volume,
            location2=b_synapse_gt_segmentation.output().dataset_location,
            overlap_volume=output_volume,
            output_location=os.path.join(root_out, "gt-neuron-synapse-matches.json"))
        gt_neuron_synapse_match_task.min_overlap_percent = 0.0
        gt_neuron_synapse_match_task = factory.gen_connect_synapses_task(
            volume=output_volume,
            neuron_location=b_gt.output().dataset_location,
            synapse_location=b_synapse_gt_segmentation.output().dataset_location,
            output_location=os.path.join(root_out, "gt-neuron-synapse-matches.json"),
            xy_dilation=6,
            z_dilation=0,
            min_contact=0)
        gt_neuron_synapse_match_task.set_requirement(b_gt)
        gt_neuron_synapse_match_task.set_requirement(b_synapse_gt_segmentation)
        acc_task = FakeAllConnectedComponentsTask(
            volume=output_volume,
            input_location=b_neuroproof.output().dataset_location,
            output_location=os.path.join(root_out, "all-connected-components.json"))
        acc_task.set_requirement(b_neuroproof)
        stats_task = factory.gen_synapse_statistics_task(
            synapse_matches=[match_task.output().path],
            detected_synapse_connections=[conn_task.output().path],
            neuron_map=acc_task.output().path,
            gt_neuron_maps=[neuron_match_task.output().path],
            gt_synapse_connections=[gt_neuron_synapse_match_task.output().path],
            output_location=os.path.join(root_out, "synapse-stats.json"))
        stats_task.set_requirement(gt_neuron_synapse_match_task)
        stats_task.set_requirement(match_task)
        stats_task.set_requirement(conn_task)
        stats_task.set_requirement(acc_task)
        stats_task.set_requirement(neuron_match_task)
        return stats_task
    
    def run(self):
        tot_tp = 0
        tot_fp = 0
        tot_fn = 0
        tot_tp_synapses = 0
        tot_fp_synapses = 0
        tot_fn_synapses = 0
        for task in self.requires():
            with task.output().open("r") as fd:
                d = json.load(fd)
                tot_tp += d["n_true_positives"]
                tot_fp += d["n_false_positives"]
                tot_fn += d["n_false_negatives"]
                tot_tp_synapses += d["n_true_positive_synapses"]
                tot_fp_synapses += d["n_false_positive_synapses"]
                tot_fn_synapses += d["n_false_negative_synapses"]
                
        with self.output().open("w") as fd:
            json.dump(dict(n_true_positives=tot_tp,
                           n_false_positives=tot_fp,
                           n_false_negatives=tot_fn,
                           n_true_positive_synapses=tot_tp_synapses,
                           n_false_positive_synapses=tot_fp_synapses,
                           n_false_negative_synapses=tot_fn_synapses,
                           precision=float(tot_tp)/(tot_tp+tot_fp),
                           recall=float(tot_tp)/(tot_tp+tot_fn),
                           synapse_precision=float(tot_tp_synapses) / (tot_tp_synapses + tot_fp_synapses),
                           synapse_recall=float(tot_tp_synapses) / (tot_tp_synapses + tot_fn_synapses)),
                      fd)

class FakeAllConnectedComponentsTask(RequiresMixin,
                                     luigi.Task):
    
    volume = VolumeParameter()
    input_location = DatasetLocationParameter()
    output_location = luigi.Parameter()
    
    def input(self):
        yield TargetFactory().get_volume_target(
            volume=self.volume,
            location=self.input_location)
    
    def output(self):
        return luigi.LocalTarget(self.output_location)
    
    def run(self):
        seg = self.input().next().imread()
        areas = np.bincount(seg.flatten())
        areas[0] = 0
        l = np.where(areas > 0)[0]
        with self.output().open("w") as fd:
            json.dump(dict(count=0,
                           volumes=[(self.volume.to_dictionary(),
                                     [(_, _) for _ in l])]), fd)
