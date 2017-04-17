'''Run Neuroproof on a probability map and segmentation

Neuroproof should be precompiled - one of the parameters to the task is
the executable's location.
'''

from cv2 import imread, imwrite
import enum
import h5py
import json
import luigi
import multiprocessing
import numpy as np
import os
import subprocess
import tempfile
import rh_logger

from .utilities import RequiresMixin, RunMixin, CILKCPUMixin, DatasetMixin
from ..targets import DestVolumeReader
from .neuroproof_common import NeuroproofVersion
from .neuroproof_common import write_prob_volume, write_seg_volume

class NeuroproofTaskMixin(DatasetMixin):
    
    prob_loading_plan_path = luigi.Parameter(
        description="Location of the membrane probability dataset. "
        "Note: the probabilities can't be sharded.")
    additional_loading_plan_paths = luigi.Parameter(
        default=[],
        description="Locations of additional probability maps "
        "to aid Neuroproof")
    input_seg_loading_plan_path = luigi.Parameter(
        description="Location of the input segmentation dataset.")
    
    def input(self):
        '''Yield the probability volume target and segmentation volume target'''
        loading_plans = [self.input_seg_loading_plan_path,
                         self.prob_loading_plan_path
                         ] + list(self.additional_loading_plan_paths)
        for loading_plan in loading_plans:
            for tgt in DestVolumeReader(loading_plan).get_source_targets():
                yield tgt
                
    def estimate_memory_usage(self):
        '''Return an estimate of bytes of memory required by this task'''
        v1 = np.prod([2048, 2048, 100])
        m1 = (3685308 + 152132) * 1000
        v2 = np.prod([1436, 1436, 65])
        m2 = (1348048 + 152100) * 1000
        volume = self.output().volume
        #
        # Model is Ax + B + ALx where x is the output volume and AL is the
        # number of additional locations.
        #
        B = (v1 * m2 - v2 * m1) / (v1 - v2)
        A = (float(m1) - B) / v1
        v = np.prod([volume.width, 
                     volume.height, 
                     volume.depth])
        AL = len(self.additional_loading_plan_paths)
        return int(A * v + AL * v + B)


class NeuroproofRunMixin:
    neuroproof = luigi.Parameter(
        description="Location of the neuroproof_graph_predict binary")
    neuroproof_ld_library_path = luigi.Parameter(
        description="Library paths to Neuroproof's shared libraries. "
        "This should include paths to CILK stdc++ libraries, Vigra libraries, "
        "JSONCPP libraries, and OpenCV libraries.")
    classifier_filename = luigi.Parameter(
        description="The Vigra random forest classifier or OpenCV random "
        "forest agglomeration classifier. In addition, there may be a file "
        "with the given filename with \"_ignore.txt\" appended which gives "
        "the indices of the features to ignore and similarly a file with "
        "\"_config.json\" appended which gives configuration information to "
        "neuroproof.")
    threshold = luigi.FloatParameter(
        default=0.2,
        description="Segmentation threshold for neuroproof")
    watershed_threshold = luigi.IntParameter(
        default=0,
        description="Threshold used for removing small bodies as a "
                    "post-processing step")
    neuroproof_version = luigi.EnumParameter(
        enum=NeuroproofVersion,
        default=NeuroproofVersion.MIT,
        description="The command-line convention to be used to run the "
        "Neuroproof binary")
    
    def ariadne_run(self):
        '''Run the neuroproof subprocess'''
        if self.neuroproof_version == NeuroproofVersion.MINIMAL:
            self.run_standard()
        elif self.neuroproof_version == NeuroproofVersion.FLY_EM:
            self.run_optimized_with_copy()
        elif self.neuroproof_version == NeuroproofVersion.FAST:
            self.run_fast()
        else:
            self.run_optimized()

    def run_standard(self):
        '''Run the out-of-the-box neuroproof'''
        #
        # Write the segmentation and membrane probabilities to one
        # big temporary hdf5 file
        #
        inputs = self.input()
        prob_volume = inputs.next()
        seg_volume = inputs.next()
        additional_maps = list(inputs)
        h5file = tempfile.mktemp(".h5")
        probfile = tempfile.mktemp(".h5")
        rh_logger.logger.report_event("Neuroproof watershed: %s" % h5file)
        rh_logger.logger.report_event("Neuroproof probabilities: %s" % probfile)
        pool = multiprocessing.Pool(2)
        seg_result = pool.apply_async(
            write_seg_volume,
            args=(h5file, seg_volume, "segmentation"))
        prob_result = pool.apply_async(
            write_prob_volume,
            args=(prob_volume, additional_maps, probfile, "probabilities", 
                  False))
        pool.close()
        pool.join()
        seg_result.get()
        prob_result.get()
        outfile = tempfile.mktemp(".h5")
        rh_logger.logger.report_event("Neuroproof output: %s" % outfile)
        
        try:
            args = [self.neuroproof,
                    "-threshold", str(self.threshold),
                    "-algorithm", "1",
                    "-nomito",
                    "-min_region_sz", "0",
                    "-watershed", h5file, "segmentation",
                    "-prediction", probfile, "probabilities",
                    "-output", outfile, "segmentation",
                    "-classifier", self.classifier_filename]
            rh_logger.logger.report_event(" ".join(args))
            
            #
            # Inject the custom LD_LIBRARY_PATH into the subprocess environment
            #
            env = os.environ.copy()
            if "LD_LIBRARY_PATH" in env:
                ld_library_path = self.neuroproof_ld_library_path + os.pathsep +\
                    env["LD_LIBRARY_PATH"]
            else:
                ld_library_path = self.neuroproof_ld_library_path
            env["LD_LIBRARY_PATH"] = ld_library_path
            self.configure_env(env)
            #
            # Do the dirty deed...
            #
            subprocess.check_call(args, env=env, close_fds=True)
            #
            # There's an apparent bug in Neuroproof where it writes
            # the output to "fo.h5" for example, when you've asked it
            # to send the output to "foo.h5"
            #
            alt_outfile = os.path.splitext(outfile)[0][:-1] + ".h5"
            if (not os.path.exists(outfile)) and os.path.exists(alt_outfile):
                outfile=alt_outfile
            #
            # Finish the output volume
            #
            with h5py.File(outfile, "r") as fd:
                self.output().imwrite(
                    fd["segmentation"][:].astype(np.uint32))
        finally:
            os.remove(h5file)
            os.remove(probfile)
            if os.path.exists(outfile):
                os.remove(outfile)
    
    def run_optimized_with_copy(self):
        '''Run the MIT neuroproof, but copying everything'''
        inputs = self.input()
        prob_volume = inputs.next()
        seg_volume = inputs.next()
        additional_maps = list(inputs)
        h5file = tempfile.mktemp(".h5")
        probfile = tempfile.mktemp(".h5")
        rh_logger.logger.report_event("Neuroproof watershed: %s" % h5file)
        rh_logger.logger.report_event("Neuroproof probabilities: %s" % probfile)
        pool = multiprocessing.Pool(2)
        seg_result = pool.apply_async(
            write_seg_volume,
            args=(h5file, seg_volume, "stack"))
        prob_result = pool.apply_async(
            write_prob_volume,
            args=(prob_volume, additional_maps, probfile, "volume/predictions"))
        pool.close()
        pool.join()
        seg_result.get()
        prob_result.get()
        outfile = tempfile.mktemp(".h5")
        rh_logger.logger.report_event("Neuroproof output: %s" % outfile)
        try:
            args = [self.neuroproof,
                    h5file,
                    probfile,
                    self.classifier_filename,
                    "--output-file", outfile,
                    "--threshold", str(self.threshold),
                    "--watershed-threshold", str(self.watershed_threshold)]
            rh_logger.logger.report_event(" ".join(args))
            
            #
            # Inject the custom LD_LIBRARY_PATH into the subprocess environment
            #
            env = os.environ.copy()
            if "LD_LIBRARY_PATH" in env:
                ld_library_path = self.neuroproof_ld_library_path + os.pathsep +\
                    env["LD_LIBRARY_PATH"]
            else:
                ld_library_path = self.neuroproof_ld_library_path
            env["LD_LIBRARY_PATH"] = ld_library_path
            self.configure_env(env)
            #
            # Do the dirty deed...
            #
            subprocess.check_call(args, env=env, close_fds=True)
            #
            # There's an apparent bug in Neuroproof where it writes
            # the output to "fo.h5" for example, when you've asked it
            # to send the output to "foo.h5"
            #
            alt_outfile = os.path.splitext(outfile)[0][:-1] + ".h5"
            if (not os.path.exists(outfile)) and os.path.exists(alt_outfile):
                outfile=alt_outfile
            #
            # Finish the output volume
            #
            with h5py.File(outfile, "r") as fd:
                self.output().imwrite(
                    fd["stack"][:].astype(np.uint32))
        finally:
            os.remove(h5file)
            os.remove(probfile)
            if os.path.exists(outfile):
                os.remove(outfile)
        
    def run_optimized(self):
        '''Run the MIT neuroproof'''
        #
        # The arguments for neuroproof_graph_predict:
        #
        output_target = self.output()
        output_target.create_directories()
        output = self.storage_plan
        #
        # neuroproof_graph_predict will take a .json file in place of a
        # prediction file. It has the following format:
        #
        # { "probabilities": [
        #       "<probability-loading-plan-1>",
        #       ...
        #       "<probability-loading-plan-N"
        #   ]
        #   "config": {
        #        "invert": [ True or False per probability ],
        #        "use-loading-plan": True,
        #        "use-storage-plan": True
        #   }
        #   "watershed": "watershed-loading-plan",
        #   "output": "output-storage-plan" }
        #
        # config is optional as are its key/value pairs. Predictably,
        # "invert" is False by default.
        #
        probabilities = \
            [self.prob_loading_plan_path] + \
            list(self.additional_loading_plan_paths)
        watershed = self.input_seg_loading_plan_path
        config_path = \
            os.path.splitext(self.classifier_filename)[0] + "_config.json"
        if os.path.isfile(config_path):
            config = json.load(open(config_path, "r"))
        else:
            config = {}
        config["use-loading-plans"] = True
        config["use-storage-plans"] = True
        d = dict(config=config,
                 probabilities=probabilities,
                 watershed=watershed,
                 output=output)
        fd, json_path = tempfile.mkstemp(".json")
        f = os.fdopen(fd, "w")
        json.dump(d, f)
        f.close()
        try:
            args = [self.neuroproof,
                    "--threshold", str(self.threshold),
                    "--watershed-threshold", str(self.watershed_threshold),
                    json_path,
                    json_path,
                    self.classifier_filename]
            #
            # Inject the custom LD_LIBRARY_PATH into the subprocess environment
            #
            env = os.environ.copy()
            if "LD_LIBRARY_PATH" in env:
                ld_library_path = self.neuroproof_ld_library_path + os.pathsep +\
                    env["LD_LIBRARY_PATH"]
            else:
                ld_library_path = self.neuroproof_ld_library_path
            env["LD_LIBRARY_PATH"] = ld_library_path
            self.configure_env(env)
            #
            # Do the dirty deed...
            #
            subprocess.check_call(args, env=env, close_fds=True)
            #
            # Finish the output volume
            #
            # We collect some summary statistics here that are added to
            # the JSON file.
            #
            data = output_target.imread()
            d = json.load(open(output_target.storage_plan_path))
            areas = np.bincount(data.ravel())
            areas[0] = 0
            labels = np.where(areas > 0)[0]
            areas = areas[labels]
            d["areas"] = areas.tolist()
            d["labels"] = labels.tolist()
            with output_target.open("w") as fd:
                json.dump(d, fd)
        finally:
            os.remove(json_path)
    
    def run_fast(self):
        '''Run using Tim Kaler's speedup + NeuroProof_plan'''
        #
        # Make the target directories for the .tif files
        #
        output_target = self.output()
        output_target.create_directories()
        
        arguments = [self.neuroproof,
                     "-watershed", self.input_seg_loading_plan_path,
                     "-prediction", self.prob_loading_plan_path,
                     "-classifier", self.classifier_filename,
                     "-output", self.storage_plan,
                     "-threshold", str(self.threshold),
                     "-algorithm", "1",
                     "-nomito",
                     "-min_region_sz", str(self.watershed_threshold)]
        #
        # Inject the custom LD_LIBRARY_PATH into the subprocess environment
        #
        env = os.environ.copy()
        if "LD_LIBRARY_PATH" in env:
            ld_library_path = self.neuroproof_ld_library_path + os.pathsep +\
                env["LD_LIBRARY_PATH"]
        else:
            ld_library_path = self.neuroproof_ld_library_path
        env["LD_LIBRARY_PATH"] = ld_library_path
        self.configure_env(env)
        #
        # Do the dirty deed...
        #
        subprocess.check_call(arguments, env=env)
        #
        # Finish the output volume
        #
        # We collect some summary statistics here that are added to
        # the JSON file.
        #
        data = output_target.imread()
        d = json.load(open(output_target.storage_plan_path))
        areas = np.bincount(data.ravel())
        areas[0] = 0
        labels = np.where(areas > 0)[0]
        areas = areas[labels]
        d["areas"] = areas.tolist()
        d["labels"] = labels.tolist()
        with output_target.open("w") as fd:
            json.dump(d, fd)
        

class NeuroproofTask(NeuroproofTaskMixin, NeuroproofRunMixin,
                     RequiresMixin, RunMixin, CILKCPUMixin, luigi.Task):
    task_namespace = "ariadne_microns_pipeline"
