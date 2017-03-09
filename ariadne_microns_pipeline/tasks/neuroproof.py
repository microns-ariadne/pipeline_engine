'''Run Neuroproof on a probability map and segmentation

Neuroproof should be precompiled - one of the parameters to the task is
the executable's location.
'''

from cv2 import imread, imwrite
import h5py
import json
import luigi
import numpy as np
import os
import subprocess
import tempfile

from .utilities import RequiresMixin, RunMixin, CILKCPUMixin, DatasetMixin
from ..targets import DestVolumeReader

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
                         ] + self.additional_locations
        for loading_plan in loading_plans:
            for tgt in DestVolumeReader(loading_plan):
                yield tgt
                
    def estimate_memory_usage(self):
        '''Return an estimate of bytes of memory required by this task'''
        v1 = np.prod([2048, 2048, 100])
        m1 = (3685308 + 152132) * 1000
        v2 = np.prod([1436, 1436, 65])
        m2 = (1348048 + 152100) * 1000
        #
        # Model is Ax + B + ALx where x is the output volume and AL is the
        # number of additional locations.
        #
        B = (v1 * m2 - v2 * m1) / (v1 - v2)
        A = (float(m1) - B) / v1
        v = np.prod([self.volume.width, 
                     self.volume.height, 
                     self.volume.depth])
        AL = len(self.additional_locations)
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
    watershed_threshold = luigi.FloatParameter(
        default=0,
        description="Threshold used for removing small bodies as a "
                    "post-processing step")
    
    def ariadne_run(self):
        '''Run the neuroproof subprocess'''
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
            [self.prob_loading_plan_path] + self.additional_loading_plan_paths
        watershed = self.input_seg_loading_plan_path
        config_path = \
            os.path.splitext(self.classifier_filename)[0] + "_config.json"
        if os.path.isfile(config_path):
            config = json.load(open(config_path, "r"))
        else:
            config = {}
        config["use-loading-plan"] = True
        config["use-storage-plan"] = True
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
            output_target.finish_imwrite(np.dtype(np.uint32))
        finally:
            os.remove(json_path)

class NeuroproofTask(NeuroproofTaskMixin, NeuroproofRunMixin,
                     RequiresMixin, RunMixin, CILKCPUMixin, luigi.Task):
    task_namespace = "ariadne_microns_pipeline"
