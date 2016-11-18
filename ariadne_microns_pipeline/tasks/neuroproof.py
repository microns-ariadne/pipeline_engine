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

from ..parameters import VolumeParameter, DatasetLocationParameter
from ..parameters import MultiDatasetLocationParameter
from ..targets.factory import TargetFactory
from utilities import RequiresMixin, RunMixin, CILKCPUMixin

class NeuroproofTaskMixin:
    
    volume = VolumeParameter(
        description="The extents of the volume being neuroproofed")
    prob_location = DatasetLocationParameter(
        description="Location of the membrane probability dataset. "
        "Note: the probabilities can't be sharded.")
    additional_locations = MultiDatasetLocationParameter(
        default=[],
        description="Locations of additional probability maps "
        "to aid Neuroproof")
    input_seg_location = DatasetLocationParameter(
        description="Location of the input segmentation dataset. "
        "Note: the segmentation can't be sharded.")
    output_seg_location = DatasetLocationParameter(
        description="Location of the output segmentation dataset")
    
    def input(self):
        '''Yield the probability volume target and segmentation volume target'''
        tf = TargetFactory()
        yield tf.get_volume_target(self.prob_location, self.volume)
        yield tf.get_volume_target(self.input_seg_location, 
                                   self.volume)
        for location in self.additional_locations:
            yield tf.get_volume_target(location, self.volume)
    
    def output(self):
        '''Return the output segmentation'''
        tf = TargetFactory()
        return tf.get_volume_target(self.output_seg_location, self.volume)

class NeuroproofRunMixin:
    neuroproof = luigi.Parameter(
        description="Location of the neuroproof_graph_predict binary")
    neuroproof_ld_library_path = luigi.Parameter(
        description="Library paths to Neuroproof's shared libraries. "
        "This should include paths to CILK stdc++ libraries, Vigra libraries, "
        "JSONCPP libraries, and OpenCV libraries.")
    classifier_filename = luigi.Parameter(
        description="The Vigra random forest classifier or OpenCV random "
        "forest agglomeration classifier.")
    
    def ariadne_run(self):
        '''Run the neuroproof subprocess'''
        #
        # The arguments for neuroproof_graph_predict:
        #
        # watershed-file: directory containing the segmentation .png files
        # prediction-file: directory containing the probability .png files
        # classifier-file: path to either the .xml or .h5 agglomeration
        #                  classifier
        #
        inputs = self.input()
        prob_volume = inputs.next()
        seg_volume = inputs.next()
        additional_maps = list(inputs)
        #
        # neuroproof_graph_predict will take a .json file in place of a
        # prediction file. It has the following format:
        #
        # { "probabilities": [
        #      [ filenames of channel 0],
        #      [ filenames of channel 1],
        #      ...
        #      [ filenames of channel N]]
        #   "watershed": [ filenames of watershed ],
        #   "output": [ filenames to write on output] }
        #
        probabilities = [tgt.get_filenames() for tgt in
                         [prob_volume] + additional_maps]
        watershed = seg_volume.get_filenames()
        output_target = self.output()
        output = output_target.anticipate_filenames()
        d = dict(probabilities=probabilities,
                 watershed=watershed,
                 output=output)
        fd, json_path = tempfile.mkstemp(".json")
        f = os.fdopen(fd, "w")
        json.dump(d, f)
        f.close()
        try:
            args = [self.neuroproof,
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
