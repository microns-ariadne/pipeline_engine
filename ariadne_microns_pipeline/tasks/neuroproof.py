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
import rh_logger

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
    wants_standard_neuroproof = luigi.BoolParameter(
        description = "Use the standard interface to Neuroproof")
    
    def ariadne_run(self):
        '''Run the neuroproof subprocess'''
        if self.wants_standard_neuroproof:
            self.run_standard()
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
        fd, h5file = tempfile.mkstemp(".h5")
        rh_logger.logger.report_event("Neuroproof input: %s" % h5file)
        with h5py.File(h5file, "w") as h:
            h.create_dataset("segmentation", data=seg_volume.imread())
            n_channels = len(additional_maps) + 2
            probs = h.create_dataset(
                "probabilities",
                shape=(prob_volume.depth, prob_volume.height, 
                       prob_volume.width, n_channels),
                dtype=np.float32)
            membrane = prob_volume.imread().astype(np.float32) / 255
            probs[:, :, :, 0] = membrane
            probs[:, :, :, 1] = 1 - membrane
            for idx, tgt in enumerate(additional_maps):
                probs[:, :, :, idx+2] = tgt.imread().astype(np.float32) / 255
        os.close(fd)
        outfile = tempfile.mktemp(".h5")
        rh_logger.logger.report_event("Neuroproof output: %s" % outfile)
        
        try:
            args = [self.neuroproof,
                    "-threshold", str(self.threshold),
                    "-algorithm", "1",
                    "-nomito",
                    "-min_region_sz", "0",
                    "-watershed", h5file, "segmentation",
                    "-prediction", h5file, "probabilities",
                    "-output", outfile, "segmentation",
                    "-classifier", self.classifier_filename]
            rh_logger_logger.report_event(" ".join(args)
            
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
            os.remove(outfile)
        
    def run_optimized(self):
        '''Run the MIT neuroproof'''
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
        # Get the anticipated filenames and make sure the directories
        # for them have been created.
        #
        output_target = self.output()
        output = output_target.anticipate_filenames()
        outdirs = set()
        for filename in output:
            outdirs.add(os.path.dirname(filename))
        for directoryname in outdirs:
            if not os.path.isdir(directoryname):
                os.makedirs(directoryname)
        #
        # neuroproof_graph_predict will take a .json file in place of a
        # prediction file. It has the following format:
        #
        # { "probabilities": [
        #      [ filenames of channel 0],
        #      [ filenames of channel 1],
        #      ...
        #      [ filenames of channel N]]
        #   "config": {
        #        "invert": [ True or False per probability ]
        #   }
        #   "watershed": [ filenames of watershed ],
        #   "output": [ filenames to write on output] }
        #
        # config is optional as are its key/value pairs. Predictably,
        # "invert" is False by default.
        #
        probabilities = [tgt.get_filenames() for tgt in
                         [prob_volume] + additional_maps]
        watershed = seg_volume.get_filenames()
        config_path = \
            os.path.splitext(self.classifier_filename)[0] + "_config.json"
        if os.path.isfile(config_path):
            config = json.load(open(config_path, "r"))
        else:
            config = {}
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
