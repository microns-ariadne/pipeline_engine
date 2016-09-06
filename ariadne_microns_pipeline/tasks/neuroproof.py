'''Run Neuroproof on a probability map and segmentation

Neuroproof should be precompiled - one of the parameters to the task is
the executable's location.
'''

from cv2 import imread, imwrite
import luigi
import numpy as np
import os
import subprocess
import tempfile

from ..parameters import VolumeParameter, DatasetLocationParameter
from ..targets.factory import TargetFactory
from utilities import RequiresMixin, RunMixin, CILKCPUMixin

class NeuroproofTaskMixin:
    
    volume = VolumeParameter(
        description="The extents of the volume being neuroproofed")
    prob_location = DatasetLocationParameter(
        description="Location of the membrane probability dataset. "
        "Note: the probabilities can't be sharded.")
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
        prob_volume, seg_volume = list(self.input())
        #
        # This neuroproof takes color segmentation files as a stack of .PNG
        # files in a single directory. The .png files are in color with the
        # uppermost 8 bits of a 24-bit integer in the blue channel,
        # the middle 8 bits in the green channel and the lower 8 bits
        # in the red channel.
        #
        # For utmost safety, we read it into core and write it back out to
        # a temporary directory.
        #
        prob_ds = prob_volume.imread()
        seg_ds = seg_volume.imread()
        prob_tempdir = tempfile.mkdtemp()
        input_seg_tempdir = tempfile.mkdtemp()
        output_seg_tempdir = tempfile.mkdtemp()
        output_seg_file = os.path.join(output_seg_tempdir, "neuroproof")
        try:
            for z in range(prob_ds.shape[0]):
                path = os.path.join(prob_tempdir, "%04d.png" % z)
                plane = prob_ds[z]
                imwrite(path, plane)
            for z in range(seg_ds.shape[0]):
                path = os.path.join(input_seg_tempdir, "%04d.png" % z)
                plane = np.dstack((seg_ds[z] >> 16, 
                                   (seg_ds[z] >> 8) & 0xff,
                                   seg_ds[z] & 0xff)).astype(np.uint8)
                imwrite(path, plane)
            
            args = [self.neuroproof,
                    "--output-file=%s" % output_seg_file,
                    input_seg_tempdir,
                    prob_tempdir,
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
            subprocess.check_call(args, env=env)
            #
            # Read the result and write into the volume
            #
            output_volume = self.output()
            np_files = sorted(filter(
                lambda _:_.startswith("neuroproof") and _.endswith(".png"),
                os.listdir(output_seg_tempdir)))
            planes = []
            for z, filename in enumerate(np_files):
                plane = imread(os.path.join(output_seg_tempdir, filename), 1)
                plane = (
                    plane[:, :, 0].astype(np.uint32) +
                    (plane[:, :, 1].astype(np.uint32) << 8) +
                    (plane[:, :, 2].astype(np.uint32) << 16))
                planes.append(plane)
            output_volume.imwrite(np.array(planes))
        finally:
            for path in prob_tempdir, input_seg_tempdir, output_seg_tempdir:
                for filename in os.listdir(path):
                    os.remove(os.path.join(path, filename))
                os.rmdir(path)


class NeuroproofTask(NeuroproofTaskMixin, NeuroproofRunMixin,
                     RequiresMixin, RunMixin, CILKCPUMixin, luigi.Task):
    task_namespace = "ariadne_microns_pipeline"
