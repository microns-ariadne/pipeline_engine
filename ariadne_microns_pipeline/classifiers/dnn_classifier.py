'''At present, this wraps the fc_dnn classifier for AC3

The classifier has the path to the kernel matrices hardcoded at
./K11_S1_AC3_256_cc_3D_PAD

Parameters to the executable are 
Input type: 2 to read the files in a directory
X/Y padding: 26
Z depth of classifier:
# z planes
width
height
# matrices: 1
# classes: 2
'''

from cv2 import imread, imwrite
import luigi
import multiprocessing
import numpy as np
import os
import rh_config
import subprocess
import tempfile
from ..targets.classifier_target import AbstractPixelClassifier

class DNNClassifier(AbstractPixelClassifier):
    
    config = rh_config.config["fc_dnn"]
    
    def get_x_pad(self):
        return int(self.config["xy_pad"])
    
    def get_y_pad(self):
        return int(self.config["xy_pad"])
    
    def get_z_pad(self):
        return (int(self.config["z_depth"]) - 1)/2
    
    def get_class_names(self):
        return ["membrane"]
    
    def get_resources(self):
        '''Request CPU resources
        
        By default, ask for all CPU resources or, if "cpu_count" is
        configured in the fc_dnn section of .rh-config.yaml, use that
        instead.
        '''
        max_cpu_count = luigi.configuration.get_config().getint(
            "resources", "cpu_count", default=sys.maxint)
        config_cpu_count = int(self.config.get(
            "cpu_count", multiprocessing.cpu_count()))
        cpu_count = min(max_cpu_count, config_cpu_count)
        return dict(cpu_count=cpu_count)
    
    def classify(self, image, x, y, z):
        dir_in = tempfile.mkdtemp()
        dir_out = tempfile.mkdtemp()
        try:
            #
            # Write image to .png
            #
            for z in range(image.shape[0]):
                path = os.path.join(dir_in, "%04d.png" % z)
                imwrite(path, image[z])
            exec_dir = self.config["path"]
            args = ["src/run_dnn",          # The program's name
                    "2",                    # Load images in a directory
                    str(self.config["xy_pad"]), # padding needed by kernel
                    str(self.config["z_depth"]), # # planes needed by kernel
                    str(image.shape[0]),    # # planes in image
                    str(image.shape[2]),    # width
                    str(image.shape[1]),    # height
                    "1",                    # # planes per image
                    "2",                    # # of classes in classifier
                    dir_in, dir_out]
            env = os.environ.copy()
            if "ld_library_path" in self.config:
                env["LD_LIBRARY_PATH"] =\
                    os.pathsep.join(self.config["ld_library_path"])
            env["CILK_NWORKERS"] = str(self.get_resources()["cpu_count"])
            subprocess.check_call(args, cwd=exec_dir, env=env)
            volume = np.zeros((image.shape[0] - 2 * self.get_z_pad(),
                               image.shape[1] - 2 * self.get_y_pad(),
                               image.shape[2] - 2 * self.get_x_pad()),
                              np.uint8)
            membrane_class = self.config["membrane_class"]
            for z in range(volume.shape[0]):
                path = os.path.join(
                    dir_out, "%04d.png-probs-%s.png" % (z, membrane_class))
                volume[z] = imread(path, 2)
            return dict(membrane=volume)
                     
        finally:
            for directory in dir_in, dir_out:
                for filename in os.listdir(directory):
                    os.remove(os.path.join(directory, filename))
                os.rmdir(directory)