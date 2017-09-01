import h5py
import luigi
import numpy as np
from scipy.ndimage import gaussian_filter, zoom

from .utilities import RequiresMixin, RunMixin, DatasetMixin
from ..parameters import EMPTY_LOCATION
from ..targets import DestVolumeReader


class MaskBorderTaskMixin(DatasetMixin):
    prob_loading_plan_path = luigi.Parameter(
        description="The location of the probability files")
    
    def input(self):
        for tgt in DestVolumeReader(self.prob_loading_plan_path) \
            .get_source_targets():
            yield tgt
    
    def estimate_memory_usage(self):
        '''Return an estimate of bytes of memory required by this task'''
        v1 = np.prod([1888, 1416, 70])
        m1 = 716834 * 1000
        v2 = np.prod([1888, 1416, 42])
        m2 = 468952 * 1000
        volume = self.output().volume
        #
        # Model is Ax + B where x is the output volume
        #
        B = (v1 * m2 - v2 * m1) / (v1 - v2)
        A = (float(m1) - B) / v1
        v = np.prod([volume.width, 
                     volume.height, 
                     volume.depth])
        return int(A * v + B)


class MaskBorderRunMixin:
    #
    # Optional parameters
    #
    threshold = luigi.IntParameter(
        default = 250,
        description="Mask out voxels whose membrane probability is this value "
                    "or higher (range = 0-256)")
    smoothing_xy = luigi.FloatParameter(
        default=0,
        description="Smoothing in the x and y direction. "
                    "Zero means smoothing off")
    smoothing_z = luigi.FloatParameter(
        default=0,
        description="Smoothing in the Z direction")
    mask_file = luigi.Parameter(
        default=EMPTY_LOCATION,
        description="An HDF5 file containing a mask of additional masked "
        "areas")
    mask_dataset_name = luigi.Parameter(
         default="mask",
         description="The name of the dataset within the mask file")
    mask_x_resolution = luigi.FloatParameter(
        default=1.0,
        description="The X resolution of the mask in terms of volume voxels "
        "per mask voxel, e.g. \"2.0\" means that the mask is 1/2 res.")
    mask_y_resolution = luigi.FloatParameter(
        default=1.0,
        description="The Y resolution of the mask in terms of volume voxels "
        "per mask voxel, e.g. \"2.0\" means that the mask is 1/2 res.")
    mask_z_resolution = luigi.FloatParameter(
        default=1.0,
        description="The Z resolution of the mask in terms of volume voxels "
        "per mask voxel, e.g. \"2.0\" means that the mask is 1/2 res.")
    
    def ariadne_run(self):
        '''Create the mask of pixels to watershed'''
        prob_lp = DestVolumeReader(self.prob_loading_plan_path)
        prob = prob_lp.imread()
        if self.smoothing_xy != 0 or self.smoothing_z != 0:
            prob = gaussian_filter(prob.astype(float32),
                                   sigma=(self.smoothing_z,
                                          self.smoothing_xy,
                                          self.smoothing_xy))
        mask = (prob < self.threshold).astype(np.uint8)
        del prob
        if self.mask_file != EMPTY_LOCATION:
            with h5py.File(self.mask_file, "r") as fd:
                ds = fd[self.mask_dataset_name]
                x0p = prob_lp.volume.x
                x1p = prob_lp.volume.x1
                y0p = prob_lp.volume.y
                y1p = prob_lp.volume.y1
                z0p = prob_lp.volume.z
                z1p = prob_lp.volume.z1
                if self.mask_x_resolution == 1 and \
                   self.mask_y_resolution == 1 and \
                   self.mask_z_resolution == 1:
                    mask = mask & ds[z0p:z1p, y0p:y1p, x0p:x1p]
                else:
                    x0m, x1m = [int(float(x) / self.mask_x_resolution)
                                for x in x0p, x1p]
                    y0m, y1m = [int(float(y) / self.mask_y_resolution)
                                                for y in y0p, y1p]
                    z0m, z1m = [int(float(z) / self.mask_z_resolution)
                                                for z in z0p, z1p]
                    mask = mask & zoom(
                        ds[z0m:z1m, y0m:y1m, x0m:x1m],
                        zoom=(self.mask_z_resolution,
                              self.mask_y_resolution, 
                              self.mask_x_resolution),
                        order=0)
                    
        self.output().imwrite(mask)


class MaskBorderTask(MaskBorderTaskMixin, MaskBorderRunMixin, 
                     RequiresMixin, RunMixin, luigi.Task):

    task_namespace = "ariadne_microns_pipeline"
