'''Repair a segmentation'''

import h5py
import luigi
import numpy as np
from scipy.ndimage import zoom

from .utilities import DatasetMixin, RunMixin
from ..parameters import EMPTY_LOCATION
from ..targets.volume_target import DestVolumeReader

class RepairSegmentationTask(DatasetMixin, RunMixin, luigi.Task):
    task_namespace='ariadne_microns_pipeline'
    
    segmentation_loading_plan_path = luigi.Parameter(
        description="The path to the original segmentation's loading plan")
    repair_file=luigi.Parameter(
        description="The repaired segmentation as an .hdf5 file")
    repair_file_dataset_name = luigi.Parameter(
        default="stack",
        description="The name of the dataset within the hdf5 file")
    blood_vessel_file = luigi.Parameter(
        default = EMPTY_LOCATION,
        description="The file of areas to mask out (blood vessels)")
    blood_vessel_dataset_name = luigi.Parameter(
        default="stack",
        description="The name of the dataset within the blood vessel file")
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
    local_mapping=luigi.ListParameter(
        description="A list of the local segment IDs in the volume")
    global_mapping=luigi.ListParameter(
        description="A list of the global segment IDs matching the local ones")
    segments_to_repair=luigi.ListParameter(
        description="The global IDs of segments that should be repaired")
    repair_segments_to_exclude=luigi.ListParameter(
        default=[],
        description="Mask out these segments from the repair volume")
    
    def input(self):
        lp = DestVolumeReader(self.segmentation_loading_plan_path)
        for sp in lp.get_source_targets():
            yield sp
    
    def ariadne_run(self):
        lp = DestVolumeReader(self.segmentation_loading_plan_path)
        seg = lp.imread()
        
        repair = self.load_volume(self.repair_file,
                                  self.repair_file_dataset_name)
        #
        # Apply the global mapping. Zero out the segments to be repaired
        #
        mapping = np.zeros(np.max(seg) + 1, seg.dtype)
        mapping[np.array(self.local_mapping)] = np.array(self.global_mapping)
        for broken in self.segments_to_repair:
            mapping[mapping == broken] = 0
        seg = mapping[seg]
        for broken in self.repair_segments_to_exclude:
            repair[broken] = 0
        #
        # Zero out the blood vessels too
        #
        if self.blood_vessel_file != EMPTY_LOCATION:
            bv = self.load_volume(self.blood_vessel_file,
                                  self.blood_vessel_dataset_name)
            seg[bv] = 0
        #
        # Replace the repair voxels
        #
        mask = repair != 0
        seg[mask] = repair[mask]
        #
        # Calculate metadata
        #
        hist = np.bincount(seg.flatten())
        labels = np.where(hist > 0)[0]
        if hist[0] > 0:
            labels = labels[:1]
        areas = hist[labels].tolist()
        labels = labels.tolist()
        self.output().imwrite(seg, metadata=dict(areas=areas, labels=labels))
    
    def load_volume(self, filename, dataset_name):
        x0d = lp.volume.x / self.x_upsampling
        x1d = (lp.volume.x1 + self.x_upsampling - 1) / self.x_upsampling
        y0d = lp.volume.y / self.y_upsampling
        y1d = (lp.volume.y1 + self.y_upsampling - 1) / self.y_upsampling
        z0d = lp.volume.z / self.z_upsampling
        z1d = (lp.volume.z1 + self.z_upsampling - 1) / self.z_upsampling
        with h5py.File(filename, "r") as fd:
            repair_downsampled_ds = fd[dataset_name]
            repair_downsampled = ds[z0d:z1d, y0d:y1d, x0d:x1d]
        #
        # Upsample the repair volume
        #
        repair = zoom(repair_downsampled,
                          (self.z_upsampling, self.y_upsampling, self.x_upsampling),
                          order=0)
        x0u, x1u = [_ * self.x_upsampling for _ in x0d, x1d]
        y0u, y1u = [_ * self.y_upsampling for _ in y0d, y1d]
        z0u, z1u = [_ * self.z_upsampling for _ in z0d, z1d]
        return repair[lp.volume.z - z0u:lp.volume.z1 - z0u,
                            lp.volume.y - y0u:lp.volume.y1 - y0u,
                            lp.volume.x - x0u:lp.volume.x1 - x0u]
        