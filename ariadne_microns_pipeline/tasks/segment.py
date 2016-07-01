import luigi
from scipy.ndimage import gaussian_filter, label

from ..algorithms import watershed
from ..targets.factory import TargetFactory


class SegmentTaskMixin:
    '''Segment a 3d volume
    
    Given a probability map, find markers and perform a segmentation on
    that volume.
    '''
    prob_paths = luigi.ListParameter(
        description="The sharded paths to the probability volume")
    prob_dataset_name = luigi.Parameter(
        description="The name of the probability dataset")
    prob_dataset_pattern = luigi.Parameter(
        description="The file pattern to use when naming probability datasets")
    mask_paths = luigi.ListParameter(
        description="The sharded paths to the masks")
    mask_dataset_name = luigi.Parameter(
        description="The name of the mask dataset")
    mask_dataset_pattern = luigi.Parameter(
        description="The file pattern to use when naming mask datasets")
    seg_paths = luigi.ListParameter(
        description="The sharded paths to the output segmentation")
    seg_dataset_name = luigi.Parameter(
       description="The name of the segmentation dataset")
    seg_dataset_pattern = luigi.Parameter(
        description="The file pattern to use when naming segmentation files")
    x = luigi.IntParameter(
        description="The X offset of the volume")
    y = luigi.IntParameter(
        description="The Y offset of the volume")
    z = luigi.IntParameter(
        description="The Z offset of the volume")
    width = luigi.IntParameter(
        description="The width of the volume")
    height = luigi.IntParameter(
        description="The height of the volume")
    depth = luigi.IntParameter(
        description = "The depth of the volume")

    def input(self):
        yield TargetFactory().get_volume_target(
            self.prob_paths,
            self.prob_dataset_name,
            self.prob_dataset_pattern,
            self.x,
            self.y,
            self.z,
            self.width,
            self.height,
            self.depth)
        yield TargetFactory().get_volume_target(
            self.mask_paths,
            self.mask_dataset_name,
            self.mask_dataset_pattern,
            self.x,
            self.y,
            self.z,
            self.width,
            self.height,
            self.depth)
    
    def output(self):
        yield TargetFactory().get_volume_target(
            self.seg_paths,
            self.seg_dataset_name,
            self.seg_dataset_pattern,
            self.x,
            self.y,
            self.z,
            self.width,
            self.height,
            self.depth)
        

class SegmentRunMixin:
    
    #
    # optional parameters
    #
    sigma_xy = luigi.FloatParameter(
        description=
        "The sigma of the smoothing Gaussian in the x & y directions",
        default=3)
    sigma_z = luigi.FloatParameter(
        description=
        "The sigma of the smoothing Gaussian in the z direction",
        default=.4)
    threshold = luigi.FloatParameter(
        description="The threshold cutoff for the seeds",
        default=1)
    
    def ariadne_run(self):
        prob_volume, mask_volume = self.input()
        seg_volume = self.output()
        prob = prob_volume.imread()
        mask = mask_volume.imread() != 0
        sigma = (self.sigma_z, self.sigma_xy, self.sigma_xy)
        smoothed = gaussian_filter(prob.astype(np.float32), sigma)
        bin_markers = smoothed > self.threshold
        labels, count = label(bin_markers)
        seg = watershed(prob, labels)
        seg_volume.imwrite(seg)

class SegmentTask(SegmentTaskMixin, SegmentRunMixin, luigi.Task):
    
    def run(self):
        self.ariadne_run()
