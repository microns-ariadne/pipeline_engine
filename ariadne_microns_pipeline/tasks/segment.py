import luigi
import numpy as np
from scipy.ndimage import gaussian_filter, label
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix

from .find_seeds import Dimensionality
from ..algorithms import watershed
from ..algorithms.morphology import parallel_distance_transform
from ..parameters import EMPTY_LOCATION, VolumeParameter
from ..targets import DestVolumeReader
from .utilities import RequiresMixin, RunMixin, SingleThreadedMixin, \
     DatasetMixin

class SegmentTaskMixin(DatasetMixin):
    '''Segment a 3d volume
    
    Given a probability map, find markers and perform a segmentation on
    that volume.
    '''
    prob_loading_plan_path = luigi.Parameter(
        description="The location of the probability volume")
    seed_loading_plan_path = luigi.Parameter(
        description="The location of the seeds for the watershed")
    mask_loading_plan_path = luigi.Parameter(
        description="The location of the mask volume")
    dimensionality = luigi.EnumParameter(
        enum=Dimensionality,
        description="Do either a 2D or 3D watershed, depending on this param")

    def input(self):
        loading_plans = [self.prob_loading_plan_path, 
                         self.seed_loading_plan_path,
                         self.mask_loading_plan_path]
        for loading_plan in loading_plans:
            for tgt in DestVolumeReader(loading_plan).get_source_targets():
                yield tgt
    
    def estimate_memory_usage(self):
        '''Return an estimate of bytes of memory required by this task'''
        v1 = np.prod([1888, 1416, 70])
        m1 = 3510513 * 1000
        v2 = np.prod([1888, 1416, 42])
        m2 = 1929795 * 1000
        volume = self.output().volume
        #
        # Model is Ax + B where x is volume in voxels
        #
        B = (v1 * m2 - v2 * m1) / (v1 - v2)
        A = (float(m1) - B) / v1
        v = np.prod([volume.width, volume.height, volume.depth])
        return int(A * v + B)

class SegmentRunMixin:
    
    sigma_xy = luigi.FloatParameter(
        description=
        "The sigma of the smoothing Gaussian in the x & y directions",
        default=3)
    sigma_z = luigi.FloatParameter(
        description=
        "The sigma of the smoothing Gaussian in the z direction",
        default=.4)
    use_distance = luigi.BoolParameter(
        description="Do a watershed on the distance transform of the membrane")
    threshold = luigi.IntParameter(
        default=128,
        description="The threshold for membrane if using the distance transform")
    xy_nm = luigi.FloatParameter(
        default=4,
        description="Size of a voxel in the x and y directions")
    z_nm = luigi.FloatParameter(
        default=30,
        description="Size of a voxel in the z direction")
    n_cores = luigi.IntParameter(
        default=4,
        description="# of cores to use when computing distance transform")
    xy_overlap = luigi.IntParameter(
        default=100,
        description="Amt of overlap between blocks in the x and y directions")
    z_overlap = luigi.IntParameter(
        default=10,
        description="Amt of overlap between blocks in the z direction")
    xy_block_size = luigi.IntParameter(
        default=512,
        description="X and Y block size for parallel distance transform")
    z_block_size = luigi.IntParameter(
        default=50,
        description="Z block size for parallel distance transform")

    def ariadne_run(self):
        prob_volume, seed_volume, mask_volume = [
            DestVolumeReader(_) for _ in
            self.prob_loading_plan_path,
            self.seed_loading_plan_path,
            self.mask_loading_plan_path]
        seg_volume = self.output()
        prob = prob_volume.imread()
        labels = seed_volume.imread()
        mask = mask_volume.imread() != 0
        labels[~ mask] = 0
        prob[~mask] = 255
        if self.use_distance:
            not_membrane = prob < self.threshold
            dt = parallel_distance_transform(
                not_membrane, self.xy_nm, self.z_nm, 
                self.xy_overlap, self.z_overlap,
                self.xy_block_size, self.z_block_size, self.n_cores)
            max_dt = np.max(dt)
            prob = ((max_dt - dt) * 255 / max_dt).astype(np.uint8)
            del dt
            del not_membrane
        else:
            smoothed = gaussian_filter(
                prob, (self.sigma_z, self.sigma_xy, self.sigma_xy))
            smoothed[~mask] = 255
            prob = smoothed
        if self.dimensionality == Dimensionality.D3:
            seg = watershed(prob, labels)
        else:
            seg = np.zeros(prob.shape, np.uint16)
            for z in range(prob.shape[0]):
                seg[z:z+1] = watershed(prob[z:z+1], labels[z:z+1])
        seg[~mask] = 0
        seg_volume.imwrite(seg.astype(np.uint32))

class SegmentTask(SegmentTaskMixin, SegmentRunMixin, RequiresMixin, 
                  RunMixin, SingleThreadedMixin, luigi.Task):
    
    task_namespace = "ariadne_microns_pipeline"

class SegmentCCTaskMixin(DatasetMixin):
    
    prob_loading_plan_path = luigi.Parameter(
        description="The location of the probability volume")
    mask_loading_plan_path = luigi.Parameter(
        default=EMPTY_LOCATION,
        description="The location of the mask volume")
    threshold = luigi.IntParameter(
        default=190,
        description="The probability threshold (from 0-255) to use as a"
        "cutoff for (not) membrane")
    classes = luigi.ListParameter(
        default=[],
        description="If the volume is categorical (e.g. 1=pre-synaptic, "
        "2=post-synaptic), then numeric classes (e.g. [1, 2] for both) "
        "can be used instead of a numeric threshold.")
    fg_is_higher = luigi.BoolParameter(
        description="True if the foreground is > threshold")
    
    def input(self):
        loading_plans = [self.prob_loading_plan_path]
        if self.mask_loading_plan_path != EMPTY_LOCATION:
            loading_plans.append(self.mask_loading_plan_path)
        for loading_plan in loading_plans:
            for tgt in DestVolumeReader(loading_plan).get_source_targets():
                yield tgt
    
    def apply_threshold(self, volume):
        '''Apply the threshold to the volume to get a binary volume
        
        volume: a Numpy array to threshold
        '''
        if len(self.classes) > 0:
            fg = np.zeros(volume.shape, bool)
            for value in self.classes:
                fg |= volume == value
        elif self.fg_is_higher:
            fg = (volume > self.threshold)
        else:
            fg = (volume < self.threshold)
        return fg

class SegmentCC2DRunMixin:    

    sigma = luigi.FloatParameter(
        default=0.0,
        description="Smoothing sigma of Gaussian applied prior to segmentation")
    
    def ariadne_run(self):
        prob_target = DestVolumeReader(self.prob_loading_plan_path)
        if self.mask_loading_plan_path != EMPTY_LOCATION:
            mask_target = DestVolumeReader(self.mask_loading_plan_path)
        else:
            mask_target = None
        threshold = self.threshold
        if self.sigma > 0:
            prob = gaussian_filter(prob_target.imread(),
                                   (0, self.sigma, self.sigma))
        fg = self.apply_threshold(volume)
        del prob
        if mask_target is not None:
            fg = fg & mask_target.imread()
            
        labels = np.zeros(fg.shape, np.uint32)
        offset = 0
        for i in range(labels.shape[0]):
            l, count = label(fg[i])
            m = l != 0
            labels[i, m] = l[m] + offset
            offset += count
        self.output().imwrite(labels)

class SegmentCC2DTask(SegmentCCTaskMixin,
                    SegmentCC2DRunMixin,
                    RequiresMixin,
                    RunMixin,
                    SingleThreadedMixin,
                    luigi.Task):
    '''The Segment2DTask performs 2D connected components on membrane probs
    
    The task breaks a volume into X/Y planes. It thresholds the membrane
    probabilities and then it finds the connected components in the plane,
    using non-membrane as the foreground.
    '''

class SegmentCC3DRunMixin: 
    xy_sigma = luigi.FloatParameter(
        default=0.0,
        description="Smooothing sigma in the x and y direction "
                    "applied prior to thresholding")
    z_sigma = luigi.FloatParameter(
        default=0.0,
        description="Smoothing sigma applied in the z direction")
    
    def ariadne_run(self):
        prob_target = DestVolumeReader(self.prob_loading_plan_path)
        if self.mask_loading_plan_path != EMPTY_LOCATION:
            mask_target = DestVolumeReader(self.mask_loading_plan_path)
        else:
            mask_target = None
        prob = prob_target.imread()
        if self.xy_sigma > 0 or self.z_sigma > 0:
            prob = gaussian_filter(
                prob, sigma=(self.z_sigma, self.xy_sigma, self.xy_sigma))
        fg = self.apply_threshold(prob)
        del prob
        if mask_target is not None:
            fg = fg & mask_target.imread()
        labels, count = label(fg)
        self.output().imwrite(labels)

class SegmentCC3DTask(SegmentCCTaskMixin,
                    SegmentCC3DRunMixin,
                    RequiresMixin,
                    RunMixin,
                    SingleThreadedMixin,
                    luigi.Task):
    '''The Segment2DTask performs 3D connected components on membrane probs
    
    The task thresholds the input and then performs a 3d connected components
    on the foreground using a 6-connected structuring element.
    '''

class UnsegmentTaskMixin(DatasetMixin):
    input_loading_plan_path = luigi.Parameter(
        description="The location for the input segmentation")
    
    def input(self):
        for tgt in DestVolumeReader(self.input_loading_plan_path) \
            .get_source_targets():
            yield tgt
    
class UnsegmentRunMixin:
    use_min_contact = luigi.BoolParameter(
        default=False,
        description="Break an object between two planes with a minimum of "
        "contact")
    contact_threshold = luigi.IntParameter(
        default=100,
        description="Break objects with less than this number of area overlap")
    
    def ariadne_run(self):
        if self.use_min_contact:
            self.break_at_min_contact()
        else:
            self.shard_all_objects()
    
    def break_at_min_contact(self):
        '''Break objects in the Z direction at points of minimum contact'''
        stack = DestVolumeReader(self.input_loading_plan_path).imread()
        plane1 = stack[0]
        #
        # The offset for new objects
        offset = np.max(stack) + 1
        #
        # Mapping of input segmentation to output
        #
        mapping = np.arange(offset)
        for z in range(stack.shape[0]-1):
            #
            # Count the # of pixels per object that overlap
            #
            plane0 = plane1
            plane1 = stack[z+1]
            mask = (plane0 == plane1) & (plane0 != 0)
            histogram = np.bincount(plane0[mask])
            #
            # Find the ones that are below threshold
            #
            not_much_contact = np.where(
                (histogram > 0) & (histogram < self.contact_threshold))[0]
            n_small = len(not_much_contact)
            if n_small > 0:
                #
                # Remap them to new object numbers going forward
                #
                mapping[not_much_contact] = np.arange(n_small) + offset
                offset += n_small
            #
            # Write the plane using the object mapping
            #
            stack[z+1] = mapping[plane1]
        self.output().imwrite(stack)
        
    def shard_all_objects(self):
        '''Convert a 3-d segmentation into planes of 2-d segmentations'''
        
        stack = self.input().next().imread()
        #
        # For each plane, compute the connected components using
        # the connections between similarly-labeled pixels as the edges.
        #
        offset = 1
        idx = np.arange(np.prod(stack.shape[1:])).reshape(stack.shape[1:])
        for z in range(stack.shape[0]):
            plane = stack[z]
            i1 = plane.shape[0]
            j1 = plane.shape[1]
            #
            # Make arrays of the 4-connected components
            #
            a = []
            b = []
            for i0a, i0b, i1a, i1b, j0a, j0b, j1a, j1b in (
                ( 0,   1,  -1,  i1,   0,   0,  j1,  j1),
                ( 1,   0,  i1,  -1,   0,   0,  j1,  j1),
                ( 0,   0,   0,   0,   0,   1,  -1,  j1),
                ( 0,   0,   i1, i1,   1,   0,  j1,  -1)):
                la = plane[i0a:i1a, j0a:j1a].flatten()
                idxa = idx[i0a:i1a, j0a:j1a].flatten()
                lb = plane[i0b:i1b, j0b:j1b].flatten()
                idxb = idx[i0b:i1b, j0b:j1b].flatten()
                mask = (la != 0) & (la == lb)
                a.append(idxa[mask])
                b.append(idxb[mask])
            a.append(idx.flatten())
            b.append(idx.flatten())
            a = np.hstack(a)
            b = np.hstack(b)
            m = coo_matrix((np.ones(len(a)), (a, b)))
            count, l = connected_components(m)
            l[idx[plane == 0]] = 0
            plane = l[idx]
            plane[plane != 0] += plane.dtype.type(offset)
            stack[z] = plane
            offset += count
        self.output().imwrite(stack)

class UnsegmentTask(UnsegmentTaskMixin,
                    UnsegmentRunMixin,
                    RequiresMixin,
                    RunMixin,
                    SingleThreadedMixin,
                    luigi.Task):
    '''Convert a 3D segmentation into a stack of 2D segmentations
    
    A 3D object may be incorrectly connected by a thin neck in the Z
    direction due to anisotropy. This task takes the 3D segmentation and
    breaks it into planes of 2D objects. Each plane is separated and
    a connected-components graph is formed from adjacent (4-connected)
    pixels that have the same label. A new plane of labels is formed from
    the connected components assignments, offset by the number of previous
    connected components
    '''
    
    task_namespace = "ariadne_microns_pipeline"
    
class ZWatershedTaskMixin(DatasetMixin):
    
    volume=VolumeParameter(
         description="The volume to be segmented")
    x_prob_loading_plan_path=luigi.Parameter(
        description="Location of the x probability map")
    y_prob_loading_plan_path=luigi.Parameter(
        description="Location of the y probability map")
    z_prob_loading_plan_path=luigi.Parameter(
        description="Location of the z probability map")

    def input(self):
        for loading_plan in self.x_prob_loading_plan_path,\
            self.y_prob_loading_plan_path, self.z_prob_loading_plan_path:
            for tgt in DestVolumeReader(loading_plan).get_source_targets():
                yield tgt
    
    def estimate_memory_usage(self):
        '''Return an estimate of bytes of memory required by this task'''
        v1 = np.prod([1625, 1625, 204])
        m1 = 19345268 * 1000
        v2 = np.prod([1408, 1408, 145])
        m2 = 9063256 * 1000
        volume = self.output().volume
        #
        # Model is Ax + B where x is volume in voxels
        #
        B = (v1 * m2 - v2 * m1) / (v1 - v2)
        A = (float(m1) - B) / v1
        v = np.prod([volume.width, volume.height, volume.depth])
        return int(A * v + B)
    

class ZWatershedRunMixin:
    
    threshold=luigi.IntParameter(
        default=40000,
        description="The targeted size threshold for a segment")
    
    def ariadne_run(self):
        import zwatershed
        
        xtgt, ytgt, ztgt = [
            DestVolumeReader(loading_plan) for loading_plan in
            self.x_prob_loading_plan_path, self.y_prob_loading_plan_path,
            self.z_prob_loading_plan_path]
        volume = np.zeros((3,
                           xtgt.volume.depth,
                           xtgt.volume.height,
                           xtgt.volume.width), np.uint8)
        for i, tgt in enumerate((ztgt, ytgt, xtgt)):
            volume[i] = tgt.imread()
        result = zwatershed.zwatershed(volume, [self.threshold])[0]
        self.output().imwrite(result)

class ZWatershedTask(ZWatershedTaskMixin,
                     ZWatershedRunMixin,
                     RequiresMixin,
                     RunMixin,
                     SingleThreadedMixin,
                     luigi.Task):
    '''The Z-Watershed task segments a volume using x, y and z affinity maps
    
    '''
    
    task_namespace="ariadne_microns_pipeline"