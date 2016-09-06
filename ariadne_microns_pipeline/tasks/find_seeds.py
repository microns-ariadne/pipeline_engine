import enum
import luigi
import numpy as np
from scipy.ndimage import gaussian_filter, label, distance_transform_edt
from scipy.ndimage import grey_dilation, grey_erosion

from ..parameters import VolumeParameter, DatasetLocationParameter
from ..targets.factory import TargetFactory
from .utilities import RequiresMixin, RunMixin, SingleThreadedMixin


class SeedsMethodEnum(enum.Enum):
    '''Enumeration of the seed finding algorithms'''

    '''Find seeds by smoothing with a Gaussian and thresholding
    
    The probabilities are smoothed with an anisotropic Gaussian with
    different x/y and z sigmas, then the result is thresholded and
    connected components are found.
    '''
    Smoothing=1
    
    '''Find seeds by finding maxima in the distance transform
    
    The probabilities are thresholded, then the distance from the
    membrane is computed. A top-hat filter is applied to find the maximum
    within a given radius and a distance threshold is applied to weed out
    seeds near the membrane.
    '''
    DistanceTransform=2
    
    '''Use connected components instead of finding seeds
    
    The probabilities are thresholded and connected components is run
    on pixels lower than the threshold.
    '''
    ConnectedComponents=3


class Dimensionality(enum.Enum):
    '''Determines whether to do something in 2D or 3D'''
    
    '''Process a 3d volume as 2d planes'''
    D2=2
    
    '''Process a 3d volume as a whole'''
    D3=3
    
class FindSeedsTaskMixin:
    
    volume = VolumeParameter(
        description="The volume being segmented")
    prob_location = DatasetLocationParameter(
        description="The location of the membrane probabilities")
    seeds_location = DatasetLocationParameter(
        description="The location of the volume with seed labels")
    
    def input(self):
        yield TargetFactory().get_volume_target(
            location=self.prob_location, volume=self.volume)
    
    def output(self):
        return TargetFactory().get_volume_target(
            location=self.seeds_location, volume=self.volume)


class FindSeedsRunMixin:
    
    dimensionality=luigi.EnumParameter(
        enum=Dimensionality,
        description="Whether to find seeds in each 2D plane or in the "
        "volume as a whole")
    method=luigi.EnumParameter(
        enum=SeedsMethodEnum,
        description="The algorithm used to find seeds")
    sigma_xy = luigi.FloatParameter(
        description=
        "The sigma of the smoothing Gaussian in the x & y directions",
        default=3)
    sigma_z = luigi.FloatParameter(
        description=
        "The sigma of the smoothing Gaussian in the z direction",
        default=.4)
    threshold = luigi.FloatParameter(
        description="The intensity threshold cutoff for the seeds",
        default=1)
    minimum_distance = luigi.FloatParameter(
        default=5,
        description="The minimum distance allowed between seeds")
    distance_threshold = luigi.FloatParameter(
        default=5,
        description="The distance threshold cutoff for the seeds")
    
    def find_using_2d_smoothing(self, probs):
        '''Find seeds in each plane, smoothing, then thresholding
        
        :param probs: the probability volume
        '''
        offset=0
        seeds = []
        for plane in probs.astype(np.float32):
            smoothed = gaussian_filter(plane.astype(np.float32), self.sigma_xy)
            eroded = grey_erosion(smoothed, size=self.minimum_distance)
            thresholded = (smoothed < self.threshold) & (smoothed == eroded)
            labels, count = label(thresholded)
            labels[labels != 0] += offset
            offset += count
            seeds.append(labels)
        return np.array(seeds)
    
    def find_using_3d_smoothing(self, probs):
        '''Find seeds after smoothing and thresholding

        :param probs: the probability volume
        '''
        sigma = (self.sigma_z, self.sigma_xy, self.sigma_xy)
        smoothed = gaussian_filter(probs.astype(np.float32), sigma)
        eroded = grey_erosion(smoothed, size=self.minimum_distance)
        thresholded = (smoothed < self.threshold) & (smoothed == eroded)
        labels, count = label(thresholded)
        return labels
    
    def find_using_2d_distance(self, probs):
        '''Find seeds in each plane by distance transform

        :param probs: the probability volume
        '''
        offset=0
        seeds = []
        for plane in probs.astype(np.float32):
            thresholded = plane < self.threshold
            distance = distance_transform_edt(thresholded)
            dilated = grey_dilation(distance, size=self.minimum_distance)
            mask = (distance == dilated) & (distance >= self.distance_threshold)
            labels, count = label(mask)
            labels[labels != 0] += offset
            offset += count
            seeds.append(labels)
        return np.array(seeds)
    
    def find_using_3d_distance(self, probs):
        distance = []
        for plane in probs.astype(np.float32):
            thresholded = plane < self.threshold
            distance.append(distance_transform_edt(thresholded))
        distance = np.array(distance)
        dilated = grey_dilation(distance, size=self.minimum_distance)
        mask = (distance == dilated) & (distance >= self.distance_threshold)
        labels, count = label(mask)
        return labels
        
    def ariadne_run(self):
        prob_target = self.input().next()
        probs = prob_target.imread()
        if self.method == SeedsMethodEnum.Smoothing:
            if self.dimensionality == Dimensionality.D2:
                seeds = self.find_using_2d_smoothing(probs)
            else:
                seeds = self.find_using_3d_smoothing(probs)
        else:
            if self.dimensionality == Dimensionality.D2:
                seeds = self.find_using_2d_distance(probs)
            else:
                seeds = self.find_using_3d_distance(probs)
        seeds = seeds.astype(np.uint32)
        self.output().imwrite(seeds)


class FindSeedsTask(FindSeedsTaskMixin,
                    FindSeedsRunMixin,
                    RequiresMixin,
                    RunMixin,
                    SingleThreadedMixin,
                    luigi.Task):
    '''Find seeds for a watershed.
    
    This task takes a probability map as input and produces a "segmentation"
    where the seeds for a watershed are labeled, using a different index
    per seed.
    
    You can choose between different algorithms using the "method" parameter
    and you can either find seeds in each 2D plane or in the whole 3d
    volume.
    '''
    task_namespace = "ariadne_microns_pipeline"
