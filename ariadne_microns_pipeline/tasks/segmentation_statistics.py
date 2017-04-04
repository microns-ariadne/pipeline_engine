'''Segmentation statistics tasks.

This module's task calculates accuracy statistics and outputs them in JSON
'''

import json
import luigi
import matplotlib
from matplotlib.backends.backend_pdf import FigureCanvasPdf
import numpy as np
from scipy.sparse import coo_matrix
from scipy.ndimage import grey_erosion, grey_dilation

from ..algorithms.evaluation import segmentation_metrics, Rand, f_info
from ..algorithms.vi import split_vi, bits_to_nats
from ..targets import DestVolumeReader
from ..parameters import EMPTY_LOCATION
from .connected_components import ConnectivityGraph
from .utilities import RequiresMixin, RunMixin


class SegmentationStatisticsTaskMixin:
    
    test_loading_plan_path = luigi.Parameter(
        description="The location of the test dataset")
    ground_truth_loading_plan_path = luigi.Parameter(
        description="The location of the ground truth dataset")
    connectivity = luigi.Parameter(
        default=EMPTY_LOCATION,
        description="The connectivity graph .json file that is the output "
                    "of the AllConnectedComponentsTask. If not present, "
                    "then statistics are done on the local label IDs.")
    output_path = luigi.Parameter(
        description="The path to the JSON output file.")
    
    def input(self):
        for loading_plan in self.test_loading_plan_path, \
            self.ground_truth_loading_plan_path:
            for tgt in DestVolumeReader(loading_plan).get_source_targets():
                yield tgt
        if self.connectivity != EMPTY_LOCATION:
            yield luigi.LocalTarget(self.connectivity)
    
    def output(self):
        return luigi.LocalTarget(path=self.output_path)


class SegmentationStatisticsRunMixin:
    
    xy_erosion = luigi.IntParameter(
        default=1,
        description="# of pixels to erode segmentations in the x and y "
        "directions.")
    z_erosion = luigi.IntParameter(
        default=1,
        description="# of pixels to erode segmentations in the z direction")
    
    def erode_seg(self, seg):
        '''Erode the segmentation passed (or not)
        
        The segmentation is eroded by the factors in the xy_erosion and 
        z_erosion
        
        seg: the segmentation to erode, which is done in-place
        '''
        if self.xy_erosion == 0 and self.z_erosion == 0:
            return
        strel = np.ones((self.z_erosion*2 + 1, 
                         self.xy_erosion*2 + 1,
                         self.xy_erosion*2 + 1), bool)
        mask = grey_erosion(seg, footprint=strel) == \
               grey_dilation(seg, footprint=strel)
        seg[~ mask] = 0

    def ariadne_run(self):
        '''Run the segmentation_metrics on the test and ground truth'''
        
        test_volume = DestVolumeReader(self.test_loading_plan_path)
        gt_volume = DestVolumeReader(self.ground_truth_loading_plan_path)
        test_labels = test_volume.imread()
        self.erode_seg(test_labels)
        gt_labels = gt_volume.imread()
        self.erode_seg(gt_labels)
        if self.connectivity != EMPTY_LOCATION:
            with open(self.connectivity, "r") as fd:
                c = ConnectivityGraph.load(fd)
                test_labels = c.convert(test_labels, test_volume.volume)
        d = segmentation_metrics(gt_labels, test_labels, per_object=True)
        rand = d["Rand"]
        F_Info = d["F_Info"]
        vi = d["VI"]
        gt, detected, counts = d["tot_pairwise"]
        pairs = dict(gt=gt.tolist(), 
                     detected=detected.tolist(),
                     counts=counts.tolist())
        d = dict(rand=rand["F-score"],
                 rand_split=rand["split"],
                 rand_merge=rand["merge"],
                 vi=vi["F-score"],
                 vi_split=vi["split"],
                 vi_merge=vi["merge"],
                 F_info=F_Info["F-score"],
                 F_info_split=F_Info["split"],
                 F_info_merge=F_Info["merge"],
                 per_object = d["per_object"],
                 pairs=pairs,
                 x=self.volume.x,
                 y=self.volume.y,
                 z=self.volume.z,
                 width=self.volume.width,
                 height=self.volume.height,
                 depth=self.volume.depth)
        with self.output().open("w") as fd:
            json.dump(d, fd)


class SegmentationStatisticsTask(SegmentationStatisticsTaskMixin,
                                 SegmentationStatisticsRunMixin,
                                 RequiresMixin, RunMixin,
                                 luigi.Task):
    '''Compute the Rand index and V-info scores for a segmentation
    
    Given a segmented volume and a ground truth segmentation volume,
    compute the Rand Index and V-info scores. The scores are stored in
    a JSON-encoded dictionary. Separate scores are compiled for merge
    and split errors in addition to an overall score.
    
    See http://journal.frontiersin.org/article/10.3389/fnana.2015.00142/full
    (the ISBI 2013 segmentation challenge) for the forumlas and discussion.
    
    Format of JSON dictionary:
    
    rand: rand score
    rand_split: contribution of split segmentations to rand error
    rand_merge: contribution of merged segmentations to rand error
    vi: the VI score (in log base e form). 2*entropy of gt / detected pixel
        pair combinations - the entropy of ground-truth pixels - the
        entropy of the detected pixels.
    F_info: the mutual information of the gt and detected channel divided
            by 1/2 of the sum of the ground truth and predicted entropy
    F_info_split: the contribution to the F_info from splits
    F_info_merge: the contribution to the F_info from merges
    pairs: a dictionary of the ground-truth ids (key = "gt"),
           the detected ids (key = "detected") and the counts of pixels having 
           the given ground-truth, detected ID combinations (key = "counts")
    x, y, z: the origin of the volume
    width, height, depth: the size of the volume.
    '''
    
    task_namespace = "ariadne_microns_pipeline"
    

class SegmentationReportTask(RequiresMixin, RunMixin, luigi.Task):
    '''Compose the segmentation report'''
    
    task_namespace = "ariadne_microns_pipeline"

    statistics_locations = luigi.ListParameter(
        description=
        "Paths to the output files of the SegmentationStatisticsTask")
    pdf_location=luigi.Parameter(
        description="The path to the .pdf report")
    
    def input(self):
        for statistics_location in self.statistics_locations:
            yield luigi.LocalTarget(statistics_location)
    
    def output(self):
        return luigi.LocalTarget(self.pdf_location)
    
    def ariadne_run(self):
        d = dict(rand=[],
                 rand_split=[],
                 rand_merge=[],
                 F_info=[],
                 F_info_split=[],
                 F_info_merge=[],
                 vi=[],
                 vi_split=[],
                 vi_merge=[])
        gt = []
        detected = []
        counts = []
        for tgt in self.input():
            with tgt.open("r") as fd:
                data = json.load(fd)
            for key in d:
                d[key].append(data[key])
            gt.append(data["pairs"]["gt"])
            detected.append(data["pairs"]["detected"])
            counts.append(data["pairs"]["counts"])
        for key in d:
            d[key] = np.array(d[key])
        #
        # Rollup the statistics
        #
        matrix = coo_matrix((np.hstack(counts),
                             (np.hstack(gt), np.hstack(detected))))
        matrix.sum_duplicates()
        matrix = matrix.tocsr()
        #
        # The contingency table for the VI calculation is the fractional
        # component of each gt /detected pair.
        # The order for the vi code is [seg, gt], which is why we transpose
        #
        contingency_table = matrix.transpose() / matrix.sum()
        gt, detected = matrix.nonzero()
        counts = matrix[gt, detected].A1
        gt_counts = np.bincount(gt, counts)
        gt_counts = gt_counts[gt_counts > 0]
        detected_counts = np.bincount(detected, counts)
        detected_counts = detected_counts[detected_counts > 0]
        frac_counts = counts.astype(float) / counts.sum()
        frac_gt = gt_counts.astype(float) / gt_counts.sum()
        frac_detected = detected_counts.astype(float) / detected_counts.sum()
        tot_vi_merge, tot_vi_split = bits_to_nats(split_vi(contingency_table))
        tot_vi = tot_vi_merge + tot_vi_split
        tot_rand = Rand(frac_counts, frac_gt, frac_detected, .5)
        tot_rand_split = Rand(frac_counts, frac_gt, frac_detected, 0)
        tot_rand_merge = Rand(frac_counts, frac_gt, frac_detected, 1)
        tot_f_info = f_info(frac_counts, frac_gt, frac_detected, .5)
        tot_f_info_split = f_info(frac_counts, frac_gt, frac_detected, 0)
        tot_f_info_merge = f_info(frac_counts, frac_gt, frac_detected, 1)
        #
        # Draw it
        #
        figure = matplotlib.figure.Figure()
        figure.set_size_inches(8, 11)
        columns = filter(lambda _:not _.startswith("vi"), sorted(d.keys()))
        vi_data = d['vi']
        vis_data = d['vi_split']
        vim_data = d['vi_merge']
        ax = figure.add_axes((0.05, 0.1, 0.55, 0.70))
        ax.boxplot([d[c][~ np.isnan(d[c])] for c in columns])
        ax.set_xticklabels(columns, rotation=15)
        ax.set_yticks(np.linspace(0, 1.0, 11))
        ax.set_xlim(.5, len(columns)+1)
        for i, mean in enumerate([d[c].mean() for c in columns]):
            ann = ax.annotate("%.2f" % mean, 
                        xy=(i+1, mean),
                        xytext=(i+1.4, mean),
                        bbox=dict(boxstyle="round", fc="white", ec="gray"),
                        arrowprops=dict(arrowstyle="->", color="gray"))
            matplotlib.pyplot.setp(ann, fontsize=6)
        vi_ax = figure.add_axes((0.65, 0.1, 0.30, 0.70))
        vi_ax.boxplot([_[~np.isnan(_)] for _ in vi_data, vim_data, vis_data],
                      labels=['VI\n(nats)', 'Merge', 'Split'])
        for i, data in enumerate([vi_data, vim_data, vis_data]):
            ann = vi_ax.annotate(
                "%.2f" % data.mean(),
                xy=(i+1, data.mean()),
                xytext=(i+1.2, data.mean()+.1),
                bbox=dict(boxstyle="round", fc="white", ec="gray"),
                arrowprops=dict(arrowstyle="->", color="gray"))
            matplotlib.pyplot.setp(ann, fontsize=6)
        for a in ax, vi_ax:
            matplotlib.pyplot.setp(a.get_xmajorticklabels(), fontsize=8)
            matplotlib.pyplot.setp(a.get_ymajorticklabels(), fontsize=6)
            totals = "Total /split / merge VI=%.2f / %.2f / %.2f\n" %\
                (tot_vi, tot_vi_split, tot_vi_merge)
            totals += "Rand=%.2f / %.2f / %.2f\n" % \
                (tot_rand, tot_rand_split, tot_rand_merge)
            totals += "F info=%.2f / %.2f / %.2f" % \
                (tot_f_info, tot_f_info_split, tot_f_info_merge)
        figure.text(.5, .90, "Segmentation accuracy\n"+totals,
                    ha='center', va='top',
                    size='x-large')
        canvas = FigureCanvasPdf(figure)
        figure.savefig(self.pdf_location)