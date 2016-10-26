'''Segmentation statistics tasks.

This module's task calculates accuracy statistics and outputs them in JSON
'''

import json
import luigi
import matplotlib
from matplotlib.backends.backend_pdf import FigureCanvasPdf
import numpy as np
from scipy.sparse import coo_matrix

from ..algorithms.evaluation import segmentation_metrics, vi, Rand, f_info
from ..parameters import VolumeParameter, DatasetLocationParameter
from ..targets.factory import TargetFactory
from .connected_components import ConnectivityGraph
from .utilities import RequiresMixin, RunMixin


class SegmentationStatisticsTaskMixin:
    
    volume = VolumeParameter(
        description="The volume on which to run the statistics")
    test_location = DatasetLocationParameter(
        description="The location of the test dataset")
    test_volume = VolumeParameter(
        description="The volume of the test dataset")
    ground_truth_location = DatasetLocationParameter(
        description="The location of the ground truth dataset")
    ground_truth_volume = VolumeParameter(
        description="The volume of the ground-truth dataset")
    connectivity = luigi.Parameter(
        default="/dev/null",
        description="The connectivity graph .json file that is the output "
                    "of the AllConnectedComponentsTask. If not present, "
                    "then statistics are done on the local label IDs.")
    output_path = luigi.Parameter(
        description="The path to the JSON output file.")
    
    def input(self):
        tf = TargetFactory()
        yield tf.get_volume_target(self.test_location, self.test_volume)
        yield tf.get_volume_target(self.ground_truth_location, 
                                   self.ground_truth_volume)
        if self.connectivity != "/dev/null":
            yield luigi.LocalTarget(self.connectivity)
    
    def output(self):
        return luigi.LocalTarget(path=self.output_path)


class SegmentationStatisticsRunMixin:
    
    def cutout(self, segmentation, volume):
        '''Limit the segmentation to the task's volume
        
        :param segmentation: the input segmentation
        :param volume: the global location of the segmentation
        
        returns the segmentation restricted to the task's volume
        '''
        x0 = max(0, self.volume.x - volume.x)
        x1 = min(volume.width - x0, self.volume.x1 - volume.x)
        y0 = max(0, self.volume.y - volume.y)
        y1 = min(volume.height - y0, self.volume.y1 - volume.y)
        z0 = max(0, self.volume.z - volume.z)
        z1 = min(volume.depth - z0, self.volume.z1 - volume.z)
        return segmentation[z0:z1, y0:y1, x0:x1]
    
    def ariadne_run(self):
        '''Run the segmentation_metrics on the test and ground truth'''
        
        inputs = self.input()
        test_volume = inputs.next()
        gt_volume = inputs.next()
        test_labels = self.cutout(test_volume.imread(), test_volume.volume)
        gt_labels = self.cutout(gt_volume.imread(), gt_volume.volume)
        try:
            with inputs.next().open("r") as fd:
                c = ConnectivityGraph.load(fd)
                test_labels = c.convert(test_labels, test_volume.volume)
        except StopIteration:
            pass
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
                 vi=vi,
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
        for statistics_location in statistics_locations:
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
                 vi=[])
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

        #
        # Rollup the statistics
        #
        matrix = coo_matrix((np.hstack(counts),
                             (np.hstack(gt), np.hstack(detected))))
        matrix.sum_duplicates()
        gt, detected = matrix.nonzero()
        counts = matrix.tocsr()[gt, detected]
        gt_counts = np.bincount(gt, counts)
        gt_counts = gt_counts[gt_counts > 0]
        detected_counts = np.bincount(detected, counts)
        detected_counts = detected_counts[detected_counts > 0]
        tot_vi = vi(counts, gt_counts, detected_counts)
        tot_rand = Rand(counts, gt_counts, detected_counts, .5)
        tot_f_info = f_info(counts, gt_counts, detected_counts, .5)
        
        figure = matplotlib.figure.Figure()
        figure.set_size_inches(8, 11)
        columns = filter(lambda _:_ != "vi", sorted(d.keys()))
        vi_data = d['vi']
        ax = figure.add_axes((0.05, 0.1, 0.65, 0.60))
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
        vi_ax = figure.add_axes((0.75, 0.1, 0.20, 0.80))
        vi_ax.boxplot(vi_data[~np.isnan(vi_data)], labels=['VI (nats)'])
        ann = vi_ax.annotate(
            "%.2f" % vi.mean(),
            xy=(1, vi_data.mean()),
            xytext=(1.2, vi_data.mean()+.1),
            bbox=dict(boxstyle="round", fc="white", ec="gray"),
            arrowprops=dict(arrowstyle="->", color="gray"))
        matplotlib.pyplot.setp(ann, fontsize=6)
        for a in ax, vi_ax:
            matplotlib.pyplot.setp(a.get_xmajorticklabels(), fontsize=8)
            matplotlib.pyplot.setp(a.get_ymajorticklabels(), fontsize=6)
        figure.text(.5, .95, "Segmentation accuracy", ha='center', va='top',
                size='x-large')
        totals = "VI=%.2f, Rand=%.2f, F info=%.2f" % \
            (tot_vi, tot_rand, tot_f_info)
        figure.text(.5, .75, totals, ha='center', va='bottom', size='large')
        canvas = FigureCanvasPdf(figure)
        figure.savefig(self.pdf_location)