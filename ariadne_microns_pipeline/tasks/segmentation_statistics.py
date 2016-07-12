'''Segmentation statistics tasks.

This module's task calculates accuracy statistics and outputs them in JSON
'''

import json
import luigi
import matplotlib
from matplotlib.backends.backend_pdf import FigureCanvasPdf
import pandas

from ..algorithms.evaluation import segmentation_metrics
from ..parameters import VolumeParameter, DatasetLocationParameter
from ..targets.factory import TargetFactory
from .utilities import RequiresMixin


class SegmentationStatisticsTaskMixin:
    
    volume = VolumeParameter(
        description="The volume of the test and ground_truth datasets")
    test_location = DatasetLocationParameter(
        description="The location of the test dataset")
    ground_truth_location = DatasetLocationParameter(
        description="The location of the ground truth dataset")
    output_path = luigi.Parameter(
        description="The path to the JSON output file.")
    
    def input(self):
        tf = TargetFactory()
        yield tf.get_volume_target(self.test_location, self.volume)
        yield tf.get_volume_target(self.ground_truth_location, self.volume)
    
    def output(self):
        return luigi.LocalTarget(path=self.output_path)


class SegmentationStatisticsRunMixin:
    
    def ariadne_run(self):
        '''Run the segmentation_metrics on the test and ground truth'''
        
        test_volume, gt_volume = list(self.input())
        test_labels = test_volume.imread()
        gt_labels = gt_volume.imread()
        d = segmentation_metrics(gt_labels, test_labels)
        rand = d["Rand"]
        vi = d["VI"]
        d = dict(rand=rand["F-score"],
                 rand_split=rand["split"],
                 rand_merge=rand["merge"],
                 vi=vi["F-score"],
                 vi_split=vi["split"],
                 vi_merge=vi["merge"],
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
                                 RequiresMixin,
                                 luigi.Task):
    '''Compute the Rand index and V-info scores for a segmentation
    
    Given a segmented volume and a ground truth segmentation volume,
    compute the Rand Index and V-info scores. The scores are stored in
    a JSON-encoded dictionary. Separate scores are compiled for merge
    and split errors in addition to an overall score.
    
    See http://journal.frontiersin.org/article/10.3389/fnana.2015.00142/full
    (the ISBI 2013 segmentation challenge) for the forumlas and discussion.
    '''
    
    task_namespace = "ariadne_microns_pipeline"
    
    def run(self):
        self.ariadne_run()


class SegmentationReportTask(RequiresMixin, luigi.Task):
    '''Compose the segmentation report'''
    
    csv_location=luigi.Parameter(
        description="The path to the .csv file with the statistics")
    pdf_location=luigi.Parameter(
        description="The path to the .pdf report")
    
    def input(self):
        yield luigi.LocalTarget(self.csv_location)
    
    def output(self):
        return luigi.LocalTarget(self.pdf_location)
    
    def run(self):
        dataframe = pandas.read_csv(
            self.csv_location)
        figure = matplotlib.figure.Figure()
        columns = sorted(filter(
            lambda _:_ not in ('x', 'y', 'z', 'width', 'height', 'depth'),
            dataframe.columns))
        ax = figure.add_subplot(1, 1, 1)
        ax.boxplot([dataframe[c] for c in columns],
                   labels = columns)
        ax.set_title("Segmentation accuracy")
        canvas = FigureCanvasPdf(figure)
        figure.savefig(self.pdf_location)