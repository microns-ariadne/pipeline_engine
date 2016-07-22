import json
import luigi
import numpy as np
import os
import rh_logger

from ..parameters import VolumeParameter, DatasetLocationParameter
from ..parameters import Volume, DatasetLocation
from .utilities import RequiresMixin, RunMixin
from ..targets.factory import TargetFactory
from .download_from_butterfly import DownloadFromButterflyTask

class VisualizeTaskMixin:
    
    volume = VolumeParameter(
        description="The volume to visualize")
    image_location=DatasetLocationParameter(
        description="The location of the image dataset")
    prob_location=DatasetLocationParameter(
        description="The location of the membrane probabilities")
    seeds_location=DatasetLocationParameter(
        description="The location of the watershed seed segmentation")
    seg_location=DatasetLocationParameter(
        description="The location of the segmentation")
    output_location=luigi.Parameter(
        description="The location for the .mp4 movie output")
    
    def input(self):
        yield TargetFactory().get_volume_target(self.image_location,
                                                self.volume)
        yield TargetFactory().get_volume_target(self.prob_location,
                                                self.volume)
        yield TargetFactory().get_volume_target(self.seeds_location,
                                                self.volume)
        yield TargetFactory().get_volume_target(self.seg_location,
                                                self.volume)
    
    def output(self):
        return luigi.LocalTarget(self.output_location)


class VisualizeRunMixin:
    #
    # Optional parameters
    #
    size_x_inches=luigi.FloatParameter(
        default=6,
        description="The target size of the output in the X direction "
        "in inches")
    size_y_inches=luigi.FloatParameter(
        default=6,
        description="The target size of the output in the Y direction "
        "in inches")
    first_frame=luigi.IntParameter(
        default=0,
        description="Start at this frame instead of frame # 0")
    end_frame=luigi.IntParameter(
        default=0,
        description="Stop the movie at this frame. Default=do all")
    
    def ariadne_run(self):
        from matplotlib import animation
        from matplotlib.cm import ScalarMappable
        import matplotlib.pyplot as plt
        
        #
        # Some of this is from a recipe 
        # http://jakevdp.github.io/blog/2013/05/12/
        #        embedding-matplotlib-animations/
        #
        image_target, prob_target, seeds_target, seg_target = list(self.input())
        image_volume = image_target.imread()
        prob_volume = prob_target.imread()
        seeds_volume = seeds_target.imread()
        perm = np.random.RandomState(1234).permutation(np.max(seeds_volume)+1)
        seg_volume = seg_target.imread()
        figure = plt.figure(figsize=(self.size_x_inches, self.size_y_inches))
        ax = figure.add_subplot(1, 1, 1)
        sm = ScalarMappable(cmap='jet')
        sm.set_array(np.unique(seeds_volume))
        sm.autoscale()
        prob_sm = ScalarMappable(cmap='jet')
        prob_sm.set_array(np.linspace(0, 255, 256))
        prob_sm.autoscale()
        offset = self.first_frame
        end = self.end_frame if self.end_frame != 0 else image_volume.shape[0]
        
        def init_fig():
            ax.clear()
            ax.set_ylim(self.volume.height, 0)
            ax.set_xlim(0, self.volume.width)
        
        def animate(i1):
            i = i1 - offset
            rh_logger.logger.report_event("Processing frame %d" % i)
            ax.imshow(image_volume[i], cmap='gray')
            sv = sm.to_rgba(perm[seeds_volume[i]])
            sv[seeds_volume[i]==0, 3] = 0
            ax.imshow(sv)
            sv = sm.to_rgba(perm[seg_volume[i]])
            sv[seg_volume[i] == 0, 3] = 0
            sv[:, :, 3] *= .4
            ax.imshow(sv)
            prob = prob_sm.to_rgba(prob_volume[i])
            prob[:, :, 3] *= .4
            prob[prob_volume[i] < 128, 3] = 0
            ax.imshow(prob)
        
        anim = animation.FuncAnimation(figure, animate, init_func=init_fig,
                                       frames = end-offset,
                                       interval=1)
        plt.close(figure)
        anim.save(self.output().path, fps=1, extra_args=['-vcodec', 'libx264'])


class VisualizeTask(VisualizeTaskMixin,
                    VisualizeRunMixin,
                    RequiresMixin,
                    RunMixin,
                    luigi.Task):
    task_namespace="ariadne_microns_pipeline"

class PipelineVisualizeTask(luigi.Task):
    '''Visualize the results within a pipeline directory'''
    
    experiment=luigi.Parameter(
        description="The name of the butterfly experiment")
    sample=luigi.Parameter(
        description="The name of the butterfly sample")
    dataset=luigi.Parameter(
        default="sem",
        description="The name of the butterfly dataset")
    channel=luigi.Parameter(
        default="raw",
        description="The name of the butterfly image channel, e.g. \"raw\"")
    directory=luigi.Parameter(
        description='The directory produced by a pipeline, e.g. '
        '".../ac3/unknown/sem/raw/26/26/1"')
    output_location=luigi.Parameter(
        description="The location for the movie file.")
    url=luigi.Parameter(
        default="http://localhost:2001/api",
        description="The URL of the butterfly server")
    #
    # Optional parameters
    #
    use_neuroproof=luigi.BoolParameter(
        description="Use the neuroproof segmentation")
    size_x_inches=luigi.FloatParameter(
        default=6,
        description="The target size of the output in the X direction "
        "in inches")
    size_y_inches=luigi.FloatParameter(
        default=6,
        description="The target size of the output in the Y direction "
        "in inches")
    first_frame=luigi.IntParameter(
        default=0,
        description="Start at this frame instead of frame # 0")
    end_frame=luigi.IntParameter(
        default=0,
        description="Stop the movie at this frame. Default=do all")
    
    task_namespace="ariadne_microns_pipeline"
    
    def requires(self):
        membrane_done = filter(lambda _: _.endswith("membrane.done"),
                               os.listdir(self.directory))
        assert len(membrane_done) == 1, \
               "Malformed directory: no membrane.done file"
        md_file = os.path.join(self.directory, membrane_done[0])
        md = json.load(open(md_file, "r"))
        dimensions = md["dimensions"]
        volume = Volume(md["x"], md["y"], md["z"], 
                        dimensions[2], dimensions[1], dimensions[0])
    
        image_location = DatasetLocation(
            [self.directory], 
            "image", 
            "{x:09d}_{y:09d}_{z:09d}_image")
        dfb_task = DownloadFromButterflyTask(
            experiment=self.experiment,
            sample=self.sample,
            dataset=self.dataset,
            channel=self.channel,
            volume=volume,
            destination=image_location,
            url=self.url)
        if self.use_neuroproof:
            seg_location = DatasetLocation(
                [self.directory], "neuroproof", 
                "{x:09d}_{y:09d}_{z:09d}_neuroproof")
        else:
            seg_location=DatasetLocation(
                    [self.directory], "segmentation", 
                    "{x:09d}_{y:09d}_{z:09d}_segmentation")
        vt = VisualizeTask(
            volume=volume,
            image_location=image_location,
            prob_location=DatasetLocation(
                [self.directory], "membrane", 
                "{x:09d}_{y:09d}_{z:09d}_membrane-membrane"),
            seeds_location=DatasetLocation(
                [self.directory], "seeds", "{x:09d}_{y:09d}_{z:09d}_seeds"),
            seg_location=seg_location,
            output_location=self.output_location,
            size_x_inches=self.size_x_inches,
            size_y_inches=self.size_y_inches,
            first_frame=self.first_frame,
            end_frame=self.end_frame)
        vt.set_requirement(dfb_task)
        yield vt
