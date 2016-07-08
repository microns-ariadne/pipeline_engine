import luigi
from ..targets.classifier_target\
     import PixelClassifierTarget
from ..targets.factory import TargetFactory
from ..parameters import VolumeParameter, DatasetLocationParameter
from ..parameters import Volume, DatasetLocation
from .utilities import RequiresMixin


class ClassifyTaskMixin:
    '''Classify pixels

    This class has a pixel classifier as its output. The file containing the
    pixel classifier should be a pickled instance of a subclass of
    PixelClassifier. The instance should have the classification parameters
    pre-loaded or bound internally.
    '''
    
    classifier_path=luigi.Parameter(
        description="Location of the pickled classifier file")
    volume = VolumeParameter(
        description="The volume of the input image")
    image_location = DatasetLocationParameter(
        description="The location of the input image volume")
    prob_roots = luigi.ListParameter(
        description="The paths of the sharded root directories "
        "for the probability map files.")
    class_names = luigi.DictParameter(
        description="The class names to save. The classifier is interrogated "
        "for the probability map for each class and only the named classes "
        "are saved. Keys are the names of the classes and values are the "
        "names of the datasets to be created.")
    pattern = luigi.Parameter(
        description="A filename pattern for str.format(). See one of the "
        "volume targets for details.")
    
    def input(self):
        yield self.get_classifier_target()
        yield TargetFactory().get_volume_target(self.image_location,
                                                self.volume)
    def get_classifier_target(self):
        if not hasattr(self, "__classifier_target"):
            self.__classifier_target = \
                PixelClassifierTarget(self.classifier_path)
        return self.__classifier_target
    
    def get_classifier(self):
        return self.get_classifier_target().classifier
    
    @property
    def out_x(self):
        '''The global x coordinate of the output volume'''
        return self.volume.x + self.get_classifier().get_x_pad()
    
    @property
    def out_width(self):
        '''The width of the output volume'''
        return self.volume.width - 2 * self.get_classifier().get_x_pad()
    
    @property
    def out_y(self):
        '''The global y coordinate of the output volume'''
        return self.volume.y + self.get_classifier().get_y_pad()
    
    @property
    def out_height(self):
        '''The height of the output volume'''
        return self.volume.height - 2 * self.get_classifier().get_y_pad()
    
    @property
    def out_z(self):
        '''The global Z coordinate of the output volume'''
        return self.volume.z + self.get_classifier().get_z_pad()
    
    @property
    def out_depth(self):
        '''The # of planes in the output volume'''
        return self.volume.depth - 2 * self.get_classifier().get_z_pad()
    
    @property
    def output_volume(self):
        '''The volume of the output'''
        return Volume(self.out_x, self.out_y, self.out_z,
                      self.volume.width, self.volume.height, self.volume.depth)
    
    def output(self):
        volume = Volume(self.out_x, self.out_y, self.out_z,
                        self.out_width, self.out_height, self.out_depth)
        return TargetFactory().get_multivolume_target(
            roots=self.prob_roots,
            channels=self.class_names.values(),
            pattern=self.pattern,
            volume=volume)
    

class ClassifyRunMixin:
    
    def ariadne_run(self):
        '''Run the classifier on the input volume to produce the outputs'''
        classifier_target, image_target = list(self.input())
        classifier = classifier_target.classifier
        image = image_target.imread()
        probs = classifier.classify(image)
        
        output_target = self.output()
        for class_name, dataset_name in self.class_names.items():
            output_target.imwrite(dataset_name, probs[class_name])

class ClassifyTask(ClassifyTaskMixin, ClassifyRunMixin, 
                   RequiresMixin, luigi.Task):
    '''Classify an image, producing one or more probability maps
    
    '''
    task_namespace = "ariadne_microns_pipeline"

    def run(self):
        self.ariadne_run()


class ClassifyShimTask(RequiresMixin, luigi.Task):
    '''A shim task to return one of the ClassifyTask's probability maps
    
    The multivolume input parameters can be deduced from the classifier
    but appear to be necessary for defining the shim task as a unit.
    '''
    
    mv_roots = luigi.ListParameter(
        description="The paths of the sharded root directories "
        "for the multivolume produced by the classifier.")
    mv_class_names = luigi.DictParameter(
        description="The multivolume class name mapping")
    mv_pattern = luigi.Parameter(
        description="The multivolume's filename pattern")
    volume = VolumeParameter(
        description="The output volume of the classifier")
    dataset_name = luigi.Parameter(
        description="The name of one of the ClassifyTask outputs")
    
    @staticmethod
    def make_shim(classify_task, dataset_name):
        shim = ClassifyShimTask(mv_roots=classify_task.prob_roots,
                                 mv_class_names=classify_task.class_names,
                                 mv_pattern=classify_task.pattern,
                                 volume=classify_task.output_volume,
                                 dataset_name=dataset_name)
        shim.set_requirement(classify_task)
        return shim
    
    def input(self):
        yield self.requirements[0].output()
    
    def output(self):
        mv = self.input().next()
        return mv.get_channel(self.dataset_name)
    
    @property
    def output_volume(self):
        return self.requirements[0].output_volume
    
    @property
    def output_location(self):
        channel = self.requirements[0].output().get_channel(self.dataset_name)
        return DatasetLocation(channel.paths,
                               self.dataset_name,
                               channel.pattern)
