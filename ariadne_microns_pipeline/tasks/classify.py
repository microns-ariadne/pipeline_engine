import luigi
from ariadne_microns_pipeline.targets.classifier_target\
     import PixelClassifierTarget


class PixelClassifierTask(luigi.ExternalTask):
    '''Make a pixel classifier

    This class has a pixel classifier as its output. The file containing the
    pixel classifier should be a pickled instance of a subclass of
    PixelClassifier. The instance should have the classification parameters
    pre-loaded or bound internally.
    '''
    
    classifier_path=luigi.Parameter(
        description="Location of the pickled classifier file")
    
    def outputs(self):
        return PixelClassifierTarget(self.classifier_path)

all = [PixelClassifierTask]