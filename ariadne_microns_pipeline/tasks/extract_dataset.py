import luigi
from ariadne_microns_pipeline.targets.hdf5_target import HDF5FileTarget

class ExtractDatasetTask(luigi.task):
    '''Extract a dataset target from an HDF5FileTarget
    
    This task does very little work - it provides the Luigi infrastructure
    needed to get a single dataset from an HDF5 file that might have many.
    '''
    
    path = luigi.Parameter(
        description="The path to the HDF5 file")
    dataset_path = luigi.Parameter(
        description="The path to the dataset within the HDF5 file")
    
    def input(self):
        return HDF5FileTarget(self.path)
    
    def output(self):
        return HDF5FileTarget(self.path).get_subtarget(self.dataset_path)
    
    def run(self):
        pass
