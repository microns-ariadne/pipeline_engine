'''Generate the synapse connectivity ground truth'''

import json
import luigi
import numpy as np
import os
import rh_logger
from scipy.sparse import coo_matrix

from ..parameters import Volume, VolumeParameter
from ..parameters import DatasetLocation, DatasetLocationParameter
from ..parameters import EMPTY_DATASET_LOCATION
from ..targets.butterfly_target import ButterflyChannelTarget
from ..tasks.factory import AMTaskFactory
from ..tasks import Dimensionality
from ..tasks.utilities import RunMixin

SEG_DATASET = "segmentation"
SYN_DATASET = "synapses"
SYN_SEG_DATASET = "synapse-segmentation"
ANALYSIS_FILE = "analysis.json"

'''The documentation for the JSON output'''
JSON_DOC = '''
        graph: a dictionary of 3 vectors
           a: the object number of the first neurite
           b: the object number of the second neurite
        
        paths: a dictionary of 3 vectors - this structure enumerates
               all paths from one neuron through a second neuron to a third
               neuron.
'''

class SynapseGtPipelineTaskMixin:
    
    experiment = luigi.Parameter(
        description="The butterfly experiment")
    sample = luigi.Parameter(
        description="The ID of the biological sample")
    dataset = luigi.Parameter(
        description="The name of the volume that was imaged")
    neuron_channel = luigi.Parameter(
        description="The name of the neuron segmentation channel")
    synapse_channel = luigi.Parameter(
        description="The name of the synapse channel")
    volume=VolumeParameter(
        description="The volume to be analyzed")
    destination = luigi.Parameter(
        description="The directory to hold the analysis data")
    
    # Optional parameters
    block_width = luigi.IntParameter(
        description="Width of one of the processing blocks",
        default=2048)
    block_height = luigi.IntParameter(
        description="Height of one of the processing blocks",
        default=2048)
    block_depth = luigi.IntParameter(
        description="Number of planes in a processing block",
        default=2048)
    url = luigi.Parameter(
        default="http://localhost:2001/api",
        description="The URL of the butterfly host")
    
    def requires(self):
        self.compute_requirements()
        return self.requirements
    
    def inputs(self):
        for task in self.analysis_tasks.flatten():
            yield task.output()

    def output(self):
        return luigi.LocalTarget(
            os.path.join(self.destination, ANALYSIS_FILE))
        
    def compute_extents(self):
        '''Compute various block boundaries and padding
        
        self.{x, y, z}{0, 1} - the start and end extents in the x, y & z dirs

        self.n_{x, y, z} - the number of blocks in the x, y and z dirs

        self.{x, y, z}s - the starts of each block (+1 at the end so that
        self.xs[n], self.xs[n+1] are the start and ends of block n)
        '''
        butterfly = ButterflyChannelTarget(
            self.experiment, self.sample, self.dataset, self.neuron_channel, 
            self.url)
        self.x1 = min(butterfly.x_extent, 
                      self.volume.x + self.volume.width)
        self.y1 = min(butterfly.y_extent,
                      self.volume.y + self.volume.height)
        self.z1 = min(butterfly.z_extent,
                      self.volume.z + self.volume.depth)
        self.x0 = self.volume.x
        self.y0 = self.volume.y
        self.z0 = self.volume.z
        self.useable_width = self.x1 - self.x0
        self.useable_height = self.y1 - self.y0
        self.useable_depth = self.z1 - self.z0
        #
        # Compute equi-sized blocks (as much as possible)
        #
        self.n_x = int((self.useable_width-1) / self.block_width) + 1
        self.n_y = int((self.useable_height-1) / self.block_height) + 1
        self.n_z = int((self.useable_depth-1) / self.block_depth) + 1
        self.xs = np.linspace(self.x0, self.x1, self.n_x + 1).astype(int)
        self.ys = np.linspace(self.y0, self.y1, self.n_y + 1).astype(int)
        self.zs = np.linspace(self.z0, self.z1, self.n_z + 1).astype(int)

    def get_dirs(self, x, y, z):
        '''Return a directory suited for storing a file with the given offset
        
        Create a hierarchy of directories in order to limit the number
        of files in any one directory.
        '''
        return [os.path.join(self.destination,
                             self.experiment,
                             self.sample,
                             self.dataset,
                             str(x),
                             str(y),
                             str(z))]
    
    def get_pattern(self, dataset_name):
        return "{x:09d}_{y:09d}_{z:09d}_"+dataset_name
    
    def get_dataset_location(self, volume, dataset_name):
        return DatasetLocation(self.get_dirs(volume.x, volume.y, volume.z),
                               dataset_name,
                               self.get_pattern(dataset_name))
    
    def compute_requirements(self):
        '''Make the task dependency graph'''
        if not hasattr(self, "requirements"):
            self.factory = AMTaskFactory()
            self.compute_extents()
            #
            # Step 1: download neuron segmentation
            #
            self.make_download_segmentation_tasks()
            #
            # Step 2: download synapse labels
            #
            self.make_download_synapse_tasks()
            #
            # Step 3: do connected components on synapses
            #
            self.make_cc_tasks()
            #
            # Step 4: analyze the blocks for connections
            #
            self.make_analysis_tasks()
            #
            # Make the requirements
            #
            self.requirements = self.analysis_tasks.flatten().tolist()
    
    def make_download_segmentation_tasks(self):
        '''Make the tasks that download neuron segmentations from bfly'''
        rh_logger.logger.report_event("Making tasks to fetch segmentation gt")
        self.seg_butterfly_tasks = np.zeros(
            (self.n_z, self.n_y, self.n_x), object)
        for zi in range(self.n_z):
            z0, z1 = self.zs[zi], self.zs[zi+1]
            for yi in range(self.n_y):
                y0 = self.ys[yi]
                y1 = self.ys[yi+1]
                for xi in range(self.n_x):
                    x0 = self.xs[xi]
                    x1 = self.xs[xi+1]
                    volume = Volume(x0, y0, z0, x1-x0, y1-y0, z1-z0)
                    location =self.get_dataset_location(volume, SEG_DATASET)
                    task = self.factory.gen_get_volume_task(
                        experiment=self.experiment,
                        sample=self.sample,
                        dataset=self.dataset,
                        channel=self.neuron_channel,
                        url=self.url,
                        volume=volume,
                        location=location)
                    self.seg_butterfly_tasks[zi, yi, xi] = task
    
    def make_download_synapse_tasks(self):
        '''Make the tasks that download synapse labels from bfly'''
        rh_logger.logger.report_event("Making tasks to fetch synapse labels")
        self.syn_butterfly_tasks = np.zeros(
            (self.n_z, self.n_y, self.n_x), object)
        for zi in range(self.n_z):
            z0, z1 = self.zs[zi], self.zs[zi+1]
            for yi in range(self.n_y):
                y0 = self.ys[yi]
                y1 = self.ys[yi+1]
                for xi in range(self.n_x):
                    x0 = self.xs[xi]
                    x1 = self.xs[xi+1]
                    volume = Volume(x0, y0, z0, x1-x0, y1-y0, z1-z0)
                    location =self.get_dataset_location(volume, SYN_DATASET)
                    task = self.factory.gen_get_volume_task(
                        experiment=self.experiment,
                        sample=self.sample,
                        dataset=self.dataset,
                        channel=self.synapse_channel,
                        url=self.url,
                        volume=volume,
                        location=location)
                    self.syn_butterfly_tasks[zi, yi, xi] = task
    
    def make_cc_tasks(self):
        '''Make tasks that perform connected components on synapse labels'''
        rh_logger.logger.report_event("Making tasks to segment synapses")
        self.cc_tasks = np.zeros(
            (self.n_z, self.n_y, self.n_x), object)
        for zi in range(self.n_z):
            for yi in range(self.n_y):
                for xi in range(self.n_x):
                    syn_task = self.syn_butterfly_tasks[zi, yi, xi]
                    syn_location = syn_task.output().dataset_location
                    volume = syn_task.volume
                    syn_seg_location = self.get_dataset_location(
                        volume, SYN_SEG_DATASET)
                    #
                    # Parameters for segmentation:
                    #    No mask
                    #    3D connected components
                    #    threshold = everything other than zero
                    #
                    task = self.factory.gen_cc_segmentation_task(
                        volume=volume,
                        prob_location=syn_location,
                        mask_location=EMPTY_DATASET_LOCATION,
                        seg_location=syn_seg_location,
                        threshold=0,
                        dimensionality=Dimensionality.D3,
                        fg_is_higher=True)
                    self.cc_tasks[zi, yi, xi] = task
                    task.set_requirement(syn_task)
    
    def make_analysis_tasks(self):
        '''Analyze the connections at the level of neurites and synapses
        
        Utilize the connected components task to match neurites against
        segments. This produces an array matching neurite to segment.
        
        We can then use this array to build the graph of neurite to neurite
        by joining neurites connected to the same synapse.
        '''
        rh_logger.logger.report_event("Making tasks to find connections")
        self.analysis_tasks = np.zeros(
            (self.n_z, self.n_y, self.n_x), object)
        for zi in range(self.n_z):
            for yi in range(self.n_y):
                for xi in range(self.n_x):
                    seg_task = self.seg_butterfly_tasks[zi, yi, xi]
                    syn_task = self.cc_tasks[zi, yi, xi]
                    volume = seg_task.volume
                    location = os.path.join(
                         self.get_dirs(volume.x, volume.y, volume.z)[0],
                         ANALYSIS_FILE)
                    task = self.factory.gen_connected_components_task(
                        volume, seg_task.output().dataset_location,
                        volume, syn_task.output().dataset_location,
                        volume, location)
                    task.set_requirement(seg_task)
                    task.set_requirement(syn_task)
                    self.analysis_tasks[zi, yi, xi] = task

class SynapseGtPipelineRunMixin:
    
    def ariadne_run(self):
        '''Combine all JSON files into a single output
        
        The final analysis has these components
        
        ''' + JSON_DOC
        # ------------------------------------
        #
        #      Graph computation
        #
        #-------------------------------------
        #
        # Get the neurite to synapse associations
        #
        neurite = []
        synapse = []
        overlap = []
        offset = 0
        for target in self.inputs():
            d = json.load(target.open("r"))
            connections = np.array(d["connections"])
            neurite.append(connections[:, 0])
            synapse.append(connections[:, 1] + offset)
            overlap.append(np.array(d["counts"]))
            offset += np.max(connections[:, 1])
        #
        # Create a sparse array of counts of overlapping voxels in order
        # to aggregate multiple possible instances of same neuron / synapse
        # combo
        #
        neurite, synapse, overlap = map(np.hstack, [neurite, synapse, overlap])
        matrix = coo_matrix((overlap, (neurite, synapse)))
        matrix.sum_duplicates()
        neurite, synapse = matrix.nonzero()
        overlap = matrix.tocsr()[neurite, synapse].getA1()
        #
        # sort by -overlap and synapse to get neurites per synapse with
        #      neurites with most overlap first.
        #
        order = np.lexsort((-overlap, synapse))
        neurite, synapse, overlap = \
            [_[order] for _ in neurite, synapse, overlap]
        first = np.hstack([[True], synapse[:-1] != synapse[1:], [True]])
        indices = np.where(first)[0]
        counts = indices[1:] - indices[:-1]
        indices = indices[:-1]
        #
        # Potentially, there are some synapses with only a single neurite
        # overlapping. Get rid of that corner case.
        #
        mask = counts == 1
        synapses_with_too_few_neurites = synapse[indices[mask]]
        indices, counts = [_[~mask] for _ in indices, counts]
        #
        # And finally, a and b are the first and second neurite in the
        # indexed array
        #
        a = neurite[indices]
        b = neurite[indices + 1]
        synapse = synapse[indices]
        #---------------------------------------------
        #
        #     A -> B -> C Path computation
        #
        #---------------------------------------------
        #
        # Get the unique a/b combos.
        #
        matrix = coo_matrix((np.ones(len(a)*2), 
                             (np.hstack((a, b)), np.hstack((b, a)))))
        matrix.sum_duplicates()
        #
        # In Einstein notation, we want
        #
        #   ab
        # m    m
        #        bc
        #
        #          ac
        # to get m     which is the # of ways to get from a to c
        #
        # That's the dot product of the matrix with itself. How handy.
        #
        # Remember to erase the diagonals which are self -> something -> self
        matrix = np.dot(matrix, matrix)
        aa, cc = matrix.nonzero()
        mask = aa != cc
        aa, cc = aa[mask], cc[mask]
        #
        # Put together the dictionary
        #
        # To do: maybe the counts of ways to get from A->B is interesting?
        #
        result = dict(
            graph = dict(a=a.tolist(), 
                         b=b.tolist()),
            paths = tuple([_.tolist() for _ in aa, cc]))
        #
        # Write it out
        #
        json.dump(result, self.output().open("w"))

class SynapseGtPipelineTask(SynapseGtPipelineTaskMixin,
                            SynapseGtPipelineRunMixin,
                            RunMixin,
                            luigi.Task):
    '''Pipeline to compute neurite/synapse connections in the ground truth
    
    This pipeline creates a json file in the destination that has the
    following structure:
    
    '''+JSON_DOC
    
    task_namespace = "ariadne_microns_pipeline"
