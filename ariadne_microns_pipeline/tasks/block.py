'''The BlockTask constructs an output block from input blocks

'''

import json
import luigi
from ariadne_microns_pipeline.targets.hdf5_target import HDF5VolumeTarget

class BlockTaskMixin:
    '''The block task constructs creates a block of data from volumes'''

    output_volume_path = luigi.Parameter()
    output_volume_dataset_path = luigi.Parameter()
    input_volumes = luigi.ListParameter(
        description="A list of the input volumes to be combined to "
        "produce the output volume. The format is a JSON list of JSON "
        " two-tuples of path and dataset name, e.g. "
        '[["/tmp/foo.h5", "img"], ["/tmp/bar.h5", "img"]]')
    x = luigi.IntParameter(
        description="The X offset of the volume to be generated")
    y = luigi.IntParameter(
        description="The Y offset of the volume to be generated")
    z = luigi.IntParameter(
        description="The Z offset of the volume to be generated")
    width = luigi.IntParameter(
        description="The width of the volume to be generated")
    height = luigi.IntParameter(
        description="The height of the volume to be generated")
    depth = luigi.IntParameter(
        description="The depth of the volume to be generated")
    
    def input(self):
        '''Return the volumes to be assembled'''
        for path, dataset in json.loads(self.input_volumes):
            yield HDF5VolumeTarget(path, dataset)
    
    def output(self):
        '''Return the volume target that will be written'''
        return HDF5VolumeTarget(output_volume_path, output_volume_dataset_path)


class BlockTaskRunMixin:
    '''Combine the inputs to produce the output
    
    The algorithm is simple - take the inputs in turn, find the intersection
    with the output volume. The output volume datatype is the same as that
    of the first input volume.
    '''
    
    def ariadne_run(self):
        '''Loop through the inputs, compositing their volumes on the output'''
        
        # TO_DO: a clever implementation would optimize the block organization
        #        of the output volume dataset by looking at the average
        #        size of the blocks being written.
        output_volume = self.output()
        first = True
        for input_volume in self.input():
            #
            # Compute the portion of the input volume that overlaps
            # with the requested output volume.
            #
            x0 = max(self.x, input_volume.x)
            x1 = min(self.x + self.width, input_volume.x + input_volume.width)
            if x0 >= x1:
                continue
            y0 = max(self.y, input_volume.y)
            y1 = min(self.y + self.height, input_volume.y + input_volue.height)
            if y0 >= y1:
                continue
            z0 = max(self.z, input_volume.z)
            z1 = min(self.z + self.depth, input_volume.z + input_volume.depth)
            if z0 >= z1:
                continue
            subvolume = input_volume.imread_part(x0, x1, y0, y1, z0, z1)
            if first:
                first = False
                output_volume.create_volume(
                    self.x, self.y. self.z, 
                    self.width, self.height, self.depth,
                    subvolume.dtype)
                
            output_volume.imwrite_part(subvolume, x0, y0, z0)


class BlockTask(BlockTaskMixin,
                BlockTaskRunMixin,
                luigi.Task):
    '''Copy blocks from the inputs to produce the output'''
    
    def run(self):
        self.ariadne_run()