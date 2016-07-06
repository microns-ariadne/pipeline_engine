'''The BlockTask constructs an output block from input blocks

'''

import json
import luigi
from ..targets.factory import TargetFactory
from ..parameters import DatasetLocationParameter, VolumeParameter
from ..parameters import MultiVolumeParameter

class BlockTaskMixin:
    '''The block task constructs creates a block of data from volumes'''

    output_location = DatasetLocationParameter(
        description="Location of volume to be created")
    output_volume = VolumeParameter(
        description="Volume to be extracted from input datasets")
    input_volumes = MultiVolumeParameter(
        description="The volumes that will be composited to form the output "
        "volume.")
    def input(self):
        '''Return the volumes to be assembled'''
        tf = TargetFactory()
        for d in self.input_volumes:
            yield tf.get_volume_target(d["location"], d["volume"])
    
    def output(self):
        '''Return the volume target that will be written'''
        return tf.get_volume_target(self.output_location, self.output_volume)


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
                output_volume.create_volume(subvolume.dtype)
                
            output_volume.imwrite_part(subvolume, x0, y0, z0)


class BlockTask(BlockTaskMixin,
                BlockTaskRunMixin,
                luigi.Task):
    '''Copy blocks from the inputs to produce the output'''
    
    def run(self):
        self.ariadne_run()