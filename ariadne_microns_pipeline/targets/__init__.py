'''ariadne_microns_pipeline.targets package

The targets package has the different kinds of targets for the Ariadne / Microns
pipeline system.
'''

from volume_target import SrcVolumeTarget, DestVolumeReader

all = [SrcVolumeTarget, DestVolumeReader]
