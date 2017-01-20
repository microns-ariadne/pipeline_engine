'''Copy a subvolume into a .h5 file'''

import argparse
import h5py
import json
import numpy as np
import rh_logger

from ariadne_microns_pipeline.parameters import Volume, DatasetLocation
from ariadne_microns_pipeline.targets.factory import TargetFactory

def parse_args():
    parser = argparse.ArgumentParser(
        description="Copy a subvolume from a pipeline analysis to an HDF5 file")
    parser.add_argument("--index-file", required=True,
                        help="The index file generated by the pipeline, giving"
                        " the locations of the subvolumes")
    parser.add_argument("--dataset-name", required=True,
                        help="The name of the dataset, for instance "
                        "\"membrane\".")
    parser.add_argument("--x", type=int, required=True,
                        help="The x-offset into the volume of the subvolume "
                        "to be retrieved.")
    parser.add_argument("--y", type=int, required=True,
                        help="The y-offset into the volume of the subvolume "
                        "to be retrieved.")
    parser.add_argument("--z", type=int, required=True,
                        help="The z-offset into the volume of the subvolume "
                        "to be retrieved.")
    parser.add_argument("--width", required=True, type=int,
                        help="The dimension of the subvolume in the x "
                        "direction.")
    parser.add_argument("--height", required=True, type=int,
                        help="The dimension of the subvolume in the y "
                        "direction.")
    parser.add_argument("--depth",  required=True, type=int,
                        help="The dimension of the subvolume in the z "
                        "direction.")
    parser.add_argument("--output-location", required=True,
                        help="The name of the .h5 file to be generated")
    parser.add_argument("--output-dataset-name", default=False,
                        help="The name of the HDF5 dataset within the output "
                        "file")
    parser.add_argument("--chunks", default=None,
                        help="The target dataset's chunk size, for instance "
                        "4,1024,1024 for a chunk composed of four z-slices "
                        "of size 1024 x 1024.")
    parser.add_argument("--gzip", action="store_true",
                        help="Compress output dataset using GZIP")
    parser.add_argument("--datatype", default=None,
                        help="The Numpy data type of the output, e.g. uint8. "
                        "Defaults to the datatype of the input.")
    return parser.parse_args()

def main():
    rh_logger.logger.start_process("microns-volume", "starting", [])
    #
    # Deal with decanting the arguments
    #
    args = parse_args()
    index_file = args.index_file
    dataset_name = args.dataset_name
    x = args.x
    y = args.y
    z = args.z
    width = args.width
    height = args.height
    depth = args.depth
    output_location = args.output_location
    output_dataset_name = args.output_dataset_name or dataset_name
    chunks = args.chunks
    gzip = args.gzip
    datatype = args.datatype
    #
    # Get the index file
    #
    index = json.load(open(index_file))
    #
    # Open the HDF5 file
    #
    with h5py.File(output_location, "a") as fd:
        #
        # Get the coordinates of the volume
        # 
        x0 = x
        x1 = x + width
        y0 = y
        y1 = y + height
        z0 = z
        z1 = z + depth
        ds = None
        #
        # Loop through all datasets
        #
        tf = TargetFactory()
        for volume, location in index[dataset_name]:
            volume = Volume(**volume)
            #
            # Check for dataset completely out of range
            #
            if volume.x > x1 or volume.x1 <= x0:
                continue
            if volume.y > y1 or volume.y1 <= y0:
                continue
            if volume.z > z1 or volume.z1 <= z0:
                continue
            #
            # Get the volume target
            #
            location = DatasetLocation(**location)
            target = tf.get_volume_target(location, volume)
            #
            # Compute the cutout
            #
            vx0 = max(x0, volume.x)
            vx1 = min(x1, volume.x1)
            vy0 = max(y0, volume.y)
            vy1 = min(y1, volume.y1)
            vz0 = max(z0, volume.z)
            vz1 = min(z1, volume.z1)
            #
            # Read the cutout
            #
            cutout = target.imread_part(vx0, vy0, vz0, 
                                        vx1-vx0, vy1-vy0, vz1-vz0)
            if ds is None:
                #
                # Create the dataset now that we know the datatype of
                # the input.
                #
                if datatype is None:
                    datatype = cutout.dtype
                else:
                    datatype = getattr(np, datatype)
                kwds = {}
                if chunks is not None:
                    chunks = map(int, chunks.split(","))
                    kwds["chunks"] = chunks
                if gzip:
                    kwds["compression"] = "gzip"
                ds = fd.create_dataset(output_dataset_name,
                                       shape=(depth, height, width),
                                       dtype=datatype,
                                       **kwds)
            #
            # Write the cutout
            #
            ds[vz0-z0:vz1-z0, vy0-y0:vy1-y0, vx0-x0:vx1-x0] = cutout
    rh_logger.logger.end_process("exiting", rh_logger.ExitCode.success)
    
if __name__=="__main__":
    main()
    