'''Create a simple storage plan for a dataset

Ariadne tasks that write a data volume need a storage plan to tell them
how to do it. This writes a storage plan file that directs the task to
write a volume simply to a single .tif stack.

You can use the --storage-plan switch to specify the file output by this
application.
'''

import argparse

from ariadne_microns_pipeline.targets.volume_target \
     import write_simple_storage_plan
from ariadne_microns_pipeline.parameters import Volume

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--x", type=int,
        help="The x offset of the volume to be written")
    parser.add_argument(
        "--y", type=int,
        help="The y offset of the volume to be written")
    parser.add_argument(
        "--z", type=int,
        help="The z offset of the volume to be written")
    parser.add_argument(
        "--width", type=int, 
        help="The size of the volume in the x direction")
    parser.add_argument(
        "--height", type=int,
        help="The size of the volume in the y direction")
    parser.add_argument(
        "--depth", type=int,
        help="The size of the volume in the z direction")
    parser.add_argument(
        "--plan",
        help="The name of the .plan file to be written")
    parser.add_argument(
        "--dataset",
        help="The name of the .tif file that is going to be the target "
        "of the task")
    parser.add_argument(
        "--datatype",
        help="The Numpy-style data type, e.g. \"uint8\"")
    parser.add_argument(
        "--dataset-name",
        help="The name of the dataset type, e.g. \"image\"")
    return parser.parse_args()

def main():
    args = parse_args()
    volume = Volume(args.x, args.y, args.z, args.width, args.height, args.depth)
    write_simple_storage_plan(args.plan, args.dataset, volume, 
                              args.dataset_name, args.datatype)
