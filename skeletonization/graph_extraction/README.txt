Skeletonization toolbox

Developed by Gergely Odor and Timothy Kaler

This toolbox computes the skeletons of 3D volumes (input: h5 segmentation data, output: an .swc file representing a skeleton for each object)

Note: For the skeletonization it is assumed that there are no pixelwise disconnected 3D volumes in the input

How to run the code:
./main -{s,d,f} [downsampling scale] [path of h5 files] [output_directory]
For arg 1 (-{s,d,f}) specify -s to compute and save skeletons, -d to read multiple files from a directory, -f read only one file
For arg 2 (downsampling scale) could be 1, 2 or 4.
For arg 3 (labeled_image_directory or file) specify a directory or file containing the h5 file(s) corresponding to the segmentation.
For arg 4 (output_directory) specify the directory where output files will be written, note that this directory must contain a subdirectory named SWC.

The h5 files in the directory must be of the form out*.h5
To set the dimension of the imagestack please edit graph_properties.h
To set the coarsening parameters (besides the downsampling scale) please edit graph_extraction_config.h
