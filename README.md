## Installation

The Python Luigi framework installs via standardized mechanisms, e.g.
'pip install .' from the root directory. Some of the tasks run binaries
as subprocesses. These are built with the makefile in the root directory.
This makefile builds a number of dependencies with the CILKplus compiler
and it builds the compiler itself.

You should create a tools directory to hold these dependencies. The following
libraries should be installed on your system along with their include files:

libbz2
libfftw
libhdf5
libjpeg
libpng
libtiff
libz

## Deployment

The tool locations are specified using rh_config. The .rh_config.yaml file
should have the following sections:

neuroproof:
    neuroproof_graph_predict: <location of neuroproof_graph_predict binary>
    ld_library_path:
        - <path to OpenCV libraries>
        - <path to Boost libraries>
        - <path to Vigra libraries>
        - <path to JSONCPP libraries>
        - <path to CilkPlus libraries>

Optionally, if you use the fc_dnn classifier, you should have a section
for it in .rh_config.yaml:

c_dnn:
   path: <path-to-comipiled-binary>/fc_dnn
   ld_library_path:
    - <path to OpenCV libraries>
   xy_pad: <padding needed for image>
   z_depth: <depth of NN in the Z direction>
   num_classes: <# of classes output by fc_dnn as .png files>
   membrane_class: <Zero-based index of membrane .png file>

If you want timings from Luigi, you will have to start luigid using the
same account that you use to run Luigi. Your `luigi.cfg` file should
be set up to save task history (see 
http://luigi.readthedocs.io/en/stable/configuration.html#scheduler and 
http://datapipelinearchitect.com/luigi-scheduler-history/).

You should add the following section to your .rh_config.yaml file:

luigid:
    db_connection=<sqlalchemy db connection>

## Running

To get help on the pipeline task, run it from Luigi:

    luigi --module ariadne_microns_pipeline.pipelines ariadne_microns_pipeline.PipelineTask --help


