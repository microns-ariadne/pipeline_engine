## Installation

The Python Luigi framework installs via standardized mechanisms, e.g.
'pip install .' from the root directory. Some of the tasks run binaries
as subprocesses. These are built with the makefile in the root directory.
This makefile builds a number of dependencies with the CILKplus compiler
and it builds the compiler itself.

You should create a tools directory to hold these dependencies. The following
libraries should be installed on your system along with their include files:

libbz2
libc6-dev-i386
libfftw
libhdf5
libjpeg
libpng
libtiff
libz

Tools required to compile:

cmake
flex
bison
gcc
make

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
    skeletonization:
        home-dir: <directory containing the skeletonization binary>
        ld_library_path:
            - <path to OpenCV libraries>
            - <path to HDF5 libraries>
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

### Luigi daemon configuration

The Luigi daemon (luigid) manages resources as part of scheduling tasks.
The Ariadne Microns pipeline tasks keep track of the number of cores they use,
the number of GPUs they use and the memory (the highwater mark of the resident
set size) that they use. These are controlled by the luigi

## Running

To get help on the pipeline task, run it from Luigi:

    luigi --module ariadne_microns_pipeline.pipelines ariadne_microns_pipeline.PipelineTask --help

The KerasClassifier needs to run using a worker pool in order to amortize
the cost of compiling the Theano function. To do this, start the broker:

    > microns-ipc-broker

then start one worker per GPU:

    > microns-ipc-worker

## Output
The pipeline produces quite a few files. The following are some of the
most important because they provide indexes into the files that were created.
The files are optional and can be specified on the Luigi command-line
with the following switches:

* **--connectivity-graph-location** [connectivity-graph.json file format]:
the connectivity graph is a .json file
that is an index of the neuron IDs in the segmentation of the volume. Each
block's segmentation labels each voxel with local IDs and the connectivity
graph supplies the translation between local and global IDs so that a neuron
that spans blocks can be identified in the individual blocks.

* **--synapse-connection-location** [synapse-connections.json file format]:
this is a .json file that contains
the position of every synapse along with its pre- and post-synaptic partners.

* **--index-file-location** [index.json file format]:
this is a .json file that contains a list of subvolume coordinates and
dataset locations for every channel of data produced by the pipeline
(e.g. "image", "membrane" or "neuroproof").

### JSON objects

There are two critical JSON objects encoded in the JSON files and these
correspond to Python objects that are used to reference voxel data. These
are the Volume and DatasetLocation and they represent the location of the
subvolume within the Butterfly volume and the location of the dataset on
disk. You can instantiate these using the following code (assuming you
have two JSON dictionary objects named *volume_dict* and *dataset_location_dict*):

    from ariadne_microns_pipeline.parameters import Volume, DatasetLocation
    volume = Volume(**volume_dict)
    dataset_location = DatasetLocation(**dataset_location_dict)

You can then create a PngVolumeTarget to load the voxel data from these
like so:

    from ariadne_microns_pipeline.targets.factory import TargetFactory
    tgt = TargetFactory().get_volume_target(dataset_location, volume)
    voxels = tgt.imread()

#### Volume object

The volume dictionary has the following keys:

* **x** - the x offset of the subvolume
* **y** - the y offset of the subvolume
* **z** - the z offset of the subvolume
* **width** - the width of the subvolume (x direction)
* **height** - the height of the subvolume (y direction)
* **depth** - the depth of the subvolume (z direction)

#### Dataset location object

The dataset location should ideally only be used to initialize a PngVolumeTarget.
Please see the code for that class to find out how it is used. The keys to
a dataset location:

* **roots** - This is a list of root directories for the volume. The volume
can be sharded across disk spindles to increase I/O throughput and in this case,
multiple roots will be specified, one per spindle.

* **dataset_name** - this is the name of the dataset, for instance, "neuroproof"
or "image". The .png files are stored in a subdirectory with the name given
by *dataset_name* under the path given by one of the *roots*.

* **pattern** - this is a pattern supplied to *str.format(x=x, y=y, z=z)* that
is used to name the .png files and the .done file that is written once the
volume has successfully been written to disk. For instance, the
.png file at x=20, y=20, z=1 with pattern,
`{x:09d}_{y:09d}_{z:09d}_neuroproof`, is written to the file,
`000000020_000000020_000000001_neuroproof.png`.

#### volume.done file format

Each subvolume has an accompanying .done file, written to
`roots[0]+"/"+pattern.format(x=x, y=y, z=z)+".done"`. This is a .json file
containing a dictionary with the following keys:

* **dimensions** - a 3-tuple with the depth (z), height (y) and width (x) in
that order.
* **dtype** - a string that can be passed to `numpy.dtype()` to get the
datatype of the voxel data.
* **x** - the x-origin of the subvolume
* **y** - the y-origin of the subvolume
* **z** - the z-origin of the subvolume
* **filenames** - a list of the .png files containing the volume data, arranged
in ascending z-order.

### connectivity-graph.json file format

The connectivity graph JSON file is a dictionary composed of the following
keys:

* **count** (int): the number of neurons segmented

* **locations** (list): a list of two-tuples, one per subvolume, giving the
volume and dataset location of the Neuroproof segmentation of that subvolume
(see [JSON objects](#json-objects)).

* **volumes** (list): a list of two-tuples, one per subvolume. The first
element of the two-tuple is a subvolume and the second is a list of two-tuples
composed of a local neuron-id and a global neuron-id. The local neuron-id is
the one that is used in the labeling of the subvolume's *neuroproof* dataset and
the global neuron-id is the neuron's neuron-id after stitching subvolumes
together. The convenience class,
`ariadne_microns_pipeline.tasks.connected_components.ConnectivityGraph`, can
be used to perform the translation. For instance, given a volume, `volume`
and the connectivity graph .json file location, `cg_path`:

    from ariadne_microns_pipeline.tasks.connected_components import ConnectivityGraph
    cg = ConnectivityGraph.load(cg_path)
    tgt = cg.get_tgt(volume)
    segmentation = cg.convert(tgt.imread(), volume)

produces the segmentation using the global neuron-ids.

* **joins** (list) - this is a list of three-tuples. The first two elements
in the three-tuple are two overlapping subvolumes (see [Volume object](#volume-object)).
The last element is the path to the connected-components JSON file that
contains the correspondences between matching neuron-ids in the first and
second volumes. The connected-components file is for internal use and
not documented here.

### synapse-connections.json file format

The synapse-connections JSON file contains the basic information for
every synapse in the volume. The file is arranged into five lists giving
the neuron-ids for the presynaptic partners, the postsynaptic partners and
the x, y and z locations of the synapse centers. The file contains one
dictionary with the following keys

* **neuron_1** (list)- a list of the global neuron-ids of the presynaptic
partner of each synapse
* **neuron_2** (list) - a list of the global neuron-ids of the postsynaptic
partner of each synapse
* **synapse_center** (dictionary) - a dictionary with the following keys:
  * **x** (list) - a list of the x-coordinate of the centroid of each synapse
  * **y** (list) - a list of the y-coordinate of the centroid of each synapse
  * **z** (list) - a list of the z-coordinate of the centroid of each synapse

### index.json file format

The index file is a global index into the subvolume data for each channel
produced by the analysis. The file contains a single dictionary object.
The keys of this object are the names of the channels and the values are
lists of two-tuples of volume (see [Volume object](#volume-object)) and dataset location
(see [Dataset location object](#dataset-location-object)).

The microns-volume script [ariadne_microns_pipeline/scripts/microns_volume.py]
has an example of how to use the index.json file to read an arbitrary
subvolume of an arbitrary channel.
