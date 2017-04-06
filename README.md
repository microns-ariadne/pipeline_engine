## Installation

### Python installation

You should install the pipeline engine into a virtual environment, as provided
by Virtualenv or Anaconda/Conda.

To install the pipeline engine's Python dependency:

    > Install OpenCV (e.g. sudo apt-get install python-opencv or conda install opencv)
    > virtualenv <path-to-my-new-environment>
    > source activate <path-to-my-new-environment>/bin/activate
    > pip install numpy
    > pip install cython
    > git clone https://github.com/microns-ariadne/pipeline_engine
    > git checkout use_luigi
    > cd pipeline_engine
    > git checkout use_luigi
    > pip install --process-dependency-links --trusted-host github.com .

In addition, you will have to compile Neuroproof and install either the Keras
or Caffe back-end.

### Anaconda installation

could not be simpler. Ask an admin of the vcg/microns_skeletonization project
to add you to the project. Then do the following:

    > git clone https://github.com/microns_ariadne/pipeline_engine
    > cd pipeline_engine
    > git checkout use_luigi
    > conda env create -f conda-install.yaml
    > source activate ariadne_microns_pipeline
    > pip install --process-dependency-links --editable .
    > pip install keras>=1.2.1

You may need to follow directions for configuring Theano on their website or
see directions for configuring Theano below

### Building stand alone version of Neuroproof

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

### Keras installation

The following installs Keras and Theano. (Note as of 2/6/2017, the Anaconda
version of Keras is older than 1.2.1 and will not work). You should have the
CUDA drivers and libraries pre-installed. Directions are on the NVidia website.

    > source activate <path-to-my-new-environment>/bin/activate
    > pip install keras>=1.2.1
    > pip install theano>=0.8.2

Follow directions for configuring Theano on their website.

### Theano installation

You should follow directions on their website. But if you want a shortcut.

* Make sure you have the CUDA and CUDNN installed (I will not help you there).
* Make sure /usr/local/cuda/bin (or similar) is on your path.
* Copy the following file to ~/.theanorc

    [global]
    device = gpu
    mode=FAST_RUN
    floatX=float32
    optimizer_including = cudnn:conv3d_gemm:convgrad3d_gemm:convtransp3d_gemm
    optimizer=fast_run
    exception_verbosity=high

    [lib]
    cnmem=0.50

    [dnn]
    enabled=1

    [dnn.conv]
    algo_fwd = time_on_shape_change
    algo_bwd_data = time_on_shape_change
    algo_bwd_filter = time_on_shape_change


### Caffe Installation

The Caffe classifier is optional. The version that we use has the PyGreentea
extension which uses a forked copy of Caffe. The following are the dockerfile
recipes that generate a Caffe environment:

```
FROM nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04
MAINTAINER William Grisaitis <grisaitisw@janelia.hhmi.org>
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        opencl-headers \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopenblas-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        libviennacl-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-pip \
        python-scipy && \
    rm -rf /var/lib/apt/lists/*
ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT
ENV CLONE_TAG=master
ENV COMMIT_SHA1=f32546500f60f5e756561c2f437465b4a80a37d8
RUN git clone -b ${CLONE_TAG} https://github.com/naibaf7/caffe.git . && \
    git checkout ${COMMIT_SHA1} && \
    for req in $(cat python/requirements.txt) pydot; do pip install $req; done
ADD Makefile.config.turagalab $CAFFE_ROOT/Makefile.config
RUN make pycaffe -j"$(nproc)"
ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig
WORKDIR /workspace
```

and

```
FROM turagalab/greentea:cudnn5-caffe_gt1.1
MAINTAINER William Grisaitis <grisaitisw@janelia.hhmi.org>
WORKDIR /opt/malis
RUN git clone -b master --depth 1 https://github.com/srinituraga/malis.git . && \
    pip install --upgrade numpy>=1.9 && \
    python setup.py install
WORKDIR /opt/zwatershed
ENV CLONE_TAG=v0.11
RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/TuragaLab/zwatershed.git . && \
    ./make.sh
ENV PYTHONPATH /opt/zwatershed:$PYTHONPATH
WORKDIR /opt/matplotlib
RUN git clone -b v1.5.2rc2 --depth 1 https://github.com/matplotlib/matplotlib.git . && \
    for req in $(cat requirements.txt) pydot; do pip install $req; done && \
    python setup.py install
WORKDIR /opt/PyGreentea
ENV CLONE_TAG=v0.9
RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/TuragaLab/PyGreentea.git . && \
    for req in $(cat requirements.txt) pydot; do pip install $req; done && \
    pip install PyCrypto PyPNG
ENV PYTHONPATH /opt/PyGreentea:$PYTHONPATH
WORKDIR /workspace
```

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

In general, supplementary files generated by the pipeline are json-encoded.
We use a standardized encoding of the extents of volumes in these JSON
files (see below). The volumes can also be programatically loaded using
the following code snippet (assuming that the JSON-loaded volume is
**volume_dict**):

    from ariadne_microns_pipeline.parameters import Volume

    volume = Volume(**volume_dict)

#### Volume object

The volume dictionary has the following keys:

* **x** - the x offset of the subvolume
* **y** - the y offset of the subvolume
* **z** - the z offset of the subvolume
* **width** - the width of the subvolume (x direction)
* **height** - the height of the subvolume (y direction)
* **depth** - the depth of the subvolume (z direction)

#### Storage plan file format

The pipeline uses storage plans to figure out how to store volume data on
disk. Each storage plan names a dataset type (e.g. "neuroproof" for aggregated
segmentations) and a volume and taking these together, a storage plan tells
where the pipeline will store (or has stored) a volume of data.

Storage plans describe the subvolumes that make up a volume. These subvolumes
are designed so that a loading plan will only load whole subvolumes to avoid
having to scan through a large volume to load a small amount of data. The
subvolumes are currently stored as .TIFF stacks.

Storage plans have the extension, ".storage.plan". When a task has successfully
completed a storage plan, it copies the storage plan to a file with the
extension, ".storage.done", so that downstream Luigi tasks' dependencies
will satistfy.

Storage plans can be programatically loaded using the following code snippet:

    from ariadne_microns_pipeline.targets.volume_target import SrcVolumeTarget

    data = SrcVolumeTarget(path).imread()

The storage plan is a JSON file that encodes a dictionary with the following
keys:

* **blocks** (list) - this is a list of the subvolumes in the plan. Each
element of the list is a two-tuple composed of:
    * a volume (see [JSON objects](#json-objects)) giving the location and
extent of the the subvolume
    * the path to the image stack with the volume data, e.g. a TIFF file
* **dimensions** (list) - a 3-tuple of depth (z), height (y) and width (x)
* **datatype** (string)- the Numpy datatype of the volume data, one of "uint8",
"uint16", "uint32", "float32" or "float64"
* **dataset_id** (integer) - a unique ID for the storage plan within the
context of the pipeline.
* **dataset_name** (string) - the name of the data type, e.g. "neuroproof",
"image" or "membrane"

### Load plan file format

The pipeline uses load plans to figure out how to assemble volume data,
often across multiple storage plans on behalf of one or more tasks that
need to load that data. The load plan describes the blocks to load and it
also contains a list of the .done files (from the tasks that produce the data)
that must be present before the data can be loaded.

Load plans can be programatically loaded using the following code snippet:

from ariadne_microns_pipeline.targets.volume_target import DestVolumeReader

data = DestVolumeReader(path).imread()

Load plan files are JSON-encoded and generally have extensions of
"loading.plan". The load plan is a dictionary with the following keys:

* **blocks** (list) - this is a list of two-tuples. *Note* - the order of this
list is the opposite of that of the storage plan.
    * the path to the image stack with the volume data, e.g. a TIFF file
    * a volume (see [JSON objects](#json-objects)) giving the location and
extent of the the subvolume
  * **dimensions** (list) - a 3-tuple of depth (z), height (y) and width (x)
  * **datatype** (string)- the Numpy datatype of the volume data, one of "uint8",
"uint16", "uint32", "float32" or "float64"
  * **loading_plan_id** (integer) - a unique ID for the loading plan within the
context of the pipeline.
  * **dataset_name** (string) - the name of the data type, e.g. "neuroproof",
"image" or "membrane"
  * **dataset_done_files** (list) - this is a list of the .done files for the
storage plans that comprise the loading plan. These .done
files should be present on-disk before attempting to load
the volume data.

### connectivity-graph.json file format

The connectivity graph JSON file is a dictionary composed of the following
keys:

* **count** (int): the number of neurons segmented

* **locations** (list): a list of two-tuples, one per subvolume, giving the
volume and storage location of the Neuroproof segmentation of that subvolume
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

### neuron-locations.json file format

The neuron-locations.json file lists the keypoint location for every neuron id
using the global neuron ids from the connectivity-graph.json file. These are
the voxel coordinates of the voxel in each segment that is farthest from the
edge of the neuron. The format is a dictionary of lists where each list
can be viewed as the column of a table. The dictionary keys are:
* **neuron_id** (list) - a list of the global neuron-ids in the volume.
* **x** (list) - a list of the x coordinates of the neuron keypoints
* **y** (list) - a list of the y coordinates of the neuron keypoints
* **z** (list) - a list of the z coordinates of the neuron keypoints

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
lists of two-tuples of volume (see [Volume object](#volume-object)) and the
path to the multi-plane .TIF file that contains the data for that volume.

In addition, the following dictionary keys can be used to retrieve the
image volume from Butterfly:

* *experiment* - the name of the experiment in Butterfly
* *sample* - an accession number or description of the tissue sample
* *dataset* - a name for the dataset / volume, e.g. "sem" for electron
              microscopy data
* *channel* - the name of the image channel

The microns-volume script [ariadne_microns_pipeline/scripts/microns_volume.py]
has an example of how to use the index.json file to read an arbitrary
subvolume of an arbitrary channel.

### loading plan / storage plan (*.plan) files

These are plans for how to load or store .tif stacks to make a volume. The
file is a .json file with the following structure:

* **dimensions** (list of 3 values) - the depth, height and width of the volume
* **x** (integer) - the x-offset of the volume
* **y** (integer) - the y-offset of the volume
* **z** (integer) - the z-offset of the volume
* **datatype** (string) - the Numpy dtype of the data, typically "uint8",
                          "uint16", or "uint32"
* **dataset_name** (string) - the name of the dataset's type, e.g. "image" or
                              "neuroproof"
* **dataset_id** (integer) - the database ID of the storage plan (only if
                             storage plan)
* **loading_plan_id** (integer) - the database ID of the loading plan (only if
                                loading plan)
* **blocks** (list) - a list of two-tuples. The first element of the two-tuple
                      is a volume (see Volume object above for format). The
                      second is the pathname of a .tif stack that holds the
                      block's data.

To load a volume, create an array of the given dimensions, then iterate over
the blocks, reading and placing them.
