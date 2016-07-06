## Installation

pipeline_engine has the following dependencies

OpenCV 2.4
Vigra
Cilkplus
Jsoncpp (https://github.com/open-source-parsers/jsoncpp)
HDF5 1.8+

Dependencies should have `CXXFLAGS=-std=c++11` defined, e.g. when running CMake

Building Vigra is an art in and of itself. The happy path is to build Boost
using the Cilk compiler and perhaps you will succeed. Building Boost:

* Download and unpack Boost to a directory, e.g. ~/tools/boost/boost-<version>
* CD into the Boost directory
* Type `./bootstrap.sh`. This produces the "b2" program and project-config.jam
* Find the line in project-config.jam starting with "using gcc" and edit
it to point to your compiler. Mine said
`using gcc : 4.9.0 : /home/leek/tools/cilk/cilkplus-install/bin/g++` when I was
done. I also edited the line starting with "using python" to point at my
virtualenv's Python. Boost Python is nice to have but not necessary.
* Type `./b2 --prefix=<path-to-boost-install>`
* Wait... a long time. Remember, most of Boost requires absolutely no
compilation. But Boost is big, even the part that requires compilation is
very big.

When building, you should supply the locations to the install directories for
these:

OPENCV_PREFIX
CILKPLUS_PREFIX
VIGRA_PREFIX
JSONCPP_PREFIX
