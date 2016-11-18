# Makefile for pipeline_engine
#
# This builds the world needed by fc_dnn and neuroproof. Obviously, there
# must be a better way, but I do not know it.
#
# TOOLS_PREFIX should be defined. This is where CILKPlus, Boost, Vigra and
# OpenCV will be built
#
# Define CUDA_PREFIX to point at the CUDA install.
#
# Sacrifice a goat. Or a vegetarian goat substitute.
#
#----------------------------
#
# PREREQUISITES:
#
# For Neuroproof, you need:
#
# libhdf5
#
# For Boost, you need:
#
# libbz2-dev
#
# For Vigra, you need to have the following libraries installed in their
# expected locations:
#
# libpng
# libjpeg
# libfftw
# libtiff
# libz
#
#-------------------------------------------
#
# Notes:
#
# I had to apply the following patch to CILKPLUS to get it to work on my
# computer:
#
# diff --git a/libsanitizer/sanitizer_common/sanitizer_linux.cc b/libsanitizer/sanitizer_common/sanitizer_linux.cc
# index 06e5a0a..96e9e90 100644
# --- a/libsanitizer/sanitizer_common/sanitizer_linux.cc
# +++ b/libsanitizer/sanitizer_common/sanitizer_linux.cc
# @@ -10,6 +10,8 @@
#  // sanitizer_libc.h.
#  //===----------------------------------------------------------------------===//
#  #ifdef __linux__
# +#define __ARCH_WANT_SYSCALL_NO_AT 1
# +#define __ARCH_WANT_SYSCALL_NO_FLAGS 1
# 
#  #include "sanitizer_common.h"
#  #include "sanitizer_internal_defs.h"
#
# -----------
#
# Vigra does not build vigranumpy. If you want Vigra for Python (e.g. Ilastik)
# this build is not going to be compatible and you have to build a separate
# instance for your Python.
#
# On the cluster, we use "module load hdf5" etc. to reconfigure LD_LIBRARY_PATH
# and similar to point to hdf5, fftw and jpeg. These are not in their
# standard locations which is where this makefile and Vigra's CMAKE expect
# them. It seemed fairly arbitrary to define JPEG_LIBRARY in the top-level
# script and not, for instance, define TIFF_LIBRARY even though it doesn't
# need to be defined on the cluster.
#
# I had to change the following in Vigra's CMakeCache.txt
#
#FFTW3F_INCLUDE_DIR:PATH=/n/sw/fasrcsw/apps/Core/fftw/3.3.4-fasrc08/include
#FFTW3F_LIBRARY:FILEPATH=/n/sw/fasrcsw/apps/Core/fftw/3.3.4-fasrc08/lib
#FFTW3_INCLUDE_DIR:PATH=/n/sw/fasrcsw/apps/Core/fftw/3.3.4-fasrc08/include
#FFTW3_LIBRARY:FILEPATH=/n/sw/fasrcsw/apps/Core/fftw/3.3.4-fasrc08/lib
#HDF5_CORE_LIBRARY:FILEPATH=/n/sw/fasrcsw/apps/Core/hdf5/1.8.12-fasrc08/lib64/libhdf5.so
#HDF5_HL_LIBRARY:FILEPATH=/n/sw/fasrcsw/apps/Core/hdf5/1.8.12-fasrc08/lib64/libhdf5_hl.so
#HDF5_INCLUDE_DIR:PATH=/n/sw/fasrcsw/apps/Core/hdf5/1.8.12-fasrc08/include
#HDF5_SZ_LIBRARY:FILEPATH=/n/sw/fasrcsw/apps/Core/szip/2.1-fasrc01/lib64/libsz.so
#JPEG_INCLUDE_DIR:PATH=/n/sw/fasrcsw/apps/Core/jpeg/6b-fasrc01/include
#JPEG_LIBRARY:FILEPATH=/n/sw/fasrcsw/apps/Core/jpeg/6b-fasrc01/lib


AUTOCONF_PREFIX=$(TOOLS_PREFIX)/autoconf
CILKPLUS_PREFIX=$(TOOLS_PREFIX)/cilkplus
OPENCV_PREFIX=$(TOOLS_PREFIX)/opencv-2.4
BOOST_PREFIX=$(TOOLS_PREFIX)/boost
VIGRA_PREFIX=$(TOOLS_PREFIX)/vigra
JSONCPP_PREFIX=$(TOOLS_PREFIX)/jsoncpp
SPARSEHASH_INSTALL=$(TOOLS_PREFIX)/sparsehash/sparsehash-install

CILKPLUS_CXX_COMPILER=$(CILKPLUS_PREFIX)/cilkplus-install/bin/g++
CILKPLUS_LINKER=$(CILKPLUS_PREFIX)/cilkplus-install/bin/g++
CILKPLUS_C_COMPILER=$(CILKPLUS_PREFIX)/cilkplus-install/bin/gcc

OPENCV_INCLUDE=$(OPENCV_PREFIX)/include
OPENCV_LIB=$(OPENCV_PREFIX)/lib
AUTOCONF_INSTALL=$(AUTOCONF_PREFIX)/autoconf-install

#
# This is the user-config.jam file text needed by Boost
#
define USER_CONFIG_JAM
using gcc : 4.9.0 : $(CILKPLUS_CXX_COMPILER) ;
endef
export USER_CONFIG_JAM

.PHONY: all np-merge fc_dnn all_sources cilkplus-sources clean

all: $(TOOLS_PREFIX)/sources neuroproof fc_dnn/src/run_dnn

clean:
	cd fc_dnn/src &&\
	make clean &&\
	cd neuroproof/MIT_agg/MIT_agg/neuroproof_agg/npclean &&\
	make clean

neuroproof: $(TOOLS_PREFIX)/sparsehash/sparsehash-install \
	    $(TOOLS_PREFIX)/opencv-2.4/opencv-install \
	    $(TOOLS_PREFIX)/vigra/vigra-install \
	    $(TOOLS_PREFIX)/jsoncpp/jsoncpp-install
	cd neuroproof/MIT_agg/MIT_agg/neuroproof_agg/npclean &&\
	OPENCV_DIR=$(OPENCV_PREFIX)/opencv-install \
	VIGRA_DIR=$(VIGRA_PREFIX)/vigra-install \
	SPARSEHASH_DIR=$(SPARSEHASH_INSTALL) \
	JSONCPP_DIR=$(JSONCPP_PREFIX)/jsoncpp-install \
	CILK_DIR=$(CILKPLUS_PREFIX)/cilkplus-install \
	BOOST_DIR=$(BOOST_PREFIX)/boost-install \
	make all learn

$(TOOLS_PREFIX)/sparsehash/sparsehash-install:
	mkdir -p $(TOOLS_PREFIX)/sparsehash &&\
	cd neuroproof/sparsehash &&\
	./configure --prefix=$(SPARSEHASH_INSTALL) &&\
	make &&\
	make install

all_sources: $(TOOLS_PREFIX)/sources/opencv-2.4.13.zip \
	     $(TOOLS_PREFIX)/sources/autoconf-2.64.tar.gz \
		$(CILKPLUS_PREFIX)/cilkplus-gcc\
		$(TOOLS_PREFIX)/sources/boost-1.61.0.tar.gz\
		$(TOOLS_PREFIX)/sources/vigra-1.10.0-src.tar.gz

$(TOOLS_PREFIX)/sources:
	mkdir -p $@

$(TOOLS_PREFIX)/sources/opencv-2.4.13.zip:
	wget https://github.com/opencv/opencv/archive/2.4.13.zip -O $@
#	wget http://downloads.sourceforge.net/project/opencvlibrary/opencv-unix/2.4.9/opencv-2.4.9.zip -O $@

$(TOOLS_PREFIX)/sources/autoconf-2.64.tar.gz:
	wget http://ftp.gnu.org/gnu/autoconf/autoconf-2.64.tar.gz -O $@

$(TOOLS_PREFIX)/sources/boost-1.61.0.tar.gz:
	wget http://downloads.sourceforge.net/project/boost/boost/1.61.0/boost_1_61_0.tar.gz -O $@

$(TOOLS_PREFIX)/sources/vigra-1.10.0-src.tar.gz:
	wget https://github.com/ukoethe/vigra/releases/download/Version-1-11-0/vigra-1.11.0-src.tar.gz -O $@

$(TOOLS_PREFIX)/sources/jsoncpp-1.7.3.tar.gz:
	wget https://github.com/open-source-parsers/jsoncpp/archive/1.7.3.tar.gz -O $@

$(TOOLS_PREFIX)/autoconf/autoconf-install/bin/autoconf: $(TOOLS_PREFIX)/sources/autoconf-2.64.tar.gz
	mkdir -p $(AUTOCONF_PREFIX) &&\
	cd $(AUTOCONF_PREFIX) &&\
	tar -xvf $(TOOLS_PREFIX)/sources/autoconf-2.64.tar.gz &&\
	cd autoconf-2.64 &&\
	./configure --prefix=$(AUTOCONF_INSTALL) &&\
	make &&\
	make install

$(TOOLS_PREFIX)/cilkplus/cilkplus-gcc:
	mkdir -p $(TOOLS_PREFIX)/cilkplus &&\
	cd $(CILKPLUS_PREFIX) &&\
	git clone https://github.com/gcc-mirror/gcc.git cilkplus-gcc &&\
	cd cilkplus-gcc &&\
	git checkout origin/cilkplus

$(TOOLS_PREFIX)/cilkplus/cilkplus-install/bin/g++: $(TOOLS_PREFIX)/cilkplus/cilkplus-gcc $(TOOLS_PREFIX)/autoconf/autoconf-install/bin/autoconf
	cd $(CILKPLUS_PREFIX)/cilkplus-gcc &&\
	git clean -xfd &&\
	./contrib/download_prerequisites &&\
	mkdir -p $(CILKPLUS_PREFIX)/build &&\
	cd $(CILKPLUS_PREFIX)/build &&\
	$(CILKPLUS_PREFIX)/cilkplus-gcc/configure --prefix=$(CILKPLUS_PREFIX)/cilkplus-install --enable-languages="c,c++" &&\
	PATH=$(AUTOCONF_INSTALL)/bin:$$PATH make &&\
	make install

$(TOOLS_PREFIX)/jsoncpp/jsoncpp-install: $(TOOLS_PREFIX)/sources/jsoncpp-1.7.3.tar.gz $(TOOLS_PREFIX)/cilkplus/cilkplus-install/bin/g++
	mkdir -p $(TOOLS_PREFIX)/jsoncpp
	cd $(TOOLS_PREFIX)/jsoncpp &&\
	tar -xvf $(TOOLS_PREFIX)/sources/jsoncpp-1.7.3.tar.gz &&\
	mkdir -p build &&\
	cd build &&\
	cmake -DCMAKE_INSTALL_PREFIX=$@ \
	      -DBUILD_SHARED_LIBS=ON \
	      -DBUILD_STATIC_LIBS=OFF \
	      -DCMAKE_CXX_FLAGS=-std=c++11 \
	      -DCMAKE_CXX_COMPILER=$(CILKPLUS_CXX_COMPILER) \
	      $(TOOLS_PREFIX)/jsoncpp/jsoncpp-1.7.3 &&\
	make &&\
	make install

#
# Need to patch modules/gpu/src/nvidia/core/NCVPixelOperations.hpp
#
$(TOOLS_PREFIX)/opencv-2.4/opencv-install: $(TOOLS_PREFIX)/sources/opencv-2.4.13.zip $(TOOLS_PREFIX)/cilkplus/cilkplus-install/bin/g++
	mkdir -p $(OPENCV_PREFIX) &&\
	cd $(OPENCV_PREFIX) &&\
	unzip $(TOOLS_PREFIX)/sources/opencv-2.4.13.zip &&\
	wget https://raw.githubusercontent.com/opencv/opencv/2.4.13/modules/gpu/src/nvidia/core/NCVPixelOperations.hpp \
	    -O opencv-2.4.13/modules/gpu/src/nvidia/core/NCVPixelOperations.hpp &&\
	mkdir -p build &&\
	cd build &&\
	PATH=$(TOOLS_PREFIX)/cilkplus/cilkplus-install/bin/gcc:$(PATH) \
	LIBRARY_PATH=$(TOOLS_PREFIX)/cilkplus/cilkplus-install/lib64:$(LIBRARY_PATH) \
	cmake -DCMAKE_INSTALL_PREFIX=$@ \
	      -DBUILD_SHARED_LIBS=ON \
	      -DBUILD_STATIC_LIBS=OFF \
	      -DBUILD_OPENEXR=OFF \
	      -DWITH_FFMPEG= OFF \
	      "-DCMAKE_CXX_FLAGS=-std=c++11" \
	      -DCMAKE_CXX_COMPILER=$(CILKPLUS_CXX_COMPILER) \
	      -DCMAKE_C_COMPILER=$(CILKPLUS_C_COMPILER) \
	      "-DCUDA_ARCH_BIN=3.0 3.5" \
	      -DCUDA_HOST_COMPILER=$(TOOLS_PREFIX)/cilkplus/cilkplus-install/bin/gcc \
	      -DCUDA_CUDART_LIBRARY=$(CUDA_PREFIX)/lib64/libcudart.so \
	      -DCUDA_CUDA_LIBRARY=$(CUDA_PREFIX)/lib64/libcuda.so \
	      -DCUDA_NVCC_EXECUTABLE=$(CUDA_PREFIX)/bin/nvcc \
	      "-DCUDA_NVCC_FLAGS=-std=c++11 --expt-relaxed-constexpr" \
	      -DCUDA_SDK_ROOT_DIR=$(CUDA_PREFIX) \
	      -DWITH_OPENEXR=OFF \
	      $(OPENCV_PREFIX)/opencv-2.4.13 &&\
	make &&\
	make install

$(TOOLS_PREFIX)/boost/boost-install: $(TOOLS_PREFIX)/sources/boost-1.61.0.tar.gz
	mkdir -p $(BOOST_PREFIX) && \
	cd $(BOOST_PREFIX) && \
	tar -xvf $(TOOLS_PREFIX)/sources/boost-1.61.0.tar.gz && \
	cd boost_1_61_0 && \
	echo $$USER_CONFIG_JAM > tools/build/src/user-config.jam &&\
	./bootstrap.sh &&\
	./b2 --prefix=$(BOOST_PREFIX)/boost-install &&\
	./b2 install --prefix=$(BOOST_PREFIX)/boost-install

$(TOOLS_PREFIX)/vigra/vigra-install: $(TOOLS_PREFIX)/sources/vigra-1.10.0-src.tar.gz $(BOOST_PREFIX)/boost-install
	mkdir -p $(VIGRA_PREFIX)/build &&\
	cd $(VIGRA_PREFIX) &&\
	tar -xvf $(TOOLS_PREFIX)/sources/vigra-1.10.0-src.tar.gz && \
	cd build && \
	cmake -DCMAKE_INSTALL_PREFIX=$(VIGRA_PREFIX)/vigra-install \
	      -DBoost_DIR=$(BOOST_PREFIX)/boost-install \
	      -DBoost_INCLUDE_DIR=$(BOOST_PREFIX)/boost-install/include \
	      -DBoost_LIBRARY_DIR=$(BOOST_PREFIX)/boost-install/lib \
	      -DCMAKE_CXX_COMPILER=$(CILKPLUS_CXX_COMPILER) \
	      "-DCMAKE_CXX_FLAGS=-std=c++11 -pthread -W -Wall -Wextra " \
	      -DCMAKE_C_COMPILER=$(CILKPLUS_C_COMPILER) \
	      -DWITH_BOOST_GRAPH=OFF \
	      -DWITH_BOOST_THREAD=ON \
	      -DWITH_OPENEXR=OFF \
	      -DWITH_VIGRANUMPY=OFF \
	      $(VIGRA_PREFIX)/vigra-1.11.0 &&\
	make && \
	make install

fc_dnn/src/run_dnn: $(TOOLS_PREFIX)/cilkplus/cilkplus-install/bin/g++ $(TOOLS_PREFIX)/opencv-2.4/opencv-install
	cd fc_dnn/src &&\
	CC=$(CILKPLUS_CXX_COMPILER) \
	LD=$(CILKPLUS_LINKER) \
	CILKPLUS_PREFIX=$(CILKPLUS_PREFIX)/cilkplus-install \
	OPENCV_PREFIX=$(OPENCV_PREFIX)/opencv-install \
	make

