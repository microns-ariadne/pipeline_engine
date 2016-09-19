##################################################
#
# Microns / Ariadne pipeline Dockerfile
#
# Usage:
#
#   To build (
#     sudo docker build -t microns:pipeline
#
##################################################

FROM ubuntu:15.10

#####################################################
#
# Pasted from https://github.com/NVIDIA/nvidia-docker/blob/0c35edfb16e7a65ac53e49ec5d816942608d37e8/ubuntu-14.04/cuda/7.5/runtime/Dockerfile
#
#####################################################

LABEL com.nvidia.volumes.needed="nvidia_driver"

ENV NVIDIA_GPGKEY_SUM bd841d59a27a406e513db7d405550894188a4c1cd96bf8aa4f82f1b39e0b5c1c
ENV NVIDIA_GPGKEY_FPR 889bee522da690103c4b085ed88c3d385c37d3be

RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/GPGKEY && \
    apt-key adv --export --no-emit-version -a $NVIDIA_GPGKEY_FPR | tail -n +2 > cudasign.pub && \
    echo "$NVIDIA_GPGKEY_SUM  cudasign.pub" | sha256sum -c --strict - && rm cudasign.pub && \
    echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64 /" > /etc/apt/sources.list.d/cuda.list

ENV CUDA_VERSION 7.5
LABEL com.nvidia.cuda.version="7.5"

ENV CUDA_PKG_VERSION 7-5=7.5-18
RUN apt-get update && apt-get install -y --no-install-recommends --force-yes \
        cuda-nvrtc-$CUDA_PKG_VERSION \
        cuda-cusolver-$CUDA_PKG_VERSION \
        cuda-cublas-$CUDA_PKG_VERSION \
        cuda-cufft-$CUDA_PKG_VERSION \
        cuda-curand-$CUDA_PKG_VERSION \
        cuda-cusparse-$CUDA_PKG_VERSION \
        cuda-npp-$CUDA_PKG_VERSION \
        cuda-cudart-$CUDA_PKG_VERSION && \
    ln -s cuda-$CUDA_VERSION /usr/local/cuda
    
####################################################
#
# Pasted from https://raw.githubusercontent.com/NVIDIA/nvidia-docker/865387c6b7bfac14a10b564513b120b20f682dfc/ubuntu-14.04/cuda/7.5/devel/Dockerfile
#
####################################################
RUN apt-get update && apt-get install -y --no-install-recommends --force-yes \
        cuda-core-$CUDA_PKG_VERSION \
        cuda-misc-headers-$CUDA_PKG_VERSION \
        cuda-command-line-tools-$CUDA_PKG_VERSION \
        cuda-license-$CUDA_PKG_VERSION \
        cuda-nvrtc-dev-$CUDA_PKG_VERSION \
        cuda-cusolver-dev-$CUDA_PKG_VERSION \
        cuda-cublas-dev-$CUDA_PKG_VERSION \
        cuda-cufft-dev-$CUDA_PKG_VERSION \
        cuda-curand-dev-$CUDA_PKG_VERSION \
        cuda-cusparse-dev-$CUDA_PKG_VERSION \
        cuda-npp-dev-$CUDA_PKG_VERSION \
        cuda-cudart-dev-$CUDA_PKG_VERSION \
        cuda-driver-dev-$CUDA_PKG_VERSION

####################################################
#
# Pasted from https://github.com/NVIDIA/nvidia-docker/blob/865387c6b7bfac14a10b564513b120b20f682dfc/ubuntu-14.04/cuda/7.5/devel/cudnn4/Dockerfile
#
####################################################

RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

ENV CUDNN_VERSION 4
LABEL com.nvidia.cudnn.version="4"

RUN apt-get update && apt-get install -y --no-install-recommends --force-yes \
            libcudnn4=4.0.7

RUN echo "/usr/local/cuda/lib" >> /etc/ld.so.conf.d/cuda.conf && \
    echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/cuda.conf && \
    ldconfig

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}

###################################################
#
# Begin microns ariadne install
#
###################################################
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get -y install libbz2-dev \
                       libc6-dev-i386 \
                       libfftw3-dev \
                       libhdf5-dev \
                       libjpeg-dev \
                       libpng-dev \
                       libtiff-dev \
                       libz-dev \
                       bison \
                       cmake \
                       flex \
                       gcc \
                       git \
                       make \
                       unzip \
                       wget \
                       python \
                       cython \
                       python-h5py \
                       python-matplotlib \
                       python-numpy \
                       python-pip \
                       python-scipy \
                       python-opencv \
                       python-zmq
#
# A directory for the repo source code
#
RUN mkdir /src
WORKDIR /src
ADD . /src
#
# A directory for the install
#
RUN mkdir /usr/local/microns
ENV TOOLS_PREFIX=/usr/local/microns
ENV CUDA_PREFIX=/usr/local/cuda
#
# There's a bug in the cuda docker that puts a colon at the end of
# $LIBRARY_PATH. It crashes the cilkplus compile.
#
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs
#
# Make cilkplus first so that we get an intermediate docker after
# this long step
#
RUN make -j 12 $TOOLS_PREFIX/cilkplus/cilkplus-gcc
RUN make $TOOLS_PREFIX/sources
RUN make -j 12 $TOOLS_PREFIX/opencv-2.4/opencv-install
RUN make -j 12 $TOOLS_PREFIX/boost/boost-install
RUN make -j 12
RUN pip install -r requirements.txt
RUN pip install https://github.com/Rhoana/rh_config/archive/1.0.0.tar.gz
RUN pip install https://github.com/Rhoana/rh_logger/archive/2.0.0.tar.gz
RUN pip install https://github.com/Rhoana/fast64counter/archive/master.zip#egg=fast64counter-1.0.0
RUN pip install --editable .
