#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from distutils.core import setup, Extension
import numpy as np
import os

os.environ["CC"] = os.environ["CXX"] = "/home/armafire/tools/cilkplus-install/bin/g++"

module = Extension('cilk_rw',
                   sources = ['cilk_rw.cpp'],
                   include_dirs = ['/home/armafire/tools/opencv-3-install-test/include'],
                   libraries = ['opencv_core', 'boost_filesystem', 'opencv_highgui', 'cilkrts'],
                   library_dirs = ['/home/armafire/tools/opencv-3-install-test/lib'],
                   extra_compile_args = ['-fcilkplus', '-std=c++11', '-O3']
                   )

setup(
        name = 'cilk_rw',
        version = '1.0',
        include_dirs = [np.get_include()],
        ext_modules = [module]
      )

