#!/usr/bin/env python
import os
import numpy as np

from setuptools import setup, Extension
import cython

VERSION = "0.1.0"

README = open('README.md').read()

watershed_ext = Extension(
    name="ariadne_microns_pipeline.algorithms._am_watershed",
    language="c++",
    sources=[os.path.join("watershed", _)
             for _ in "ws_alg.cpp", "ws_queue.cpp"] + [
             os.path.join("ariadne_microns_pipeline", "algorithms", 
                          "_am_watershed.pyx")],
    include_dirs=["watershed", np.get_include()])

setup(
    name='ariadne-microns-pipeline',
    version=VERSION,
    packages=["ariadne_microns_pipeline",
              "ariadne_microns_pipeline.targets",
              "ariadne_microns_pipeline.tasks",
              "ariadne_microns_pipeline.algorithms"],
    url="https://github.com/microns-ariadne/pipeline_engine",
    description="Connectome project's pipeline",
    long_description=README,
    install_requires=[
        "Cython>=0.24.0",
        "python-dateutil>=2.2",
        "enum34>=1.0.0",
        "mahotas",
        "numpy>=1.9.3",
        "pandas>=0.15.0",
        "h5py>=2.5.0",
        "scipy>=0.14.0",
        "scikit-learn",
        "sqlalchemy>=1.0.0",
        "luigi>=2.1.1",
        "pyzmq"],
    ext_modules=[watershed_ext],
    entry_points={
        "console_scripts": [
            "microns-ipc-broker = ariadne_microns_pipeline.ipc.ipcbroker:main",
            "microns-ipc-worker = ariadne_microns_pipeline.ipc.ipcworker:main",
            "microns-ipc-echo = ariadne_microns_pipeline.ipc.ipcecho:main"
            ]},
    zip_safe=False
)
