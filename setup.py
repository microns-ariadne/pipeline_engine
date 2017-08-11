#!/usr/bin/env python
import os
import numpy as np

from setuptools import setup, Extension

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
              "ariadne_microns_pipeline.analysis",
              "ariadne_microns_pipeline.algorithms",
              "ariadne_microns_pipeline.classifiers",
              "ariadne_microns_pipeline.ipc",
              "ariadne_microns_pipeline.pipelines",
              "ariadne_microns_pipeline.targets",
              "ariadne_microns_pipeline.tasks",
              "ariadne_microns_pipeline.utilities"
              ],
    url="https://github.com/microns-ariadne/pipeline_engine",
    description="Connectome project's pipeline",
    long_description=README,
    install_requires=[
        "Cython>=0.24.0",
        "frozenordereddict>=1.2.0",
        "python-dateutil>=2.2",
        "enum34>=1.0.0",
        "hungarian>=0.2.3",
        "mahotas",
        "matplotlib",
        "numpy>=1.9.3",
        "pandas>=0.15.0",
        "h5py>=2.5.0",
        "scipy>=0.14.0",
        "scikit-learn",
        "sqlalchemy>=1.0.0",
        "luigi>=2.1.1",
        "pyzmq",
        "tifffile>=0.11.1",
        "rh_config",
        "rh_logger",
        "rh-fast64counter",        
        "microns-skeletonization"],
    ext_modules=[watershed_ext],
    dependency_links = [
        'https://github.com/Rhoana/rh_config/archive/1.0.0.tar.gz#egg=rh_config-1.0.0',
        'https://github.com/Rhoana/rh_logger/archive/2.0.0.tar.gz#egg=rh_logger-2.0.0',
        'https://github.com/Rhoana/fast64counter/archive/v1.0.0.tar.gz#egg=rh-fast64counter-1.0.0',
        'git+ssh://git@github.com/microns-ariadne/microns_skeletonization.git#egg=microns_skeletonization-0.1.0'
        ],
    entry_points={
        "console_scripts": [
            "microns-ipc-broker = ariadne_microns_pipeline.ipc.ipcbroker:main",
            "microns-ipc-worker = ariadne_microns_pipeline.ipc.ipcworker:main",
            "microns-ipc-echo = ariadne_microns_pipeline.ipc.ipcecho:main",
            "microns-volume = ariadne_microns_pipeline.scripts.microns_volume:main",
            "pickle-a-classifier = ariadne_microns_pipeline.scripts.pickle_a_classifier:main",
            "create-storage-plan = ariadne_microns_pipeline.scripts.create_storage_plan:main"
            ]},
    zip_safe=False
)
