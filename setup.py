#!/usr/bin/env python
import os

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
    include_dirs=["watershed"])

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
        "dateutil>=2.2",
        "enum34>=1.0.0",
        "numpy>=1.9.3",
        "h5py>=2.6.0",
        "scipy>=0.16.0",
        "scikit-learn",
        "sqlalchemy>=1.0.0",
        "luigi>=2.1.1"],
    ext_modules=[watershed_ext],
    entry_points={},
    zip_safe=False
)
