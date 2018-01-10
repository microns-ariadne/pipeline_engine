from contextlib import contextmanager
import os
import unittest
import numpy as np
import h5py
import cPickle
from scipy.ndimage import zoom
import tempfile

from ..hdf_classifier import HDF5Classifier
from ...tasks.classify import ClassifyTask

@contextmanager
def make_hdf5(shape):
    '''Make a random uint8 hdf5 file of the given size'''
    data = np.random.RandomState(1234).randint(0, 256, shape).astype(np.uint8)
    fileno, filename = tempfile.mkstemp(".h5")
    with h5py.File(filename, "w") as fd:
        fd.create_dataset("stack", data=data)
    yield filename, data
    os.close(fileno)
    os.remove(filename)
    
    
class TestHDF5Classifier(unittest.TestCase):
    def test_01_01_pickle(self):
        pkl = cPickle.dumps(
            HDF5Classifier(["foo"], [("/bar/baz", 1, 2, 3, 4, 5, 6)]))
        c = cPickle.loads(pkl)
        self.assertIsInstance(c, HDF5Classifier)
        self.assertEqual(len(c.channel_names), 1)
        self.assertEqual(c.channel_names[0], "foo")
        self.assertEqual(len(c.blockdescs), 1)
        self.assertEqual(c.blockdescs[0].filename, "/bar/baz")
        self.assertEqual(c.blockdescs[0].volume.x, 1)
        self.assertEqual(c.blockdescs[0].volume.y, 2)
        self.assertEqual(c.blockdescs[0].volume.z, 3)
        self.assertEqual(c.blockdescs[0].volume.width, 4)
        self.assertEqual(c.blockdescs[0].volume.height, 5)
        self.assertEqual(c.blockdescs[0].volume.depth, 6)
        self.assertEqual(c.resolution, 0)
        self.assertEqual(c.axes_indexes, (0, 1, 2, 3))
    
    def test_01_02_pickle_resolution(self):
        pkl = cPickle.dumps(
            HDF5Classifier(["foo"], [("/bar/baz", 1, 2, 3, 4, 5, 6)],
                           resolution=2))
        c = cPickle.loads(pkl)
        self.assertEqual(c.resolution, 2)
    
    def test_01_03_pickle_axes_indexes(self):
        pkl = cPickle.dumps(
            HDF5Classifier(["foo"], [("/bar/baz", 1, 2, 3, 4, 5, 6)],
                           axes_indexes=(3, 0, 1, 2)))
        c = cPickle.loads(pkl)
        self.assertEqual(c.axes_indexes, (3, 0, 1, 2))
    
    def test_02_01_single_channel(self):
        with make_hdf5((1, 10, 20, 30)) as (filename, data):
            classifier = HDF5Classifier(
                ["channel"],
                [(filename, 0, 0, 0, 30, 20, 10)])
            result = classifier.classify(
                np.zeros((10, 20, 30), np.uint8), 0, 0, 0)
            self.assertEqual(len(result), 1)
            self.assertEqual(result.keys()[0], "channel")
            np.testing.assert_array_equal(result["channel"], data[0])
    
    def test_02_02_multiple_channels(self):
        with make_hdf5((2, 10, 20, 30)) as (filename, data):
            classifier = HDF5Classifier(
                ["channel1", "channel2"],
                [(filename, 0, 0, 0, 30, 20, 10)])
            result = classifier.classify(
                np.zeros((10, 20, 30), np.uint8), 0, 0, 0)
            self.assertEqual(len(result), 2)
            self.assertIn("channel1", result)
            self.assertIn("channel2", result)
            np.testing.assert_array_equal(result["channel1"], data[0])
            np.testing.assert_array_equal(result["channel2"], data[1])
    
    def test_02_03_resolution(self):
        with make_hdf5((1, 10, 20, 30)) as (filename, data):
            classifier = HDF5Classifier(
                ["channel"],
                [(filename, 0, 0, 0, 60, 40, 10)],
                resolution=1)
            result = classifier.classify(
                np.zeros((10, 40, 60), np.uint8), 0, 0, 0)
            big_data = zoom(data, (1, 1, 2, 2), order=1)
            np.testing.assert_array_equal(
                result["channel"], big_data[0])
    
    def test_02_04_offset(self):
        with make_hdf5((1, 10, 20, 30)) as (filename, data):
            classifier = HDF5Classifier(
                ["channel"],
                [(filename, 0, 0, 0, 30, 20, 10)])
            result = classifier.classify(
                np.zeros((5, 10, 15), np.uint8), 1, 2, 3)
            np.testing.assert_array_equal(
                result["channel"], data[0, 3:8, 2:12, 1:16])
    
    def test_02_05_resolution_and_offset(self):
        with make_hdf5((1, 10, 20, 30)) as (filename, data):
            classifier = HDF5Classifier(
                ["channel"],
                [(filename, 0, 0, 0, 60, 40, 10)],
                resolution=1)
            result = classifier.classify(
                np.zeros((5, 19, 15), np.uint8), 3, 2, 1)
            big_data = zoom(data[0, 1:6, 1:11, 1:9], (1, 2, 2), order=1)
            np.testing.assert_array_equal(result["channel"], 
                                          big_data[:, :-1, 1:])
    
    def test_02_07_axes(self):
        with make_hdf5((10, 20, 30, 1)) as (filename, data):
            classifier = HDF5Classifier(
                ["channel"],
                [(filename, 0, 0, 0, 30, 20, 10)],
                axes_indexes=(3, 0, 1, 2))
            result = classifier.classify(
                np.zeros((10, 20, 30), np.uint8), 0, 0, 0)
            self.assertEqual(len(result), 1)
            self.assertEqual(result.keys()[0], "channel")
            np.testing.assert_array_equal(result["channel"], data[:, :, :, 0])
    
    def test_03_01_multiple_hdf(self):
        with make_hdf5((1, 10, 20, 30)) as (filename1, data1):
            with make_hdf5((1, 10, 20, 30)) as (filename2, data2):
                classifier = HDF5Classifier(
                    ["channel"],
                    [(filename1, 0, 0, 0, 30, 20, 10),
                     (filename2, 0, 0, 10, 30, 20, 10)])
                result = classifier.classify(
                    np.zeros((10, 20, 30), np.uint8), 0, 0, 5)
                np.testing.assert_array_equal(
                    result["channel"][:5], data1[0, 5:])
                np.testing.assert_array_equal(
                    result["channel"][5:], data2[0, :5])
        
        