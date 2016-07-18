import contextlib
import unittest
import h5py
import numpy as np
import os
import shutil
import tempfile
from ariadne_microns_pipeline.targets.hdf5_target import HDF5VolumeTarget

class TestHDF5VolumeTarget(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.fd, self.path = tempfile.mkstemp(".h5", dir=self.tempdir)
        self.root, self.filename = os.path.split(self.path)
        self.pattern = os.path.splitext(self.filename)[0]
    
    def write_done_file(self, target):
        done_file = target.path
        with open(done_file, "w") as fd:
            fd.write("done")
        
    def tearDown(self):
        os.close(self.fd)
        shutil.rmtree(self.tempdir)
        
    def test_01_01_does_not_exist(self):
        t = HDF5VolumeTarget([self.root], "foo", "bar",
                             0, 0, 0, 30, 20, 10)
        self.assertFalse(t.exists())
        t = HDF5VolumeTarget([self.root], "foo", self.pattern,
                             0, 0, 0, 30, 20, 10)
        self.assertFalse(t.exists())
        with h5py.File(self.path, "w") as fd:
            fd.create_group("foo")
        self.assertFalse(t.exists())
    
    def test_01_02_exists(self):
        with h5py.File(self.path, "w") as fd:
            fd.create_dataset("foo", (10, 20, 30), np.uint8)
        t = HDF5VolumeTarget([self.root], "foo", self.pattern,
                             0, 0, 0, 30, 20, 10)
        self.write_done_file(t)
        self.assertTrue(t.exists())
    
    def test_02_01_read(self):
        r = np.random.RandomState(21)
        data = r.randint(low=0, high=255, size=(10, 20, 30)).astype(np.uint8)
        with h5py.File(self.path, "w") as fd:
            fd.create_dataset("foo", data=data)
            t = HDF5VolumeTarget([self.root], "foo", self.pattern,
                                 0, 0, 0, 30, 20, 10)
        self.write_done_file(t)
        np.testing.assert_array_equal(t.imread(), data)
    
    def test_02_03_write(self):
        r = np.random.RandomState(23)
        data = r.randint(low=0, high=255, size=(10, 20, 30)).astype(np.uint8)
        t = HDF5VolumeTarget([self.root], "foo", self.pattern,
                             1, 2, 3, 30, 20, 10)
        t.imwrite(data)
        self.assertTrue(os.path.isfile(t.path))
        with h5py.File(self.path, "r") as fd:
            ds = fd["foo"]
            np.testing.assert_array_equal(ds[:], data)
            self.assertEqual(ds.attrs["x"], 1)
            self.assertEqual(ds.attrs["y"], 2)
            self.assertEqual(ds.attrs["z"], 3)

if __name__ == "__main__":
    unittest.main()