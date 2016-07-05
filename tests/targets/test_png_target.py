from cv2 import imread, imwrite
import json
import numpy as np
import os
import unittest
import tempfile
from ariadne_microns_pipeline.targets.png_volume_target import PngVolumeTarget

class TestPngVolumeTarget(unittest.TestCase):
    
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.dataset_name = "foo"
        self.pattern = "{x:04d}_{y:04d}_{z:04d}"
        self.touchfile = os.path.join(self.root, self.dataset_name,
                                      "0001_0002_0003.png.done")
    
    def tearDown(self):
        for root, dirnames, filenames in os.walk(self.root, topdown=False):
            for filename in filenames:
                os.remove(os.path.join(root, filename))
            for dirname in dirnames:
                os.rmdir(os.path.join(root, dirname))
        os.rmdir(self.root)
    
    def make_dataset(self, r):
        self.data = r.randint(0, 255, (10, 20, 30)).astype(np.uint8)
        os.mkdir(os.path.join(self.root, "foo"))
        filenames = []
        d = dict(dimensions=(10, 20, 30),
                 dtype="u1",
                 x=1,
                 y=2,
                 z=3,
                 filenames=filenames)
        with open(self.touchfile, "w") as fd:
            for z in range(len(self.data)):
                path = os.path.join(
                    self.root, "foo", "0001_0002_%04d.Png" % (z+3))
                filenames.append(path)
                imwrite(path, self.data[z])
            json.dump(d, fd)
            
    def test_01_01_does_not_exist(self):
        t = PngVolumeTarget([self.root], self.dataset_name, self.pattern,
                            1, 2, 3, 30, 20, 10)
        self.assertFalse(t.exists())
        os.mkdir(os.path.join(self.root, "foo"))
        self.assertFalse(t.exists())
    
    def test_01_02_exists(self):
        r = np.random.RandomState(12)
        self.make_dataset(r)
        t = PngVolumeTarget([self.root], self.dataset_name, self.pattern,
                            1, 2, 3, 30, 20, 10)
        self.assertTrue(t.exists())
        
    def test_02_01_imread(self):
        r = np.random.RandomState(21)
        self.make_dataset(r)
        t = PngVolumeTarget([self.root], self.dataset_name, self.pattern,
                            1, 2, 3, 30, 20, 10)
        result = t.imread()
        np.testing.assert_array_equal(result, self.data)

    def test_03_01_imwrite(self):
        r = np.random.RandomState(31)
        data = r.randint(0, 255, (10, 20, 30)).astype(np.uint8)
        t = PngVolumeTarget([self.root], self.dataset_name, self.pattern,
                            1, 2, 3, 30, 20, 10)
        t.imwrite(data)
        self.assertTrue(os.path.exists(self.touchfile))
        with open(self.touchfile, "r") as fd:
            d = json.load(fd)
        self.assertEqual(len(d["filenames"]), 10)
        for z, path in enumerate(d["filenames"]):
            plane = imread(path, 2)
            np.testing.assert_array_equal(plane, data[z])

if __name__=="__main__":
    unittest.main()