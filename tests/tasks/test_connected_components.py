import json
import numpy as np
import os
import shutil
import tempfile
import unittest
import uuid

from ariadne_microns_pipeline.targets.factory import TargetFactory
from ariadne_microns_pipeline.tasks \
     import AllConnectedComponentsTask, ConnectedComponentsTask
from ariadne_microns_pipeline.parameters import Volume, DatasetLocation


class TestConnectedComponents(unittest.TestCase):
    
    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.output_location = os.path.join(self.tempdir, "result.json")
    
    def tearDown(self):
        shutil.rmtree(self.tempdir)
        
    def make_input(self, labeling, x, y, z):
        '''Make an input labeling for connected components
        
        :param labeling: a 3d integer array with labels for the volume
        :param volume: the volume represented by the labeling
        :returns: a volume target containing the labeling.
        '''
        dataset_name = uuid.uuid4().get_hex()
        pattern = "{x:09d}_{y:09d}_{z:09d}_"
        volume = Volume(x, y, z, 
                        width=labeling.shape[2],
                        height=labeling.shape[1],
                        depth=labeling.shape[0])
        target = TargetFactory().get_volume_target(
            DatasetLocation([self.tempdir], dataset_name, pattern),
            volume)
        target.imwrite(labeling)
        return target
    
    def test_00_nothing(self):
        # Test a case where the intersection is in the background for both
        
        a = self.make_input(np.zeros((10, 10, 10), int), 0, 0, 0)
        b = self.make_input(np.zeros((10, 10, 10), int), 9, 0, 0)
        
        task = ConnectedComponentsTask(
            volume1=a.volume,
            location1=a.dataset_location,
            volume2=b.volume,
            location2=b.dataset_location,
            overlap_volume=Volume(9, 0, 0, 1, 10, 10),
            output_location=self.output_location)
        task.run()
        result = json.load(task.output().open("r"))
        #
        # Spot checks on volumes
        #
        self.assertEqual(len(result["1"]["labels"]), 0)
        self.assertEqual(len(result["2"]["labels"]), 0)
        self.assertEqual(result["2"]["x"], 9)
        self.assertEqual(result["1"]["y"], 0)
        self.assertEqual(result["1"]["z"], 0)
        self.assertEqual(result["overlap"]["x"], 9)
        #
        # and most importantly, the connections
        #
        self.assertEqual(len(result["connections"]), 0)
    
    def test_01_nothing_that_matches(self):
        va = np.zeros((10, 10, 10), int)
        va[1, 1, 1] = 1
        va[3, 3, 3] = 2
        vb = np.ones((10, 10, 10), int)
        
        a = self.make_input(va, 0, 0, 0)
        b = self.make_input(vb, 9, 0, 0)
        
        task = ConnectedComponentsTask(
            volume1=a.volume,
            location1=a.dataset_location,
            volume2=b.volume,
            location2=b.dataset_location,
            overlap_volume=Volume(9, 0, 0, 1, 10, 10),
            output_location=self.output_location)
        task.run()
        result = json.load(task.output().open("r"))
        self.assertSetEqual(set(result["1"]["labels"]), set([1, 2]))
        self.assertSetEqual(set(result["2"]["labels"]), set([1]))
        self.assertEqual(len(result["connections"]), 0)

    def test_02_matches(self):
        va = np.zeros((10, 10, 10), int)
        va[1, 1, 9] = 1
        va[3, 4, 9] = 2
        va[5, 5, 9] = 3
        vb = np.zeros((10, 10, 10), int)
        vb[1, 1, 0] = 3
        vb[3, 4, 0] = 4
        
        a = self.make_input(va, 0, 0, 0)
        b = self.make_input(vb, 9, 0, 0)
        
        task = ConnectedComponentsTask(
            volume1=a.volume,
            location1=a.dataset_location,
            volume2=b.volume,
            location2=b.dataset_location,
            overlap_volume=Volume(9, 0, 0, 1, 10, 10),
            output_location=self.output_location)
        task.run()
        result = json.load(task.output().open("r"))
        connections = result["connections"]
        self.assertEqual(len(connections), 2)
        self.assertTrue(any([a == 1 and b == 3 for a, b in connections]))
        self.assertTrue(any([a == 2 and b == 4 for a, b in connections]))


class TestAllConnectedComponents(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.output_location = os.path.join(self.tempdir, "result.json")
        
    def tearDown(self):
        shutil.rmtree(self.tempdir)
    
    def make_input(self, labels_a, volume_a, labels_b, volume_b, connections):
        '''Make a fake connected-components JSON file
        
        :param labels_a: a list of the labels that are in the "a" volume
        :param volume_a: the "a" volume details (a Volume object)
        :param labels_b: a list of the labels that are in the "b" volume
        :param volume_b: the "b" volume details
        :param connection: a list of 2-tuples giving the correspondences
        between the labels in "a" and "b"
        :returns: the file name.
        '''
        d = {}
        for name, volume, labels in (("1", volume_a, labels_a), 
                                     ("2", volume_b, labels_b)):
            d[name] = dict(x=volume.x,
                           y=volume.y,
                           z=volume.z,
                           width=volume.width,
                           height=volume.height,
                           depth=volume.depth,
                           labels=labels)
        d["connections"] = connections
        fd, filename = tempfile.mkstemp(suffix=".json", dir=self.tempdir)
        os.close(fd)
        with open(filename, "w") as f:
            json.dump(d, f)
        return filename
    
    def test_00_nothing(self):
        # Create a file with no overlaps
        f1 = self.make_input([1], Volume(0, 0, 0, 10, 10, 10),
                             [1], Volume(10, 0, 0, 10, 10, 10),
                             [])
        f2 = self.make_input([1], Volume(0, 0, 0, 10, 10, 10),
                             [1, 2], Volume(0, 10, 0, 10, 10, 10), [])
        task = AllConnectedComponentsTask([f1, f2], self.output_location)
        task.run()
        result = json.load(task.output().open("r"))
        self.assertEqual(result["count"], 4)
        for volume, labeling in result["volumes"]:
            if volume["x"] == 0 and volume["y"] == 0 and volume["z"] == 0:
                self.assertSequenceEqual(labeling[0], [1, 1])
            elif volume["x"] == 10 and volume["y"] == 0 and volume["z"] == 0:
                self.assertSequenceEqual(labeling[0], [1, 2])
            else:
                self.assertSequenceEqual(labeling[0], (1, 3))
                self.assertSequenceEqual(labeling[1], (2, 4))
    
    def test_01_something(self):
        #
        # 0, 0, 0 component 1 <-> 0, 10, 0 component 1
        # 0, 0, 0 component 2 <-> 10, 0, 0 component 1
        #
        f1 = self.make_input([1, 2], Volume(0, 0, 0, 10, 10, 10),
                             [1, 2, 3], Volume(10, 0, 0, 10, 10, 10),
                             [[2, 1]])
        f2 = self.make_input([1, 2], Volume(0, 0, 0, 10, 10, 10),
                             [1, 2], Volume(0, 10, 0, 10, 10, 10), 
                             [[1, 1]])
        expected = { (0, 0, 0):[[1, 1], [2, 2]],
                     (10, 0, 0): [[1, 2], [2, 3], [3, 4]],
                     (0, 10, 0): [[1, 1], [2, 5]] }
        task = AllConnectedComponentsTask([f1, f2], self.output_location)
        task.run()
        result = json.load(task.output().open("r"))
        self.assertEqual(result["count"], 5)
        self.assertEqual(len(result["volumes"]), 3)
        for volume, labeling in result["volumes"]:
            key = (volume["x"], volume["y"], volume["z"])
            self.assertIn(key, expected)
            self.assertSequenceEqual(expected[key], labeling)
            
if __name__=="__main__":
    unittest.main()