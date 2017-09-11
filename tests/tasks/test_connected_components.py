import json
import numpy as np
import os
import shutil
import tempfile
import tifffile
import unittest
import uuid

from ariadne_microns_pipeline.targets.factory import TargetFactory
from ariadne_microns_pipeline.tasks \
     import AllConnectedComponentsTask, ConnectedComponentsTask
from ariadne_microns_pipeline.tasks.connected_components import \
     LogicalOperation, JoiningMethod
from ariadne_microns_pipeline.parameters import Volume
from ariadne_microns_pipeline.targets.volume_target import \
     DestVolumeReader, write_simple_loading_plan, DestVolumeReader
from ariadne_microns_pipeline.pipelines.pipeline import NP_DATASET


class TestConnectedComponents(unittest.TestCase):
    
    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.output_location = os.path.join(self.tempdir, "result.json")
    
    def tearDown(self):
        shutil.rmtree(self.tempdir)
        
    def make_input(self, labeling, x, y, z, overlap_volume):
        '''Make an input labeling for connected components
        
        :param labeling: a 3d integer array with labels for the volume
        :param volume: the volume represented by the labeling
        :returns: the loading plan path of the volume.
        '''
        dataset_name = uuid.uuid4().get_hex()
        fdtif, temptif = tempfile.mkstemp(suffix=".tif", dir=self.tempdir)
        fdlp, temploadingplan = tempfile.mkstemp(suffix="loading.plan",
                                                 dir=self.tempdir)
        x0 = overlap_volume.x - x
        x1 = overlap_volume.x1 - x
        y0 = overlap_volume.y - y
        y1 = overlap_volume.y1 - y
        z0 = overlap_volume.z - z
        z1 = overlap_volume.z1 - z
        tifffile.imsave(temptif, labeling[z0:z1, y0:y1, x0:x1])
        os.close(fdtif)
        assert os.path.isfile(temptif)
        write_simple_loading_plan(
             temploadingplan,
             temptif,
             overlap_volume,
             NP_DATASET, labeling.dtype.name)
        #
        # Add a fakey storage plan and write its done file
        #
        done_fd, done_path = tempfile.mkstemp(suffix=".storage.done",
                                              dir=self.tempdir)
        plan_path = os.path.splitext(done_path)[0] + ".plan"
        lp = json.load(open(temploadingplan))
        lp["dataset_done_files"] = [ plan_path ]
        json.dump(lp, open(temploadingplan, "w"))
        os.close(fdlp)
        lp["areas"] = np.bincount(labeling.flatten()).tolist()
        lp["labels"] = np.unique(labeling[labeling != 0]).tolist()
        json.dump(lp, open(done_path, "w"))
        os.close(done_fd)
        return DestVolumeReader(temploadingplan)
    
    def test_00_nothing(self):
        # Test a case where the intersection is in the background for both
        
        overlap_volume=Volume(9, 0, 0, 1, 10, 10)
        a = self.make_input(np.zeros((10, 10, 10), np.uint16), 0, 0, 0,
                            overlap_volume)
        b = self.make_input(np.zeros((10, 10, 10), np.uint16), 9, 0, 0,
                            overlap_volume)
        
        task = ConnectedComponentsTask(
            volume1=a.volume,
            cutout_loading_plan1_path=a.loading_plan_path,
            segmentation_loading_plan1_path=a.loading_plan_path,
            volume2=b.volume,
            cutout_loading_plan2_path=b.loading_plan_path,
            segmentation_loading_plan2_path=b.loading_plan_path,
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
        va = np.zeros((10, 10, 10), np.uint16)
        va[1, 1, 1] = 1
        va[3, 3, 3] = 2
        vb = np.ones((10, 10, 10), np.uint16)
        
        overlap_volume=Volume(9, 0, 0, 1, 10, 10)
        a = self.make_input(va, 0, 0, 0, overlap_volume)
        b = self.make_input(vb, 9, 0, 0, overlap_volume)
        
        task = ConnectedComponentsTask(
            volume1=a.volume,
            cutout_loading_plan1_path=a.loading_plan_path,
            segmentation_loading_plan1_path=a.loading_plan_path,
            volume2=b.volume,
            cutout_loading_plan2_path=b.loading_plan_path,
            segmentation_loading_plan2_path=b.loading_plan_path,
            output_location=self.output_location)
        task.run()
        result = json.load(task.output().open("r"))
        self.assertSetEqual(set(result["1"]["labels"]), set([1, 2]))
        self.assertSetEqual(set(result["2"]["labels"]), set([1]))
        self.assertEqual(len(result["connections"]), 0)

    def test_02_matches(self):
        va = np.zeros((10, 10, 10), np.uint16)
        va[1, 1, 9] = 1
        va[3, 4, 9] = 2
        va[5, 5, 9] = 3
        vb = np.zeros((10, 10, 10), np.uint16)
        vb[1, 1, 0] = 3
        vb[3, 4, 0] = 4
        
        overlap_volume=Volume(9, 0, 0, 1, 10, 10)
        a = self.make_input(va, 0, 0, 0, overlap_volume)
        b = self.make_input(vb, 9, 0, 0, overlap_volume)
        
        task = ConnectedComponentsTask(
            volume1=a.volume,
            cutout_loading_plan1_path=a.loading_plan_path,
            segmentation_loading_plan1_path=a.loading_plan_path,
            volume2=b.volume,
            cutout_loading_plan2_path=b.loading_plan_path,
            segmentation_loading_plan2_path=b.loading_plan_path,
            output_location=self.output_location)
        task.run()
        result = json.load(task.output().open("r"))
        connections = result["connections"]
        self.assertEqual(len(connections), 2)
        self.assertTrue(any([a == 1 and b == 3 for a, b in connections]))
        self.assertTrue(any([a == 2 and b == 4 for a, b in connections]))
    
    def test_03_exclude1(self):
        va = np.zeros((10, 10, 10), np.uint16)
        va[1, 1, 9] = 1
        va[3, 4, 9] = 2
        va[5, 5, 9] = 3
        vb = np.zeros((10, 10, 10), np.uint16)
        vb[1, 1, 0] = 3
        vb[3, 4, 0] = 4
        
        overlap_volume=Volume(9, 0, 0, 1, 10, 10)
        a = self.make_input(va, 0, 0, 0, overlap_volume)
        b = self.make_input(vb, 9, 0, 0, overlap_volume)
        
        task = ConnectedComponentsTask(
            volume1=a.volume,
            cutout_loading_plan1_path=a.loading_plan_path,
            segmentation_loading_plan1_path=a.loading_plan_path,
            volume2=b.volume,
            cutout_loading_plan2_path=b.loading_plan_path,
            segmentation_loading_plan2_path=b.loading_plan_path,
            exclude1=[1],
            output_location=self.output_location)
        task.run()
        result = json.load(task.output().open("r"))
        connections = result["connections"]
        self.assertEqual(len(connections), 1)
        self.assertFalse(any([a == 1 and b == 3 for a, b in connections]))
        self.assertTrue(any([a == 2 and b == 4 for a, b in connections]))

    def test_04_exclude2(self):
        va = np.zeros((10, 10, 10), np.uint16)
        va[1, 1, 9] = 1
        va[3, 4, 9] = 2
        va[5, 5, 9] = 3
        vb = np.zeros((10, 10, 10), np.uint16)
        vb[1, 1, 0] = 3
        vb[3, 4, 0] = 4
        
        overlap_volume=Volume(9, 0, 0, 1, 10, 10)
        a = self.make_input(va, 0, 0, 0, overlap_volume)
        b = self.make_input(vb, 9, 0, 0, overlap_volume)
        
        task = ConnectedComponentsTask(
            volume1=a.volume,
            cutout_loading_plan1_path=a.loading_plan_path,
            segmentation_loading_plan1_path=a.loading_plan_path,
            volume2=b.volume,
            cutout_loading_plan2_path=b.loading_plan_path,
            segmentation_loading_plan2_path=b.loading_plan_path,
            exclude2=[3],
            output_location=self.output_location)
        task.run()
        result = json.load(task.output().open("r"))
        connections = result["connections"]
        self.assertEqual(len(connections), 1)
        self.assertFalse(any([a == 1 and b == 3 for a, b in connections]))
        self.assertTrue(any([a == 2 and b == 4 for a, b in connections]))
    
    def test_05_min_overlap_percent(self):
        #
        # Test the minimum allowed overlap percent
        #
        # 49 % should be OK, 51 % should be a fail
        #
        va = np.zeros((10, 10, 10), np.uint16)
        va[1, 1, 9] = 1
        va[3, 4, 9] = 2
        va[3, 5, 9] = 2
        va[3, 6, 9] = 1
        vb = np.zeros((10, 10, 10), np.uint16)
        vb[1, 1, 0] = 3
        vb[3, 4, 0] = 4
        vb[3, 5, 0] = 3
        vb[3, 6, 0] = 4
        
        overlap_volume=Volume(9, 0, 0, 1, 10, 10)
        a = self.make_input(va, 0, 0, 0, overlap_volume)
        b = self.make_input(vb, 9, 0, 0, overlap_volume)
        
        task = ConnectedComponentsTask(
            volume1=a.volume,
            cutout_loading_plan1_path=a.loading_plan_path,
            segmentation_loading_plan1_path=a.loading_plan_path,
            volume2=b.volume,
            cutout_loading_plan2_path=b.loading_plan_path,
            segmentation_loading_plan2_path=b.loading_plan_path,
            output_location=self.output_location,
            min_overlap_percent=49)
        task.run()
        result = json.load(task.output().open("r"))
        connections = result["connections"]
        self.assertEqual(len(connections), 4)
        self.assertTrue(
            any([id1 == 1 and id2 == 3 for id1, id2 in connections]))
        self.assertTrue(
            any([id1 == 2 and id2 == 3 for id1, id2 in connections]))
        self.assertTrue(
            any([id1 == 1 and id2 == 4 for id1, id2 in connections]))
        self.assertTrue(
            any([id1 == 2 and id2 == 4 for id1, id2 in connections]))
        task = ConnectedComponentsTask(
            volume1=a.volume,
            cutout_loading_plan1_path=a.loading_plan_path,
            segmentation_loading_plan1_path=a.loading_plan_path,
            volume2=b.volume,
            cutout_loading_plan2_path=b.loading_plan_path,
            segmentation_loading_plan2_path=b.loading_plan_path,
            output_location=self.output_location,
            min_overlap_percent=51)
        task.run()
        result = json.load(task.output().open("r"))
        connections = result["connections"]
        self.assertEqual(len(connections), 0)
    
    def test_06_OR_AND(self):
        #
        # Test that OR succeeds when smaller overlaps larger but AND fails
        #
        # at 50% 1 totally overlaps 3 and 3 half-overlaps 1
        #        2 overlaps 3 by 1/3 and 3 overlaps 2 by 1/2
        #        1 does not overlap 4
        #        2 overlaps 4 by 2/3 and 4 overlaps 2 completely
        #
        # OR: 1 overlaps 3
        #     2 overlaps 3
        #     2 overlaps 4
        # AND: 1 overlaps 3
        #     2 overlaps 4
        #
        va = np.zeros((10, 10, 10), np.uint16)
        va[1, 1, 9] = 1
        va[3, 4, 9] = 2
        va[3, 5, 9] = 2
        va[3, 6, 9] = 2
        vb = np.zeros((10, 10, 10), np.uint16)
        vb[1, 1, 0] = 3
        vb[3, 4, 0] = 4
        vb[3, 5, 0] = 3
        vb[3, 6, 0] = 4
        
        overlap_volume=Volume(9, 0, 0, 1, 10, 10)
        a = self.make_input(va, 0, 0, 0, overlap_volume)
        b = self.make_input(vb, 9, 0, 0, overlap_volume)
        
        task = ConnectedComponentsTask(
            volume1=a.volume,
            cutout_loading_plan1_path=a.loading_plan_path,
            segmentation_loading_plan1_path=a.loading_plan_path,
            volume2=b.volume,
            cutout_loading_plan2_path=b.loading_plan_path,
            segmentation_loading_plan2_path=b.loading_plan_path,
            output_location=self.output_location,
            operation=LogicalOperation.OR)
        task.run()
        result = json.load(task.output().open("r"))
        connections = result["connections"]
        self.assertEqual(len(connections), 3)
        self.assertTrue(
            any([id1 == 1 and id2 == 3 for id1, id2 in connections]))
        self.assertTrue(
            any([id1 == 2 and id2 == 3 for id1, id2 in connections]))
        self.assertTrue(
            any([id1 == 2 and id2 == 4 for id1, id2 in connections]))
        task = ConnectedComponentsTask(
            volume1=a.volume,
            cutout_loading_plan1_path=a.loading_plan_path,
            segmentation_loading_plan1_path=a.loading_plan_path,
            volume2=b.volume,
            cutout_loading_plan2_path=b.loading_plan_path,
            segmentation_loading_plan2_path=b.loading_plan_path,
            output_location=self.output_location,
            operation=LogicalOperation.AND)
        task.run()
        result = json.load(task.output().open("r"))
        connections = result["connections"]
        self.assertEqual(len(connections), 2)
        self.assertTrue(
            any([id1 == 1 and id2 == 3 for id1, id2 in connections]))
        self.assertTrue(
            any([id1 == 2 and id2 == 4 for id1, id2 in connections]))
        
        

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
        for name, volume, labels, idx in (("1", volume_a, labels_a, 0), 
                                          ("2", volume_b, labels_b, 1)):
            d[name] = dict(x=volume.x,
                           y=volume.y,
                           z=volume.z,
                           width=volume.width,
                           height=volume.height,
                           depth=volume.depth,
                           labels=labels,
                           location="foo.loading.plan")
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