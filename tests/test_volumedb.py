import luigi
import unittest
import os
import tempfile

from ariadne_microns_pipeline.volumedb import VolumeDB, Persistence
from ariadne_microns_pipeline.parameters import Volume

class DummyTask(luigi.Task):
    my_parameter = luigi.Parameter()

class TestVolumeDB(unittest.TestCase):
    def setUp(self):
        self.db = VolumeDB("sqlite:///:memory:", "w")
    
    def test_00_00_nothing(self):
        #
        # Make sure we can run on an empty set
        #
        self.db.compute_subvolumes()
    
    def test_01_01_register_task(self):
        task = DummyTask(my_parameter="foo")
        task_obj = self.db.get_or_create_task(task)
        self.assertEqual(task_obj.luigi_id, task.task_id)
        self.assertEqual(task_obj.task_class, task.task_family)
        self.assertEqual(len(task_obj.parameters), 1)
        self.assertEqual(task_obj.parameters[0].name, "my_parameter")
        self.assertEqual(task_obj.parameters[0].value, "foo")
        
    def test_01_02_register_dataset_type(self):
        self.db.register_dataset_type("image", Persistence.Permanent,
                                      "Raw image data volume")
        result = self.db.engine.execute("select * from dataset_types")\
            .fetchall()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "image")
        self.assertEqual(result[0]["persistence"], Persistence.Permanent.name)
        self.assertEqual(result[0]["doc"], "Raw image data volume")
    
    def test_01_02_01_get_dataset_type(self):
        self.db.register_dataset_type("image", Persistence.Permanent,
                                          "Raw image data volume")
        dataset_type = self.db.get_dataset_type("image")
        self.assertEqual(dataset_type.name, "image")
        self.assertEqual(dataset_type.persistence, Persistence.Permanent)
        self.assertEqual(dataset_type.doc, "Raw image data volume")
    
    def test_01_03_register_dataset(self):
        task = DummyTask(my_parameter="foo")
        self.db.register_dataset_type("image", Persistence.Permanent,
                                          "Raw image data volume")
        volume = Volume(44, 22, 14, 1024, 2048, 117)
        self.db.register_dataset(task, "image", volume)
        result = self.db.find_datasets_by_type_and_volume(
            "image", volume)
        self.assertEqual(len(result), 1)
        dataset = result[0]
        self.assertEqual(dataset.task.luigi_id, task.task_id)
        self.assertEqual(dataset.volume.x0, 44)
    
    def test_01_04_dont_find_datasets(self):
        task = DummyTask(my_parameter="foo")
        self.db.register_dataset_type("image", Persistence.Permanent,
                                          "Raw image data volume")
        volume = Volume(44, 22, 14, 1024, 2048, 117)
        self.db.register_dataset(task, "image", volume)
        #
        # Check every corner case
        #
        for volume in (
            Volume(1068, 22, 14, 1024, 2048, 117),
            Volume(44, 2070, 14, 1024, 2048, 117),
            Volume(44, 22, 117+14, 1024, 2048, 117),
            Volume(0, 22, 14, 44, 2048, 117),
            Volume(44, 0, 14, 1024, 22, 117),
            Volume(44, 22, 0, 1024, 2048, 14)):
            result = self.db.find_datasets_by_type_and_volume(
                "image", volume)
            self.assertEqual(len(result), 0)
    
    def test_01_05_find_overlapping_datasets(self):
        task = DummyTask(my_parameter="foo")
        self.db.register_dataset_type("image", Persistence.Permanent,
                                          "Raw image data volume")
        volume = Volume(44, 22, 14, 1024, 2048, 117)
        self.db.register_dataset(task, "image", volume)
        #
        # Check every corner case
        #
        for volume in (
                Volume(1067, 22, 14, 1024, 2048, 117),
                Volume(44, 2069, 14, 1024, 2048, 117),
                Volume(44, 22, 117+14-1, 1024, 2048, 117),
                Volume(0, 22, 14, 45, 2048, 117),
                Volume(44, 0, 14, 1024, 23, 117),
                Volume(44, 22, 0, 1024, 2048, 15)):
            result = self.db.find_datasets_by_type_and_volume(
                    "image", volume)
            self.assertEqual(len(result), 1)
    
    def test_01_06_dont_find_dataset_of_wrong_type(self):
        task = DummyTask(my_parameter="foo")
        self.db.register_dataset_type("image", Persistence.Permanent,
                                          "Raw image data volume")
        self.db.register_dataset_type("segmentation", Persistence.Permanent)
        volume = Volume(44, 22, 14, 1024, 2048, 117)
        self.db.register_dataset(task, "image", volume)
        result = self.db.find_datasets_by_type_and_volume(
                "segmentation", volume)
        self.assertEqual(len(result), 0)
        
    
    def test_01_06_register_dataset_dependent(self):
        task = DummyTask(my_parameter="foo")
        self.db.register_dataset_type("image", Persistence.Permanent,
                                          "Raw image data volume")
        volume = Volume(44, 22, 14, 1024, 2048, 117)
        self.db.register_dataset(task, "image", volume)
        dependent_task = DummyTask(my_parameter="bar")
        self.db.register_dataset_dependent(dependent_task, "image", volume)
        dependencies = self.db.get_dependencies(dependent_task)
        self.assertEqual(len(dependencies), 1)
        self.assertEqual(dependencies[0], task.task_id)
    
    def test_01_07_register_dataset_dependent_multiple(self):
        #
        # Register a dataset dependent against 3 tasks of which only two
        # overlap
        #
        task1 = DummyTask(my_parameter="foo")
        task2 = DummyTask(my_parameter="bar")
        task3 = DummyTask(my_parameter="baz")
        self.db.register_dataset_type("image", Persistence.Permanent,
                                          "Raw image data volume")
        volume = Volume(0, 0, 0, 1024, 1024, 128)
        self.db.register_dataset(task1, "image", volume)
        volume = Volume(1024, 0, 0, 1024, 1024, 128)
        self.db.register_dataset(task2, "image", volume)
        volume = Volume(2048, 0, 0, 1024, 1024, 128)
        self.db.register_dataset(task3, "image", volume)
        dependent_task = DummyTask(my_parameter="blech")
        volume = Volume(0, 0, 0, 1025, 1024, 128)
        self.db.register_dataset_dependent(dependent_task, "image", volume)
        dependencies = self.db.get_dependencies(dependent_task)
        self.assertEqual(len(dependencies), 2)
        for dependency in dependencies:
            self.assertIn(dependency, (task1.task_id, task2.task_id))
    
    def test_01_08_register_dataset_dependent_specific(self):
        #
        # For the case where the dependent dataset knows its source
        # (e.g. if the sources overlap), choose the specified source
        #
        task1 = DummyTask(my_parameter="left")
        task2 = DummyTask(my_parameter="right")
        dependent_task = DummyTask(my_parameter="dependent")
        self.db.register_dataset_type("image", Persistence.Permanent,
                                          "Raw image data volume")
        volume = Volume(0, 0, 0, 1024, 1024, 128)
        self.db.register_dataset(task1, "image", volume)
        volume = Volume(512, 0, 0, 1024, 1024, 128)
        self.db.register_dataset(task2, "image", volume)
        volume = Volume(512, 0, 0, 512, 1024, 128)
        self.db.register_dataset_dependent(dependent_task, "image", volume,
                                           src_task=task2)
        dependencies = self.db.get_dependencies(dependent_task)
        self.assertEqual(len(dependencies), 1)
        self.assertEqual(dependencies[0], task2.task_id)
        
    def test_02_01_simple_compute_subvolumes(self):
        task = DummyTask(my_parameter="foo")
        self.db.register_dataset_type("image", Persistence.Permanent,
                                          "Raw image data volume")
        self.db.set_target_dir("/tmp/foo")
        self.db.set_temp_dir("/tmp/bar")
        volume = Volume(44, 22, 14, 1024, 2048, 117)
        self.db.register_dataset(task, "image", volume)
        dependent_task = DummyTask(my_parameter="bar")
        self.db.register_dataset_dependent(dependent_task, "image", volume)
        self.db.compute_subvolumes()
        result = self.db.get_subvolume_locations(dependent_task, "image")
        self.assertEqual(len(result), 1)
        location, volume = result[0]
        self.assertTrue(location.startswith("/tmp/foo/44/22/14/image-"))
        self.assertEqual(volume.x, 44)
    
    def test_02_02_complex_compute_subvolumes(self):
        #
        # When computing segmentation block joins, you chop thin slices
        # from the x, y and z edges and also use the entire segmentation
        #
        src_task = DummyTask(my_parameter="foo")
        dataset_name = "neuroproof"
        self.db.register_dataset_type(dataset_name, Persistence.Permanent,
                                          "Raw image data volume")
        self.db.set_target_dir("/tmp/foo")
        self.db.set_temp_dir("/tmp/bar")
        volume = Volume(2048, 2048, 256, 1024, 1024, 128)
        self.db.register_dataset(src_task, dataset_name, volume)
        #
        # The destination task that uses the entire volume
        #
        dest_task = DummyTask(my_parameter="destination")
        self.db.register_dataset_dependent(dest_task, dataset_name, volume)
        #
        # The six overlaps
        #
        # .-.-.----.-.-.
        # | | |  1 | | |
        # .-.-.----.-.-.
        # | | |    | | |
        # | |2|    |3| |
        # | | |    | | |
        # .-.-.----.-.-.
        # | | |  4 | | |
        # .-.-.----.-.-.
        #
        overlap_x0_task = DummyTask(my_parameter="overlap-x0")
        volume_x0 = Volume(2048+20, 2048+30, 256+30, 10, 1024-60, 128-60)
        self.db.register_dataset_dependent(
            overlap_x0_task, dataset_name, volume_x0)
        
        overlap_x1_task = DummyTask(my_parameter="overlap-x1")
        volume_x1 = Volume(2048+1024-30, 2048+30, 256+30, 10, 1024-60, 128-60)
        self.db.register_dataset_dependent(
            overlap_x1_task, dataset_name, volume_x1)
        
        overlap_y0_task = DummyTask(my_parameter="overlap-y0")
        volume_y0 = Volume(2048+30, 2048+20, 256+30, 1024-60, 10, 128-60)
        self.db.register_dataset_dependent(
            overlap_y0_task, dataset_name, volume_y0)
    
        overlap_y1_task = DummyTask(my_parameter="overlap-y1")
        volume_y1 = Volume(2048+30, 2048+1024-30, 256+30, 1024-60, 10, 128-60)
        self.db.register_dataset_dependent(
            overlap_y1_task, dataset_name, volume_y1)

        overlap_z0_task = DummyTask(my_parameter="overlap-z0")
        volume_z0 = Volume(2048+30, 2048+30, 256+20, 1024-60, 1024-60, 10)
        self.db.register_dataset_dependent(
            overlap_z0_task, dataset_name, volume_z0)
    
        overlap_z1_task = DummyTask(my_parameter="overlap-z1")
        volume_z1 = Volume(2048+30, 2048+30, 256+128-30, 1024-60, 1024-60, 10)
        self.db.register_dataset_dependent(
            overlap_z1_task, dataset_name, volume_z1)
        #
        # Compute the subvolumes
        #
        self.db.compute_subvolumes()
        #
        # Check the destination task
        #
        result = self.db.get_subvolume_locations(dest_task, dataset_name)
        #
        # There should be 5 x 5 x 5 = 125 (!!!) volumes. The curse of
        # dimensionality 
        #
        self.assertEqual(len(result), 125)
        for x0, x1 in ((2048, 2048+20), (2048+20, 2048+30),
                       (2048+30, 2048+1024-30),
                       (2048+1024-30, 2048+1024-20), (2048+1024-20, 2048+1024)):
            for y0, y1 in (
                (2048, 2048+20), (2048+20, 2048+30),
                (2048+30, 2048+1024-30),
                (2048+1024-30, 2048+1024-20), (2048+1024-20, 2048+1024)):
                for z0, z1 in (
                    (256, 256+20), (256+20, 256+30), (256+30, 256 + 128 - 30),
                    (256+128-30, 256+128-20), (256+128-20, 256+128)):
                    for location, volume in result:
                        if volume.x == x0 and volume.x1 == x1 and \
                           volume.y == y0 and volume.y1 == y1 and \
                           volume.z == z0 and volume.z1 == z1:
                            expected = "/tmp/foo/%d/%d/%d/%s-" % \
                                (x0, y0, z0, dataset_name)
                            self.assertTrue(location.startswith(expected))
                            break
                    else:
                        self.fail()
        #
        # The other six, they should only have one slice
        #
        for task, expected_volume in ((overlap_x0_task, volume_x0),
                                      (overlap_x1_task, volume_x1),
                                      (overlap_y0_task, volume_y0),
                                      (overlap_y1_task, volume_y1),
                                      (overlap_z0_task, volume_z0),
                                      (overlap_z1_task, volume_z1)):
            result = self.db.get_subvolume_locations(task, dataset_name)
            self.assertEqual(len(result), 1)
            location, volume = result[0]
            self.assertEqual(
                location, 
                "/tmp/foo/%d/%d/%d" % (expected_volume.x, 
                                       expected_volume.y, 
                                       expected_volume.z))
            self.assertEqual(volume.x, expected_volume.x)
            self.assertEqual(volume.x1, expected_volume.x1)
            self.assertEqual(volume.y, expected_volume.y)
            self.assertEqual(volume.y1, expected_volume.y1)
            self.assertEqual(volume.z, expected_volume.z)
            self.assertEqual(volume.z1, expected_volume.z1)
            
if __name__ == "__main__":
    unittest.main()