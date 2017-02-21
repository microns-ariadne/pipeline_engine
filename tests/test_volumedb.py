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
        self.db.register_dataset(task, 
                                 volume,
                                 "image")
        result = self.db.find_datasets_by_type_and_volume(
            "image", volume)

if __name__ == "__main__":
    unittest.main()