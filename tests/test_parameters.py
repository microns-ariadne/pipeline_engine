import json
import unittest
from ariadne_microns_pipeline.parameters import *

class TestMultiVolumeParameter(unittest.TestCase):
    
    def test_01_01_parse(self):
        volumes = [(1,2,3,4,5,6), (7,8,9,10,11,12)]
        locations = [(["/tmp/001", "/tmp/002"], "foo", "bar"),
                     (["/home/user"], "bar", "baz")]
        l = []
        for volume, location in zip(volumes, locations):
            d = dict(volume=dict(x=volume[0],
                                 y=volume[1],
                                 z=volume[2],
                                 width=volume[3],
                                 height=volume[4],
                                 depth=volume[5]),
                     location=dict(roots=location[0],
                                   dataset_name=location[1],
                                   pattern=location[2]))
            l.append(d)
        s = json.dumps(l)
        result = MultiVolumeParameter().parse(s)
        for output, expected in zip(result, l):
            location = output["location"]
            volume = output["volume"]
            elocation = expected["location"]
            evolume = expected["volume"]
            self.assertEqual(volume.x, evolume["x"])
            self.assertEqual(volume.y, evolume["y"])
            self.assertEqual(volume.z, evolume["z"])
            self.assertEqual(volume.width, evolume["width"])
            self.assertEqual(volume.height, evolume["height"])
            self.assertEqual(volume.depth, evolume["depth"])
            self.assertSequenceEqual(location.roots, elocation["roots"])
            self.assertEqual(location.dataset_name, elocation["dataset_name"])
            self.assertEqual(location.pattern, elocation["pattern"])
    
    
    def test_02_01_serialize(self):
        x = MultiVolumeParameter().serialize(
            [dict(volume=Volume(1, 2, 3, 4, 5, 6),
                  location=DatasetLocation(
                      ["/tmp/001", "/tmp/002"], "foo", "bar"))])
        y = json.loads(x)
        self.assertEqual(len(y), 1)
        volume = y[0]["volume"]
        self.assertDictEqual(
            volume, dict(x=1, y=2, z=3, width=4, height=5, depth=6))
        location = y[0]["location"]
        self.assertSequenceEqual(
            location["roots"], ("/tmp/001", "/tmp/002"))
        self.assertEqual(location["dataset_name"], "foo")
        self.assertEqual(location["pattern"], "bar")

if __name__ == "__main__":
    unittest.main()