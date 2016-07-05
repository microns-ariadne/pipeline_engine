import unittest
import numpy as np
from cv2 import imencode
from ariadne_microns_pipeline.targets.butterfly_target import ButterflyTarget
from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler
from urllib import unquote
import traceback
from threading import Thread

MY_EXPT = "myexperiment"
MY_SAMPLE = "mysample"
MY_DATASET = "mydataset"
MY_CHANNEL = "mychannel"
MY_X = 10
MY_Y = 11
MY_Z = 12

class MockButterflyRequestHandler(BaseHTTPRequestHandler):
    '''This mocks a Butterfly server'''
    
    def do_GET(self):
        try:
            assert self.path.startswith("/api/data")
            qparam_string = unquote(self.path.split("?")[-1])
            qparams = dict([_.split("=") for _ in qparam_string.split("&")])
            assert qparams["experiment"] == MY_EXPT
            assert qparams["sample"] == MY_SAMPLE
            assert qparams["dataset"] == MY_DATASET
            assert qparams["channel"] == MY_CHANNEL
            x = int(qparams["x"])
            assert x == MY_X
            y = int(qparams["y"])
            assert y == MY_Y
            z = int(qparams["z"])
            assert z == MY_Z
            width = int(qparams["width"])
            height = int(qparams["height"])
            ii = np.arange(y, y+height)[:, np.newaxis]
            jj = np.arange(x, x+width)[np.newaxis, :]
            img = (ii + jj + z).astype(np.uint8)
            _, data = imencode(".png", img)
        except:
            self.send_error(400, traceback.format_exc())
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.end_headers()
        self.wfile.write(data.data)
        
        
class TestButterflyTarget(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), MockButterflyRequestHandler)
        cls.host, cls.port = cls.server.server_address
        cls.thread = Thread(target = cls.server.serve_forever)
        cls.thread.setDaemon(True)
        cls.thread.start()
    
    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join()
    
    def make_target(self, width, height):
        return ButterflyTarget(
            url = "http://%s:%d/api" % (self.host, self.port),
            experiment = MY_EXPT,
            sample = MY_SAMPLE,
            dataset = MY_DATASET,
            channel = MY_CHANNEL,
            x = MY_X,
            y = MY_Y,
            z = MY_Z,
            width = width,
            height = height)
    
    def test_01_exists(self):
        b = self.make_target(20, 30)
        self.assertTrue(b.exists())
    
    def test_02_imread(self):
        b = self.make_target(21, 33)
        img = b.imread()
        self.assertEqual(img[0, 0], MY_X + MY_Y + MY_Z)
        self.assertEqual(img.shape[0], 33)
        self.assertEqual(img.shape[1], 21)

if __name__ == "__main__":
    unittest.main()