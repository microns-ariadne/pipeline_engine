"""ipcecho.py echo on the IPC broker/worker network

Baby test of the IPC mechanism:

microns-ipc-echo "Hello world"

or

microns-ipc-echo --fail "Hello world"
"""

import argparse
import cPickle
import os
import rh_logger
import time
import uuid
import zmq

from .protocol import *

class Work:
    def __init__(self, phrase):
        self.phrase = phrase
    
    def __call__(self):
        rh_logger.logger.report_event(self.phrase)
        return self.phrase

class KerasWork:
    '''This is work that should make Keras/Theano bind to a port'''
    def __call__(self):
        rh_logger.logger.report_event("Importing Keras")
        t0 = time.time()
        import keras
        rh_logger.logger.report_metric("Keras import time (sec)", 
                                       time.time() - t0)

class Fail:
    def __call__(self):
        raise Exception("I *&^%ed up")

def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--address",
                        default="tcp://localhost:7051",
                        help="Network address of the broker")
    parser.add_argument("--fail", default=False, action="store_true",
                        help="Throw an exeption instead of echoing")
    parser.add_argument("--keras", default=False, action="store_true",
                        help="Import Keras to initiate binding to a GPU")
    parser.add_argument("phrase",
                        help="Phrase to echo")
    args = parser.parse_args()
    return args

def main():
    rh_logger.logger.start_process("IPCEcho", "Starting")
    args = process_args()
    context = zmq.Context(1)
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.IDENTITY, str(uuid.uuid4()))
    socket.connect(args.address)
    poll = zmq.Poller()
    poll.register(socket, zmq.POLLIN)
    if args.fail:
        work = Fail()
    elif args.keras:
        work = KerasWork()
    else:
        work = Work(args.phrase)
    socket.send(cPickle.dumps(work))
    while True:
        socks = dict(poll.poll(timeout=10))
        if socks.get(socket) == zmq.POLLIN:
            reply = socket.recv_multipart()
            if not reply:
                break
            if len(reply) != 2:
                rh_logger.logger.report_event(
                    "Got %d args, not 2 from reply" % len(reply))
                continue
            payload = cPickle.loads(reply[1])
            if reply[0] == SP_RESULT:
                rh_logger.logger.report_event(payload)
                break
            elif reply[0] == SP_EXCEPTION:
                raise payload
            else:
                rh_logger.logger.report_event("Unknown message type: " + 
                                              reply[0])
    socket.setsockopt(zmq.LINGER, 0)
    socket.close()
    context.term()
    
if __name__=="__main__":
    main()