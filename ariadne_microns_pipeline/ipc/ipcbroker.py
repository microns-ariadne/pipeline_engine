'''ipcbroker.py - broker for workers

Code derived from ZMQ paranoid pirate example:
http://zguide.zeromq.org/py:ppqueue
'''

import argparse
from collections import OrderedDict
import cPickle
import logging
import rh_logger
import sys
import time
import threading
import zmq

from .protocol import *

def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend-address",
        default="tcp://127.0.0.1:7050",
        help="The network address for workers to connect")
    parser.add_argument(
        "--frontend-address",
        default="tcp://127.0.0.1:7051",
        help="The network address for clients to connect")
    parser.add_argument(
        "--heartbeat-interval",
        default="5",
        help="The initial interval between hearbeats in seconds")
    parser.add_argument(
        "--heartbeat-max",
        default="60",
        help="The maximum heartbeat interval when backing off in seconds")
    parser.add_argument(
        "--max-tries",
        default="5",
        help="Die after this many attempts at sending heartbeat w/o reply")
    parser.add_argument(
        "--lifetime",
        default="3600",
        help="Lifetime of process in seconds")
    result = parser.parse_args()

    class SBrokerArgs:
        frontend_address = result.frontend_address
        backend_address = result.backend_address
        heartbeat_interval = float(result.heartbeat_interval)
        heartbeat_max = float(result.heartbeat_max)
        max_tries = int(result.max_tries)
        lifetime = float(result.lifetime)
    return SBrokerArgs()

class WorkerQueue(object):
    def __init__(self):
        self.queue = OrderedDict()

    def ready(self, worker):
        self.queue.pop(worker.address, None)
        self.queue[worker.address] = worker

    def purge(self):
        """Look for & kill expired workers."""
        t = time.time()
        expired = []
        for address,worker in self.queue.iteritems():
            if t > worker.expiry:  # Worker expired
                expired.append(address)
        for address in expired:
            print "W: Idle worker expired: %s" % address
            self.queue.pop(address, None)

    def next(self):
        address, worker = self.queue.popitem(False)
        return address

class Worker(object):
    def __init__(self, address, expiry):
        self.address = address
        self.expiry = expiry
        
def main():
    rh_logger.logger.start_process("IPCBroker", "starting")
    args = process_args()

    context = zmq.Context(1)
    frontend = context.socket(zmq.ROUTER) # ROUTER
    backend = context.socket(zmq.ROUTER)  # ROUTER
    frontend.bind(args.frontend_address) # For clients
    backend.bind(args.backend_address)  # For workers
    
    poll_workers = zmq.Poller()
    poll_workers.register(backend, zmq.POLLIN)
    
    poll_both = zmq.Poller()
    poll_both.register(frontend, zmq.POLLIN)
    poll_both.register(backend, zmq.POLLIN)
    
    workers = WorkerQueue()
    
    heartbeat_at = time.time() + args.heartbeat_interval
    
    while True:
        if len(workers.queue) > 0:
            poller = poll_both
        else:
            poller = poll_workers
        socks = dict(poller.poll(args.heartbeat_interval * 1000))
    
        # Handle worker activity on backend
        if socks.get(backend) == zmq.POLLIN:
            # Use worker address for LRU routing
            frames = backend.recv_multipart()
            if not frames:
                break
    
            address = frames[0]
            workers.ready(Worker(address, time.time() + 
                                 args.heartbeat_interval * args.max_tries))
    
            # Validate control message, or return reply to client
            msg = frames[1:]
            if len(msg) == 1:
                if msg[0] not in (SP_READY, SP_HEARTBEAT):
                    rh_logger.logger.report_event(
                        "E: Invalid message from worker: %s" % msg,
                        logging.ERROR)
            else:
                frontend.send_multipart(msg)
    
            # Send heartbeats to idle workers if it's time
            if time.time() >= heartbeat_at:
                for worker in workers.queue:
                    msg = [worker, SP_HEARTBEAT]
                    backend.send_multipart(msg)
                heartbeat_at = time.time() + args.heartbeat_interval
        if socks.get(frontend) == zmq.POLLIN:
            frames = frontend.recv_multipart()
            if not frames:
                break
    
            frames.insert(0, SP_WORK)
            frames.insert(0, workers.next())
            backend.send_multipart(frames)
    
        workers.purge()

if __name__ == "__main__":
    main()
