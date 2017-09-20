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

if __name__ == "__main__":
    from protocol import *
else:
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
            rh_logger.logger.report_event("Idle worker expired: %s" % address)
            self.queue.pop(address, None)

    def next(self):
        address, worker = self.queue.popitem(False)
        return address

class Worker(object):
    def __init__(self, address, expiry):
        self.address = address
        self.expiry = expiry

class Broker(object):
    
    def __init__(self):
        rh_logger.logger.start_process("IPCBroker", "starting")
        self.args = process_args()

        self.context = zmq.Context(1)
        self.frontend = self.context.socket(zmq.ROUTER) # ROUTER
        self.backend = self.context.socket(zmq.ROUTER)  # ROUTER
        self.frontend.bind(self.args.frontend_address) # For clients
        self.backend.bind(self.args.backend_address)  # For workers
    
        self.poll_workers = zmq.Poller()
        self.poll_workers.register(self.backend, zmq.POLLIN)
    
        self.poll_both = zmq.Poller()
        self.poll_both.register(self.frontend, zmq.POLLIN)
        self.poll_both.register(self.backend, zmq.POLLIN)
    
        self.workers = {}
        self.work = {}
        self.reset_heartbeat()
    
    def add_worker(self, address, environment_id):
        '''Add a new or existing worker to the system
        
        :param address: the worker's ZMQ address
        :param environment_id: an arbitrary string identifying the
        worker's Python environment, e.g. the virtual environment name
        '''
        if environment_id not in self.workers:
            self.workers[environment_id] = WorkerQueue()
        expiry = \
            time.time() + self.args.heartbeat_interval * self.args.max_tries
        self.workers[environment_id].ready(
            Worker(address, expiry))
        if environment_id in self.work:
            client_address, work = self.work[environment_id].pop()
            if len(self.work[environment_id]) == 0:
                del self.work[environment_id]
            self.send_work_to_worker(environment_id, work, client_address)
    
    def send_work_to_worker(self, environment_id, work, client_address):
        '''Send work to the worker via the backend
        
        :param environment_id: the ID of the target environment for the worker
        :param work: the work message (a pickled function to run)
        :param client_address: the ZMQ address of the client who sent the work
        '''
        if environment_id not in self.workers:
            rh_logger.logger.report_event(
                "Enqueueing work for environment %s - no worker available" %
                environment_id)
            if environment_id not in self.work:
                self.work[environment_id] = []
            self.work[environment_id].insert(0, (client_address, work))
            return
        worker_address = self.workers[environment_id].next()
        if len(self.workers[environment_id].queue) == 0:
            del self.workers[environment_id]
        rh_logger.logger.report_event(
            "Sending work to %s" % worker_address)
        frames = [worker_address, SP_WORK, client_address, "", work]
        self.backend.send_multipart(frames)
                
    def reset_heartbeat(self):
        '''Get the time of the next heartbeat'''
        self.heartbeat_at = time.time() + self.args.heartbeat_interval
    
    def run_poll_loop(self):
        '''Run the polling - dishing work loop
        
        Call like this:
        while broker.run_poll_loop():
             print "tick"
             
        :returns: True to keep going, False if we were sent a "die" msg
        '''
        if len(self.workers) > 0:
            poller = self.poll_both
        else:
            poller = self.poll_workers
        socks = dict(poller.poll(self.args.heartbeat_interval * 1000))
    
        # Handle worker activity on backend
        if socks.get(self.backend) == zmq.POLLIN:
            # Use worker address for LRU routing
            frames = self.backend.recv_multipart()
            if not frames:
                return False
    
            self.handle_backend(frames)
    
            # Send heartbeats to idle workers if it's time
            if time.time() >= self.heartbeat_at:
                for environment_id, worker_queue in self.workers.items():
                    for worker in worker_queue.queue:
                        msg = [worker, SP_HEARTBEAT]
                        self.backend.send_multipart(msg)
                        rh_logger.logger.report_event("Sent heartbeat")
                self.reset_heartbeat()
        if socks.get(self.frontend) == zmq.POLLIN:
            rh_logger.logger.report_event("Received work")
            frames = self.frontend.recv_multipart()
            if not frames:
                return False
    
            self.handle_frontend(frames)
    
        dead_environments = []
        for environment_id, worker_queue in self.workers.items():
            worker_queue.purge()
            if len(worker_queue.queue) == 0:
                dead_environments.append(environment_id)
        for environment_id in dead_environments:
            del self.workers[environment_id]
        return True

    def handle_frontend(self, frames):
        '''Handle a message from the client
        
        :param frames: the multipart message from the client
        '''
        if len(frames) != 5 or len(frames[1]) != 0:
            rh_logger.logger.report_event(
                "Malformed message from client: %s" % repr(frames),
                logging.ERROR)
            return
        client_address, _, code, environment_id, work = frames
        if code == SP_WORK:
            self.send_work_to_worker(environment_id, work, client_address)
        else:
            rh_logger.logger.report_event("Unknown message code: %s" % frames)

    def handle_backend(self, frames):
        '''Handle a message from a worker
        
        :param frames: the multipart message from the worker
        '''
        address = frames[0]
    
        # Validate control message, or return reply to client
        msg = frames[1:]
        if len(msg) == 2:
            # This is a registration message from the client. The format is
            # SP_READY or SP_HEARTBEAT
            # <environment-id>
            # 
            code = msg[0]
            environment_id = msg[1]
            if code not in (SP_READY, SP_HEARTBEAT):
                rh_logger.logger.report_event(
                    "E: Invalid message from worker: %s" % msg,
                    logging.ERROR)
            if msg[0] == SP_READY:
                rh_logger.logger.report_event("Received SP_READY")
            elif msg[0] == SP_HEARTBEAT:
                rh_logger.logger.report_event("Received SP_HEARTBEAT")
            self.add_worker(address, environment_id)
        elif len(msg[1]) == 0:
            # This is a message to be forwarded to the client. The format is
            # client-address
            # -- blank --
            # msg to be forwarded
            #
            rh_logger.logger.report_event(
                "Forwarding work from %s to %s" % (frames[0], frames[1]))
            self.frontend.send_multipart(msg)
        else:
            rh_logger.logger.report_event("Unknown msg: %s" % repr(frames),
                                          log_level=logging.ERROR)

def main():
    broker = Broker()
    while broker.run_poll_loop():
        pass
    
if __name__ == "__main__":
    main()
