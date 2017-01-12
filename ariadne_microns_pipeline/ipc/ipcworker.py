'''ipcworker.py - a SLURM worker

An IPC worker follows the paranoid pirate pattern to process tasks handed
to it (see http://zguide.zeromq.org/page:all#reliable-request-reply).

Code derived from example at above. Thank you Daniel Lundin
'''
import argparse
import cPickle
import gc
import rh_logger
import sys
import time
import zmq
import uuid

from .protocol import *
from ..tasks.utilities import get_memstats

def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--broker",
        default="tcp://localhost:7050",
        help="The address of the broker")
    parser.add_argument(
        "--heartbeat-interval",
        default="5",
        help="The initial interval between heartbeats in seconds")
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
    
    class SWorkerArgs:
        broker = result.broker
        heartbeat_interval = float(result.heartbeat_interval)
        heartbeat_max = float(result.heartbeat_max)
        max_tries = int(result.max_tries)
        lifetime = float(result.lifetime)
    return SWorkerArgs()

def worker_socket(context, poller, broker):
    '''Get a new socket connected to the queue
    
    :param context: the ZMQ context
    :param poller: the poller for the newly created socket
    :param broker: the address of the broker to connect to
    '''
    worker = context.socket(zmq.DEALER)
    worker.setsockopt(zmq.IDENTITY, str(uuid.uuid4()))
    poller.register(worker, zmq.POLLIN)
    worker.connect(broker)
    worker.send(SP_READY)
    return worker

def main():
    rh_logger.logger.start_process("IPCWorker", "Starting")
    args = process_args()
    t0 = time.time()
    context = zmq.Context(1)
    poller = zmq.Poller()
    worker = worker_socket(context, poller, args.broker)
    heartbeat_timeout = time.time() + args.heartbeat_interval
    next_heartbeat = time.time() + args.heartbeat_interval
    last_heartbeat = time.time()
    heartbeat_misses = 0
    while time.time() < t0 + args.lifetime:
        socks = dict(poller.poll(args.heartbeat_interval * 1000))
        if socks.get(worker) == zmq.POLLIN:
            frames = worker.recv_multipart()
            if not frames:
                rh_logger.logger.report_event("No msg received, exiting")
                break
            if len(frames) == 0:
                rh_logger.logger.report_event("Unexpected: no frames received")
                break
            if frames[0] == SP_HEARTBEAT:
                heartbeat_misses = 0
                heartbeat_timeout = time.time() + args.heartbeat_interval
                rh_logger.logger.report_event("dub")
                last_heartbeat = time.time()
            elif frames[0] == SP_STOP:
                rh_logger.logger.report_event("Stopping")
                break
            elif frames[0] == SP_WORK:
                try:
                    client = frames[1]
                    rh_logger.logger.report_event("Received work")
                    work = cPickle.loads(frames[3])
                    result = work()
                    frames = [client, "", SP_RESULT, cPickle.dumps(result)]
                    rh_logger.logger.report_event("Finished running work")
                    for key, value in get_memstats().items():
                        rh_logger.logger.report_metric(
                            "IPCWorker."+key, value)
                    worker.send_multipart(frames)
                    rh_logger.logger.report_event("Sent work result")
                except:
                    rh_logger.logger.report_exception()
                    e = sys.exc_info()[1]
                    frames = [client, "", SP_EXCEPTION, cPickle.dumps(e)]
                    worker.send_multipart(frames)
                gc.collect()
        if heartbeat_timeout < time.time():
            heartbeat_misses += 1
            if heartbeat_misses >= args.max_tries:
                rh_logger.logger.report_event(
                    "No heartbeat after %.1f sec, exiting." % 
                    (time.time() - last_heartbeat))
        if next_heartbeat < time.time():
            worker.send(SP_HEARTBEAT)
            rh_logger.logger.report_event("lub")
            next_heartbeat = time.time() + args.heartbeat_interval
    rh_logger.logger.end_process("Process lifetime exceeded", 
                                 rh_logger.ExitCode.success)

if __name__=="__main__":
    main()
