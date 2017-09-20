"""protocol.py - constants for the IPC protocol

"""

'''Sent by worker: ready for new work.

Message format:
SP_READY
<environment-id>
'''
SP_READY="\x01"

'''Sent by both: still alive

Message format:
SP_HEARTBEAT
<environment-id>
'''
SP_HEARTBEAT="\x02"

'''Sent by broker: work follows

Message format:
SP_WORK
address
pickled function
'''
SP_WORK="\x03"

'''Sent by broker: worker, shut yourself down'''
SP_STOP="\x04"

'''The result of doing work'''
SP_RESULT="\x05"

'''An exception, thrown during the course of work'''
SP_EXCEPTION="\x06"

