'''Utility methods and classes for tasks'''

import luigi
import multiprocessing
import os
import rh_logger
import sys
import time

class RequiresMixin:
    '''This mixin lets you add task requirements dynamically
    
    This mixin adds "set_requirements()" and implements "requires()" to
    let you chain dependencies together by adding dependencies to a dependent
    class dynamically.
    '''
    
    def set_requirement(self, requirement):
        if not hasattr(self, "requirements"):
            self.requirements = []
        self.requirements.append(requirement)
    
    def requires(self):
        if not hasattr(self, "requirements"):
            return []
        return self.requirements

class RunMixin:
    '''This mixin provides a standardized run() method for a task
    
    We assume that the task contains an `ariadne_run()` method and surround
    it with a timing metric and reporting of the task's ID and parameters.
    '''
    
    def task_name(self):
        '''A standardized name for the task for logging'''
        task_namespace = getattr(self, "task_namespace", None)
        if task_namespace is None:
            task_name = self.__class__.__name__
        else:
            task_name = "%s.%s" % (task_namespace, self.__class__.__name__)
        return task_name
    
    MEMSTATS_KEYS = ["VmPeak", "VmSwap"]
    def get_memstats(self):
        '''Get statistics related to this process's use of memory
        
        VmPeak: peak memory usage in Kb
        VmSwap: swap space usage in Kb
        Returns a dictionary with key = stat, value = amt used in kb
        '''
        d = {}
        try:
            for row in open("/proc/%d/status" % os.getpid()):
                fields = row.split()
                key = fields[0][:-1]
                if key in self.MEMSTATS_KEYS:
                    d[key] = int(fields[1])
        except:
            rh_logger.logger.report_event("Failed to get memory statistics")
        return d
    
    def run(self):
        if not hasattr(rh_logger.logger, "logger"):
            rh_logger.logger.start_process("Luigi", "Starting logging")
            
        task_name = self.task_name()
        rh_logger.logger.report_event("Running %s" % self.task_id)
        for name, parameter in self.get_params():
            rh_logger.logger.report_event(
                "%s: %s=%s" % (task_name, name, repr(getattr(self, name))))
        t0 = time.time()
        try:
            self.ariadne_run()
        except:
            rh_logger.logger.report_exception()
            raise
        delta = time.time() - t0
        rh_logger.logger.report_metric(
            task_name + ".runtime", delta)
        for key, value in self.get_memstats().items():
            rh_logger.logger.report_metric(
                task_name + "." + key, value)


class CILKCPUMixin:
    '''This mixin provides a standardized mechanism for setting CPU utilization
    
    CILK uses the environment variable, CILK_NWORKERS, to set the number of
    CPUs available to a subprocess. This mixin makes that number configurable
    for a task and asks the scheduler for the given number of CPUs.
    '''
    cpu_count = luigi.IntParameter(
         default=min(
             luigi.configuration.get_config().getint(
                 "resources", "cpu_count", default=sys.maxint),
             multiprocessing.cpu_count()),
         description="The number of CPUs/CILK workers devoted to this task")
    
    def process_resources(self):
        resources = self.resources.copy()
        resources["cpu_count"] = self.cpu_count
        return resources
    
    def configure_env(self, env):
        '''Configure the CILK subprocess environment
        
        Set CILK_NWORKERS for the number of CPUs to use
        
        :param env: a copy of os.environ to be passed to the subprocess
        '''
        env["CILK_NWORKERS"] = str(self.cpu_count)


class SingleThreadedMixin:
    '''This mixin declares that its task runs single-threaded
    
    The upshot of this is that the task consumes one CPU resource.
    '''
    def process_resources(self):
        resources = self.resources.copy()
        resources["cpu_count"] = 1
        return resources


def to_hashable(x):
    '''Make dictionaries and lists hashable
    
    Convert dictionaries to Luigi FrozenOrderedDict and lists to tuples.
    '''
    if isinstance(x, list):
        return tuple([to_hashable(_) for _ in x])
    
    #
    # Make a dictionary hashable by using Luigi's FrozenOrderedDict
    # and a consistent insertion order
    #
    if isinstance(x, dict):
        return luigi.parameter.FrozenOrderedDict(
            [(k, to_hashable(x[k])) for k in sorted(x.keys())])
    #
    # Make sure all other cases are hashable before returning them
    #
    assert isinstance(hash(x), int)
    return x