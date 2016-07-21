'''Utility methods and classes for tasks'''

import luigi
import rh_logger
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