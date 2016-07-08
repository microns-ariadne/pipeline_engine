'''Utility methods and classes for tasks'''

import luigi

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