'''Utility methods and classes for tasks'''


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