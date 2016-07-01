'''The ariadne_microns_pipeline.tasks package holds all of the Ariadne tasks

The organization is the following: there are tasks and task mixins. The tasks
declare the inputs and output and parameterizations and the task mixins have
an `ariadne_run()` method that operates within the context of the tasks.

On top of the whole thing, there is a mixin for local execution and mixins
for cluster execution that connect "run()" to "ariadne_run()" and a
mapper/factory that lets you select the task class for a given task and
execution mechanism.
'''