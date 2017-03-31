import cPickle
import json
import rh_config
import rh_logger
import luigi
import zmq

from ..targets.classifier_target\
     import PixelClassifierTarget
from ..targets.volume_target import DestVolumeReader, SrcVolumeTarget
from ..parameters import VolumeParameter
from ..parameters import Volume
from ..ipc.protocol import *
from .utilities import RequiresMixin, RunMixin


class ClassifyTaskMixin:
    '''Classify pixels

    This class has a pixel classifier as its output. The file containing the
    pixel classifier should be a pickled instance of a subclass of
    PixelClassifier. The instance should have the classification parameters
    pre-loaded or bound internally.
    '''
    
    classifier_path=luigi.Parameter(
        description="Location of the pickled classifier file")
    image_loading_plan = luigi.Parameter(
        description="The filename of the dataset loading plan for the image "
                    "to be classified")
    prob_plans = luigi.DictParameter(
        description="A dictionary of dataset name to storage plan.")
    class_names = luigi.DictParameter(
        description="The class names to save. The classifier is interrogated "
        "for the probability map for each class and only the named classes "
        "are saved. Keys are the names of the classes and values are the "
        "names of the datasets to be created.")
    done_file = luigi.Parameter(
        description="The touchfile that's written after all datasets have "
        "been written to disk.")
    
    def input(self):
        yield self.get_classifier_target()
        reader = DestVolumeReader(self.image_loading_plan)
        for tgt in reader.get_source_targets():
            yield(tgt)

    def get_classifier_target(self):
        if not hasattr(self, "__classifier_target"):
            self.__classifier_target = \
                PixelClassifierTarget(self.classifier_path)
        return self.__classifier_target
    
    def get_classifier(self):
        return self.get_classifier_target().classifier
    
    def process_resources(self):
        '''Ask for the resources required by the classifier'''
        resources = self.resources.copy()
        volume = DestVolumeReader(self.image_loading_plan).volume
        resources.update(self.get_classifier_target().get_resources(volume))
        #
        # Add the process memory
        #
        if "memory" in resources:
            memory = resources["memory"]
        else:
            memory = 0
        resources["memory"] = memory + 150220000
        return resources
    
    @property
    def out_x(self):
        '''The global x coordinate of the output volume'''
        return self.volume.x + self.get_classifier().get_x_pad()
    
    @property
    def out_width(self):
        '''The width of the output volume'''
        return self.volume.width - 2 * self.get_classifier().get_x_pad()
    
    @property
    def out_y(self):
        '''The global y coordinate of the output volume'''
        return self.volume.y + self.get_classifier().get_y_pad()
    
    @property
    def out_height(self):
        '''The height of the output volume'''
        return self.volume.height - 2 * self.get_classifier().get_y_pad()
    
    @property
    def out_z(self):
        '''The global Z coordinate of the output volume'''
        return self.volume.z + self.get_classifier().get_z_pad()
    
    @property
    def out_depth(self):
        '''The # of planes in the output volume'''
        return self.volume.depth - 2 * self.get_classifier().get_z_pad()
    
    @property
    def output_volume(self):
        '''The volume of the output'''
        return Volume(self.out_x, self.out_y, self.out_z,
                      self.out_width, self.out_height, self.out_depth)
    
    def output(self):
        return luigi.LocalTarget(self.done_file)
    
class ClassifyTaskRunner(object):
    '''A class that can be used to simply reconstruct and run the classify task
    
    To use:
    
    ctr = ClassifyTaskRunner(task)
    s = cPickle.dumps(ctr)
    ctr = cPickle.loads(ctr)
    ctr()
    '''
    def __init__(self, task):
        '''Initializer
        
        :param task: a Luigi task with a to_str_params method
        '''
        self.param_str = task.to_str_params()
    
    def __call__(self):
        task = ClassifyTask.from_str_params(self.param_str)
        task()

class ClassifyRunMixin:
    
    def ariadne_run(self):
        '''Run the classifier on the input volume to produce the outputs'''
        classifier_target = self.input().next()
        classifier = classifier_target.classifier
        if classifier.run_via_ipc():
            context = zmq.Context(1)
            socket = context.socket(zmq.REQ)
            socket.setsockopt(zmq.IDENTITY, self.task_id)
            address = rh_config.config\
                .get("ipc", {})\
                .get("address", "tcp://127.0.0.1:7051")
            socket.connect(address)
            poll = zmq.Poller()
            poll.register(socket, zmq.POLLIN)
            work = cPickle.dumps(ClassifyTaskRunner(self))
            socket.send(work)
            while True:
                socks = dict(poll.poll())
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
                        rh_logger.logger.report_event(
                            "Remote execution completed")
                        break
                    elif reply[0] == SP_EXCEPTION:
                        raise payload
                    else:
                        rh_logger.logger.report_event(
                            "Unknown message type: " + reply[0])
            socket.setsockopt(zmq.LINGER, 0)
            socket.close()
            context.term()
        else:
            self()
        with self.output().open("w") as fd:
            json.dump(dict(self.prob_plans), fd)
    
    def __call__(self):
        classifier_target = self.input().next()
        classifier = classifier_target.classifier
        reader = DestVolumeReader(self.image_loading_plan)
        image = reader.imread()
        probs = classifier.classify(
            image, reader.volume.x, reader.volume.y, reader.volume.z)
        
        for class_name, dataset_name in self.class_names.items():
            prob_plan = self.prob_plans[dataset_name]
            output_target = SrcVolumeTarget(prob_plan)
            output_target.imwrite(probs[class_name])

class ClassifyTask(ClassifyTaskMixin, ClassifyRunMixin, 
                   RequiresMixin, RunMixin, luigi.Task):
    '''Classify an image, producing one or more probability maps
    
    '''
    task_namespace = "ariadne_microns_pipeline"
    
class ClassifyShimTask(RequiresMixin, luigi.Task):
    '''A shim task to return one of the ClassifyTask's probability maps
    
    The multivolume input parameters can be deduced from the classifier
    but appear to be necessary for defining the shim task as a unit.
    '''
    
    task_namespace = "ariadne_microns_pipeline"
    
    dataset_name = luigi.Parameter(
         description="The name of the desired dataset")
    storage_plan_path = luigi.Parameter(
         description="Location of the storage plan for the dataset")
    
    @staticmethod
    def make_shim(classify_task, dataset_name):
        shim = ClassifyShimTask(
            dataset_name=dataset_name,
            storage_plan_path = classify_task.prob_plans[dataset_name])
        shim.set_requirement(classify_task)
        return shim
    
    def input(self):
        yield self.requirements.copy().pop().output()
    
    def output(self):
        return SrcVolumeTarget(self.storage_plan_path)
    
    @property
    def output_volume(self):
        return self.requirements[0].output().volume
    
