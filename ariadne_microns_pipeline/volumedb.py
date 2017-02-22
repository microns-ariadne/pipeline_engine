'''volumedb.py - keep track of volumes in a pipeline

'''

import collections
import enum
import luigi
import os
import sqlalchemy
from sqlalchemy import Column,ForeignKeyConstraint, UniqueConstraint, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from .parameters import Volume

Base = declarative_base()

PATTERN="{x:09d}_{y:09d}_{z:09d}_{dataset_name:s}"

class Persistence(enum.Enum):
    '''Whether a data item should persist if no longer needed
    
    '''
    
    '''Delete after last dependent task has completed successfully'''
    Temporary=0
    '''Keep after last dependent task has completed successfully'''
    Permanent=1

class VolumeObj(Base):
    '''A volume in voxel-space'''
    __tablename__ = 'volumes'
    volume_id = Column(sqlalchemy.Integer, primary_key=True)
    x0 = Column(sqlalchemy.Integer,
                doc="The X origin of the volume in voxels")
    y0 = Column(sqlalchemy.Integer,
                doc="The Y origin of the volume in voxels")
    z0 = Column(sqlalchemy.Integer,
                doc="The Z origin of the volume in voxels")
    x1 = Column(sqlalchemy.Integer,
                doc="The limit of the vol. in the x direction (exclusive)")
    y1 = Column(sqlalchemy.Integer,
                doc="The limit of the volume in the y direction")
    z1 = Column(sqlalchemy.Integer,
                doc="The limit of the volume in the z direction")
    __table_args__ = (
        sqlalchemy.Index("volume_unique",
                         "x0", "y0", "z0", "x1", "y1", "z1",
                         unique=True),
        )

class TaskObj(Base):
    '''A Luigi task'''
    
    __tablename__ = "tasks"
    
    task_id = Column(sqlalchemy.Integer, primary_key=True)
    luigi_id = Column(sqlalchemy.Text,
                      doc="Luigi's notion of the task's ID")
    task_class = Column(sqlalchemy.Text,
                        doc="The task's Python class")
    __table_args__ = (UniqueConstraint("luigi_id"),
        )

class TaskParameterObj(Base):
    '''A parameter of a task.'''

    __tablename__ = "task_parameters"
    task_parameter_id = Column(
        "task_parameter_id", sqlalchemy.Integer, primary_key=True)
    task_id = Column(sqlalchemy.Integer,
                     ForeignKey(TaskObj.task_id),
                     doc="ID of the task associated with this parameter")
    name = Column(sqlalchemy.Text,
                  doc="The name (keyword) of this parameter")
    value = Column(sqlalchemy.Text,
                   doc="The value of the parameter as it would appear "
                   "on the command-line")
    task = relationship(TaskObj,
                        primaryjoin=task_id == TaskObj.task_id)
    __table_args__ = (UniqueConstraint("task_id", "name"),
        )

TaskObj.parameters = relationship(
    TaskParameterObj,
    primaryjoin=TaskObj.task_id==TaskParameterObj.task_id)

class DatasetTypeObj(Base):
    '''The kind of data stored in a dataset, e.g. "image"
    
    '''
    __tablename__ = "dataset_types"
    
    dataset_type_id = Column(sqlalchemy.Integer, primary_key=True)
    name = Column(sqlalchemy.Text,
                  doc="The dataset's name, e.g. \"image\".")
    persistence = Column(
        sqlalchemy.Enum(Persistence),
        doc="Whether or not to keep a dataset of this type after dependent "
        "tasks have completed")
    doc = Column(sqlalchemy.Text,
                 doc="A description of the data type, e.g. "
                 "\"The neuroproofed segmentation\".",
                 default=None)
    __table_args__ = (
            UniqueConstraint("name"),
        )

class DatasetObj(Base):
    '''A dataset is the voxel data taken on a given channel'''
    
    __tablename__ = "datasets"
    dataset_id = Column(sqlalchemy.Integer, primary_key=True)
    dataset_type_id = Column(
        sqlalchemy.Integer,
        ForeignKey(DatasetTypeObj.dataset_type_id),
        doc="The link to the dataset type, e.g. an image")
    volume_id = Column(
        sqlalchemy.Integer,
        ForeignKey(VolumeObj.volume_id),
        doc="The link to the volume encompassed by the dataset")
    task_id = Column(
        sqlalchemy.Integer,
        ForeignKey(TaskObj.task_id),
        doc="The link to the task that produced the dataset")
    __table_args__ = (
            UniqueConstraint("dataset_type_id", "volume_id"),
        )
    volume = relationship(VolumeObj, 
                          primaryjoin=volume_id==VolumeObj.volume_id)
    task = relationship(TaskObj,
                        primaryjoin=task_id==TaskObj.task_id)
    dataset_type = relationship(
        DatasetTypeObj,
        primaryjoin=dataset_type_id == DatasetTypeObj.dataset_type_id)

TaskObj.datasets = relationship(
    DatasetObj,
    primaryjoin=TaskObj.task_id == DatasetObj.task_id)

class DatasetSubvolumeObj(Base):
    '''each row is a subvolume within a dataset
       
    Note that a dataset subvolume might encompass the entire volume
    of the dataset.
    '''
    __tablename__ = "dataset_subvolumes"
    
    subvolume_id = Column(sqlalchemy.Integer, primary_key=True)
    dataset_id = Column(
        sqlalchemy.Integer,
        ForeignKey(DatasetObj.dataset_id),
        doc="The parent dataset of the volume")
    volume_id=Column(
        sqlalchemy.Integer, 
        ForeignKey(VolumeObj.volume_id),
        doc = "The volume encompassed by this subvolume")
    __table_args__ = (
            UniqueConstraint("dataset_id", "volume_id"),
        )
    dataset = relationship(DatasetObj,
                           primaryjoin=dataset_id==DatasetObj.dataset_id)
    volume = relationship(VolumeObj,
                          primaryjoin=volume_id==VolumeObj.volume_id)

 
class SubvolumeLocationObj(Base):
    '''the location on disk of a subvolume'''

    __tablename__ = "subvolume_locations"
    
    subvolume_location_id = Column(sqlalchemy.Integer, primary_key=True)
    subvolume_id = Column(sqlalchemy.Integer,
                          ForeignKey(DatasetSubvolumeObj.subvolume_id),
                          doc="The subvolume associated with this location")
    location = Column(sqlalchemy.Text,
                      doc="The location on disk")
    persistence = Column(sqlalchemy.Enum(Persistence),
                         doc="Whether to keep after last dependent task has "
                         "completed.")
    __table_args__ = (
            UniqueConstraint("subvolume_id"),
        )
    subvolume = relationship(
        DatasetSubvolumeObj,
        primaryjoin=subvolume_id==DatasetSubvolumeObj.subvolume_id)

class LoadingPlanObj(Base):
    '''A plan for loading a dataset over a volume
    
    '''
    
    __tablename__ = "loading_plans"
    
    loading_plan_id = Column(sqlalchemy.Integer, primary_key=True)
    task_id = Column(sqlalchemy.Integer, ForeignKey(TaskObj.task_id),
                     doc="The task ID of the requesting task")
    dataset_type_id = Column(sqlalchemy.Integer, 
                             ForeignKey(DatasetTypeObj.dataset_type_id),
                             doc="The dataset type link to dataset_name")
    volume_id = Column(sqlalchemy.Integer,
                       ForeignKey(VolumeObj.volume_id),
                       doc="The volume being requested")
    src_task_id = Column(sqlalchemy.Integer,
                         ForeignKey(TaskObj.task_id),
                         nullable=True,
                         doc="The source task for the dataset, if any")
    dataset_type = relation(
        DatasetTypeObj,
        primaryjoin=dataset_type_id == DatasetTypeObj.dataset_type_id)
    volume = relation(
        VolumeObj,
        primaryjoin=volume_id == VolumeObj.volume_id)
    task = relation(
        TaskObj, primaryjoin=task_id == TaskObj.task_id)
    src_task = relation(
        TaskObj, primaryjoin=src_task_id == TaskObj.task_id)
    
class SubvolumeLinkObj(Base):
    '''find all the subvolumes needed by a LoadingPlan
    
    These links let a dependent task collect its subvolumes
    '''
    __tablename__ = "subvolume_volume_links"
    subvolume_volume_link_id = Column(sqlalchemy.Integer, 
                                        primary_key=True)
    subvolume_id = Column(
        sqlalchemy.Integer,
        ForeignKey(DatasetSubvolumeObj.subvolume_id),
        doc="A link to a dataset subvolume")
    loading_plan_id=Column("loading_plan_id", sqlalchemy.Integer,
                           ForeignKey(LoadingPlanObj.loading_plan_id),
                     doc="A link to the loading plan for loading the volume")
    __table_args__ = (
            UniqueConstraint(subvolume_id, loading_plan_id),
        )
    subvolume = relationship(
        DatasetSubvolumeObj,
        primaryjoin=subvolume_id == DatasetSubvolumeObj.subvolume_id)
    loading_plan = relationship(
        LoadingPlanObj, 
        primaryjoin=loading_plan_id == LoadingPlanObj.loading_plan_id)

LoadingPlanObj.subvolume_links = relationship(
    SubvolumeLinkObj, 
    primaryjoin=LoadingPlanObj.loading_plan_id == 
    SubvolumeLinkObj.loading_plan_id)

class DatasetDependentObj(Base):
    '''tasks that are dependent on a dataset'''

    __tablename__ = "dataset_dependents"
    
    dataset_dependent_id = Column(sqlalchemy.Integer, primary_key=True)
    load_plan_id = Column(sqlalchemy.Integer, 
                          ForeignKey(LoadingPlanObj.loading_plan_id))
    dataset_id = Column(sqlalchemy.Integer, ForeignKey(DatasetObj.dataset_id))
    volume_id = Column(sqlalchemy.Integer, ForeignKey(VolumeObj.volume_id))
    __table_args__ = (UniqueConstraint("task_id", "dataset_id"),
        )
    load_plan = relationship(
        LoadPlanObj, 
        primaryjoin=load_plan_id == LoadPlanObj.load_plan_id)
    dataset = relationship(DatasetObj,
                           primaryjoin=dataset_id == DatasetObj.dataset_id)
    volume = relationship(VolumeObj,
                          primaryjoin=volume_id == VolumeObj.volume_id)

DatasetObj.dependents = relationship(
    DatasetDependentObj,
    primaryjoin=DatasetObj.dataset_id == DatasetDependentObj.dataset_id)

class SubvolumeDependentObj(Base):
    '''tasks that are dependent on a subvolume
    
         This table can be used to find the tasks that need a particular
         subvolume, e.g. after a task is done, check to see if all tasks
         dependent on a subvolume have completed.
    '''

    __tablename__ ="subvolume_dependents"
    subvolume_dependent_id = Column(sqlalchemy.Integer, primary_key=True)
    subvolume_id = Column(sqlalchemy.Integer,
                          ForeignKey(DatasetSubvolumeObj.subvolume_id),
                          doc="The subvolume required by the task")
    task_id = Column(sqlalchemy.Integer,
                     ForeignKey(TaskObj.task_id),
                     doc="The task that requires the subvolume")
    __table_args__ = (
            UniqueConstraint(subvolume_id, task_id),
        )
    subvolume = relationship(
        DatasetSubvolumeObj,
        primaryjoin=DatasetSubvolumeObj.subvolume_id == subvolume_id)
    task = relationship(
        TaskObj, primaryjoin=task_id == TaskObj.task_id)
    
class VolumeDB(object):
    '''The database of volumes in a pipeline
    
    A task might produce a volume dataset. The volume dataset has a dataset name
    and a volume (x, y, z, width, height, depth). The database keeps track
    of what task created the volume so that tasks that require portions of
    the volume can know their dependencies. A task might require only a
    *portion* of the volume. This is detected during the "build" phase, when
    the dependencies are being put together and while the index file is
    being written. If only a portion is required, internally, the volume is
    sharded into subvolumes in a way that only the required portion is loaded
    by the task that requires it.
    
    The database is written during the build portion by the luigi client,
    then read by multiple workers. The phases are:
    
    * creation - database is created on disk and connection is established
    * create_volume and use_volume - the volumes are defined and their
                 downstream use is established
    * finalization - the subvolumes are laid out and the paths to the files
                 are established. After this point, the database is read-only.
    * reading / writing - the tasks read from the database and read/write
                 the individual files.
    '''
    
    def __init__(self, db_path, mode):
        '''Initialize the database
        
        db_path - path to the database file
        mode - "r" for read-only access, "w" to create the database, "a" to
               open for writing.
        database_name - for databases that support it, the database name
                        e.g. the one passed to "use" or "create database"
        '''
        assert mode in ("r", "w", "a"), 'Mode must be one of "r", "w" or "a"'
        self.engine = sqlalchemy.create_engine(db_path)
        Session = sessionmaker()
        Session.configure(bind=self.engine)
        self.session=Session()
        self.mode = mode
        if mode == "w":
            Base.metadata.drop_all(self.engine)
            Base.metadata.create_all(self.engine)

    ######################################################
    #
    # Workflow items - the build process
    #
    ######################################################
    
    def set_temp_dir(self, temp_dir):
        '''Directory for non-persistent items'''
        self.temp_dir = temp_dir
    
    def set_target_dir(self, target_dir):
        '''Root directory for persistent items'''
        self.target_dir = target_dir
    
    def register_dataset_type(self, dataset_name, persistence, doc=None):
        '''Register a data type in the database'''
        dataset_type = DatasetTypeObj(name=dataset_name,
                                      persistence=persistence)
        if doc is not None:
            dataset_type.doc=doc
        self.session.add(dataset_type)
        self.session.commit()
    
    def get_dataset_type(self, dataset_name):
        '''Get a dataset type from the database'''
        return self.session.query(DatasetTypeObj).filter(
            DatasetTypeObj.name==dataset_name).first()
    
    def get_or_create_task(self, task, commit=True):
        '''Register a task with the database
        
        :param task: a Luigi task
        :param commit: True to commit if created, False to keep building
                       the transaction.
        :returns: a TaskObj representation of the task
        '''
        assert isinstance(task, luigi.Task)
        task_objs = self.session.execute(
            self.session.query(TaskObj).filter_by(luigi_id=task.task_id))\
            .fetchall()
        if len(task_objs) > 0:
            return task_objs[0]
        task_obj = TaskObj(luigi_id=task.task_id,
                           task_class=task.task_family)
        self.session.add(task_obj)
        for name, value in task.get_params():
            value = getattr(task, name)
            param = TaskParameterObj(name=name,
                                     value=value,
                                     task=task_obj)
            self.session.add(param)
        if commit:
            self.session.commit()
        return task_obj
    
    def get_or_create_volume_obj(self, volume, commit=True):
        '''Get or create a volume object in the database
        
        :param volume: an ariadne_microns_pipeline.parameters.Volume
        :param commit: True to commit if created, False to let it ride
                       on the current transaction
        :returns: the VolumeObj from the database
        '''
        x0 = volume.x
        x1 = x0 + volume.width
        y0 = volume.y
        y1 = y0 + volume.height
        z0 = volume.z
        z1 = z0 + volume.depth
        volume_objs = self.session.query(VolumeObj).filter_by(
                x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1).all()
        if len(volume_objs) > 0:
            return volume_objs[0]
        volume_obj = VolumeObj(x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1)
        self.session.add(volume_obj)
        if commit:
            self.session.commit()
        return volume_obj
            
    def register_dataset(self, task, dataset_name, volume):
        '''Register that a task will produce a dataset over a volume
        
        :param task: a Luigi task
        :param volume: a ariadne_microns_pipeline.parameters.Volume describing
                       the volume of the dataset
        :param dataset_name: the name of the dataset to be produced. Must be
                       one of those previously registered using
                       register_dataset_type
        '''
        assert isinstance(task, luigi.Task)
        task_obj = self.get_or_create_task(task)
        volume_obj = self.get_or_create_volume_obj(volume)
        dataset_type_obj = self.get_dataset_type(dataset_name)
        dataset_obj = DatasetObj(task=task_obj,
                                 volume=volume_obj,
                                 dataset_type=dataset_type_obj)
        self.session.add(dataset_obj)
        self.session.commit()
        return dataset_obj
    
    def find_datasets_by_type_and_volume(self, dataset_name, volume):
        '''Find all datasets with a given type that intersect a given volume
        
        :param dataset_name: find datasets whose type has this name
        :param volume: find datasets whose volumes overlap this volume
        :returns: a list of DatasetObj
        '''
        dataset_type = self.get_dataset_type(dataset_name)
        x0 = volume.x
        x1 = x0 + volume.width
        y0 = volume.y
        y1 = y0 + volume.height
        z0 = volume.z
        z1 = z0 + volume.depth
        #
        # ">" operator not supported in sqlalchemy yet.
        #
        # VolumeObj.x1.op(">", is_comparison=True)(x0) evaluates to
        # volume.x1 > x0
        #
        dataset_objs = self.session.query(DatasetObj).filter(
            sqlalchemy.and_(
                DatasetObj.dataset_type_id == DatasetTypeObj.dataset_type_id,
                DatasetTypeObj.name == dataset_name,
                DatasetObj.volume_id == VolumeObj.volume_id,
                VolumeObj.x1.op(">", is_comparison=True)(x0),
                VolumeObj.y1.op(">", is_comparison=True)(y0),
                VolumeObj.z1.op(">", is_comparison=True)(z0),
                VolumeObj.x0.op("<", is_comparison=True)(x1),
                VolumeObj.y0.op("<", is_comparison=True)(y1),
                VolumeObj.z0.op("<", is_comparison=True)(z1)))
        return dataset_objs.all()
    
    def register_dataset_dependent(self, task, dataset_name, volume, 
                                   src_task = None):
        '''Register all dependencies of a task
        
        :param task: a task that has a dataset as a requirement
        :param dataset_name: the name of the dataset, e.g. "image"
        :param volume: the required volume from the dataset
        :param src_task: the task that's the source of the dataset. By default,
        any task will do, but if specified, make sure to choose only that one
        for the case where there are overlapping datasets.
        '''
        task_obj = self.get_or_create_task(task)
        volume_obj = self.get_or_create_volume_obj(volume)
        dataset_type_obj = self.get_dataset_type(dataset_name)
        loading_plan = LoadingPlanObj(
            task = task_obj,
            volume = volume_obj,
            dataset_type = dataset_type_obj)
        if src_task is not None:
            src_task_obj = self.get_or_create_task(src_task)
            loading_plan.src_task = src_task_obj
        self.session.commit()
    
    def compute_subvolumes(self):
        '''Figure out how to break datasets into subvolumes
        
        For each dataset, find its dependents. We shard the volume at
        each intersection - for instance given a volume of 0:10, 0:10, 0:10
        and two dependents, one that requires the whole volume and one that
        requires 0:10, 0:10, 5:10, we shard the volume into two pieces:
        0:10, 0:10, 0:5 and 0:10, 0:10, 5:10.
        '''
        for loading_plan in self.session.query(LoadingPlanObj):
            volume = Volume(loading_plan.volume.x0,
                            loading_plan.volume.y0,
                            loading_plan.volume.z0,
                            loading_plan.volume.x1 - loading_plan.volume.x0,
                            loading_plan.volume.y1 - loading_plan.volume.y0,
                            loading_plan.volume.z1 - loading_plan.volume.z0)
            for dataset_obj in self.find_datasets_by_type_and_volume(
                dataset_name, volume):
                if loading_plan.src_task is not None and \
                   loading_plan.src_task.task_id != dataset_obj.task.task_id:
                    continue
                ddo = DatasetDependentObj(task=task_obj,
                                          dataset=dataset_obj,
                                          volume_id=loading_plan.volume_id)
                self.session.add(ddo)

        for dataset_obj in self.session.query(DatasetObj):
            assert isinstance(dataset_obj, DatasetObj)
            x0 = dataset_obj.volume.x0
            x1 = dataset_obj.volume.x1
            x = set([x0, x1])
            y0 = dataset_obj.volume.y0
            y1 = dataset_obj.volume.y1
            y = set([y0, y1])
            z0 = dataset_obj.volume.z0
            z1 = dataset_obj.volume.z1
            z = set([z0, z1])
            #
            # Find the shard points
            #
            for ddo in dataset_obj.dependents:
                x0a = ddo.volume.x0
                x1a = ddo.volume.x1
                y0a = ddo.volume.y0
                y1a = ddo.volume.y1
                z0a = ddo.volume.z0
                z1a = ddo.volume.z1
                if x0a > x0:
                    x.add(x0a)
                if x1a < x1:
                    x.add(x1a)
                if y0a > y0:
                    y.add(y0a)
                if y1a < y1:
                    y.add(y1a)
                if z0a > z0:
                    z.add(z0a)
                if z1a < z1:
                    z.add(z1a)
            x, y, z = sorted(x), sorted(y), sorted(z)
            #
            # Create all of the sharded volumes
            #
            for x0a, x1a in zip(x[:-1], x[1:]):
                for y0a, y1a in zip(y[:-1], y[1:]):
                    for z0a, z1a in zip(z[:-1], z[1:]):
                        volume = self.get_or_create_volume_obj(
                            Volume(x=x0a, y=y0a, z=z0a,
                                   width=x1a-x0a, height=y1a-y0a, 
                                   depth=z1a-z0a), commit=False)
                        subvolume = DatasetSubvolumeObj(
                            dataset=dataset_obj,
                            volume=volume)
                        self.session.add(subvolume)
                        volumes = set()
                        for ddo in dataset_obj.dependents:
                            x0b = ddo.volume.x0
                            x1b = ddo.volume.x1
                            y0b = ddo.volume.y0
                            y1b = ddo.volume.y1
                            z0b = ddo.volume.z0
                            z1b = ddo.volume.z1
                            key = (x0b, x1b, y0b, y1b, z0b, z1b)
                            if x0a >= x0b and x1a <= x1b and \
                               y0a >= y0b and y1a <= y1b and \
                               z0a >= z0b and z1a <= z1b:
                                if key not in volumes:
                                    link = SubvolumeVolumeLinkObj(
                                        subvolume=subvolume,
                                        volume=ddo.volume)
                                    self.session.add(link)
                                    volumes.add(key)
                                self.session.add(SubvolumeDependentObj(
                                    subvolume=subvolume,
                                    task=ddo.task))
        #
        # Assign locations
        #
        for subvolume in self.session.query(DatasetSubvolumeObj):
            persistence = subvolume.dataset.dataset_type.persistence
            dataset_name = subvolume.dataset.dataset_type.name
            if persistence == Persistence.Permanent:
                root = self.target_dir
            else:
                root = self.temp_dir
            leaf_dir = "-".join((dataset_name, str(subvolume.subvolume_id)))
            location = os.path.join(
                root, str(subvolume.volume.x0), str(subvolume.volume.y0),
                str(subvolume.volume.z0), leaf_dir)
            self.session.add(SubvolumeLocationObj(
                subvolume=subvolume,
                location=location,
                persistence=persistence))
        self.session.commit()
    
    def get_dependencies(self, task):
        '''Get a list of the tasks this task depends on
        
        :param task: the Luigi task in question
        :returns: a sequence of the Luigi task IDs of the tasks that
        must be complete before this task can run
        '''
        task2 = sqlalchemy.alias(TaskObj)
        stmt = self.session.query(TaskObj.luigi_id).filter(
            sqlalchemy.and_(
            TaskObj.task_id == DatasetObj.task_id,
            DatasetObj.dataset_id == DatasetDependentObj.dataset_id,
            DatasetDependentObj.task_id == task2.c.task_id,
            task2.c.luigi_id == task.task_id))
        return [_[0] for _ in stmt.all()]
    
    def get_subvolume_locations(self, task, dataset_name, source_task_id=None):
        '''Get the locations and volumes of datasets needed by a task
        
        This might be called to assemble the volume targets for
        
        :param task: the task requesting its dependent
        :param dataset_name: the dataset type name of the dataset to fetch
        :param source_task_id: only get datasets produced by this source task.
                            Default is to get data from wherever.
        :returns: a list of two-tuples - location and volume of the
        subvolumes needed by the task.
        '''
        result = []
        #
        # This tests if there is a DatasetDependentObj with the task ID
        # whose subvolume has the dataset_name and if there is a
        # subvolume volume link object that matches the subvolume's volume
        #
        clauses = [
            #
            # Find a task that matches the dependent's task
            #
            TaskObj.luigi_id == task.task_id,
            #
            # Find the dataset dependent record
            #
            DatasetDependentObj.task_id == TaskObj.task_id,
            #
            # Find the subvolume
            #
            DatasetDependentObj.dataset_id == DatasetSubvolumeObj.dataset_id,
            #
            # Find the link to the dataset dependent's volume
            #
            DatasetSubvolumeObj.subvolume_id ==  
            SubvolumeVolumeLinkObj.subvolume_id,
            SubvolumeVolumeLinkObj.volume_id ==  DatasetDependentObj.volume_id,
            #
            # Find the parent dataset
            #
            DatasetSubvolumeObj.dataset_id == DatasetObj.dataset_id,
            #
            # See if the dataset's data type matches dataset_name
            #
            DatasetObj.dataset_type_id == DatasetTypeObj.dataset_type_id,
            DatasetTypeObj.name == dataset_name
            ]
        if source_task is not None:
            #
            # Create an aliased task object and link it to the dataset and
            # restrict it to have the source task's task_id
            #
            source_task_alias = sqlalchemy.alias(TaskObj)
            clauses.append(DatasetObj.task_id == source_task_alias.task_id)
            clauses.append(source_task_alias.luigi_id == source_task_id)
        exists_stmt = sqlalchemy.sql.select([sqlalchemy.text("'x'")]).where(
            sqlalchemy.and_(clauses))
        for location, volume in self.session.query(
            SubvolumeLocationObj, VolumeObj).filter(
                sqlalchemy.and_(
                    SubvolumeLocationObj.subvolume_id == 
                    DatasetSubvolumeObj.subvolume_id,
                    DatasetSubvolumeObj.volume_id == VolumeObj.volume_id,
                    SubvolumeVolumeLinkObj.subvolume_id == DatasetSubvolumeObj.subvolume_id,
                    sqlalchemy.exists(exists_stmt))):
            result.append((location.location, 
                           Volume(x=volume.x0,
                                  y=volume.y0,
                                  z=volume.z0,
                                  width=volume.x1-volume.x0,
                                  height=volume.y1-volume.y0,
                                  depth=volume.z1-volume.z0),
                           location.subvolume_id))
        return result
    
    def get_src_subvolume_locations(self, task, dataset_name):
        '''Get the subvolume locations for a task that produces a subvolume
        
        :param task: the task that produced a subvolume
        :param dataset_name: the dataset produced
        :returns: a list of locations, volumes and subvolume IDs
        '''
    
all = [VolumeDB, Persistence]