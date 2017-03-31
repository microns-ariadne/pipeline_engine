'''volumedb.py - keep track of volumes in a pipeline

'''

import collections
import enum
import luigi
import numpy as np
import os
import rh_logger
import sqlalchemy
import tifffile
import time
from sqlalchemy import Column,ForeignKeyConstraint, UniqueConstraint, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from .parameters import Volume

Base = declarative_base()

PATTERN="{x:09d}_{y:09d}_{z:09d}_{dataset_name:s}"

UINT8 = "uint8"
UINT16 = "uint16"
UINT32 = "uint32"
UINT64 = "uint64"

def get_storage_plan_path(root, dataset_id, volume, dataset_name):
    '''Get the canonical path to a storage plan
    
    This lets you anticipate the location of the storage plan before it's
    actually created.
    
    :param root: the root directory of the storage hierarchy
    :param dataset_id: the dataset_id of the dataset being saved
    :param volume: a Volume object giving the volume being written
    :param dataset_name: the name of the dataset, e.g. "image"
    '''
    leaf_dir = "%s_%09d-%09d_%09d-%09d_%09d-%09d_%d.storage.plan" % (
        dataset_name, volume.x, volume.x1,
        volume.y, volume.y1,
        volume.z, volume.z1,
        dataset_id)
    location = os.path.join(
        root, str(volume.x), str(volume.y),
        str(volume.z), leaf_dir)
    return location

def get_loading_plan_path(root, loading_plan_id, volume, dataset_name):
    '''Get the canonical path to a loading plan
    
    :param root: the root directory of the storage hierarchy
    :param loading_plan_id: the loading_plan_id of the dataset being loaded
    :param volume: a Volume object giving the volume being written
    :param dataset_name: the name of the dataset, e.g. "image"
    '''
    leaf_dir = "%s_%09d-%09d_%09d-%09d_%09d-%09d_%d.loading.plan" % (
        dataset_name, volume.x, volume.x1,
        volume.y, volume.y1,
        volume.z, volume.z1,
        loading_plan_id)
    location = os.path.join(
        root, str(volume.x), str(volume.y),
        str(volume.z), leaf_dir)
    return location


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
    @classmethod
    def create(cls, engine):
        '''Create the database table
        
        We use the special rtree extension to make searches faster
        
        :param engine: a database engine, e.g. from sqlalchemy.create_engine
        '''
        engine.execute(
            ("create virtual table %s using "
             "rtree(volume_id, x0, x1, y0, y1, z0, z1)") % cls.__tablename__)

    def volume(self):
        '''Return the parameters.Volume style volume for this obj'''
        return Volume(int(np.round(self.x0)), 
                      int(np.round(self.y0)),
                      int(np.round(self.z0)),
                      int(np.round(self.x1 - self.x0)),
                      int(np.round(self.y1 -self.y0)),
                      int(np.round(self.z1-self.z0)))

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
    datatype = Column(
        sqlalchemy.Text,
        doc="A Numpy datatype, e.g. \"uint8\"")
    doc = Column(sqlalchemy.Text,
                 doc="A description of the data type, e.g. "
                 "\"The neuroproofed segmentation\".",
                 default=None)
    __table_args__ = (
            UniqueConstraint("name"),
        )

class DatasetIDObj(Base):
    '''A table that only contains dataset_ids for the datasets table
    
    There's a chicken and egg problem - you can't make a task without its
    parameters, especially critically defining ones like which dataset it
    produces. Thus the workflow is:
    
    * Get a dataset_id
    * Create your task
    * Create a dataset with that ID
    '''
    __tablename__ = "dataset_ids"
    dataset_id = Column(sqlalchemy.Integer, primary_key=True)
    
class DatasetObj(Base):
    '''A dataset is the voxel data taken on a given channel'''
    
    __tablename__ = "datasets"
    dummy_id = Column(sqlalchemy.Integer, primary_key=True)
    dataset_id = Column(sqlalchemy.Integer, 
                        ForeignKey(DatasetIDObj.dataset_id),
                        unique=True,
                        index=True)
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

class LoadingPlanIDObj(Base):
    '''A table holding just loading_plan_id numbers
    
    You can't create a task with input volumes w/o having a loading plan ID
    and you can't create a LoadingPlanObj without a task, so get the ID
    first.
    '''
    __tablename__ = "loading_plan_ids"
    loading_plan_id = Column(sqlalchemy.Integer, primary_key=True)

class LoadingPlanObj(Base):
    '''A plan for loading a dataset over a volume
    
    '''
    
    __tablename__ = "loading_plans"
    
    dummy_id = Column(sqlalchemy.Integer, primary_key=True)
    loading_plan_id = Column(sqlalchemy.Integer, 
                             ForeignKey(LoadingPlanIDObj.loading_plan_id),
                             unique=True,
                             index=True)
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
    dataset_type = relationship(
        DatasetTypeObj,
        primaryjoin=dataset_type_id == DatasetTypeObj.dataset_type_id)
    volume = relationship(
        VolumeObj,
        primaryjoin=volume_id == VolumeObj.volume_id)
    task = relationship(
        TaskObj, primaryjoin=task_id == TaskObj.task_id)
    src_task = relationship(
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
    '''Loading plans that are dependent on a dataset'''

    __tablename__ = "dataset_dependents"
    
    dataset_dependent_id = Column(sqlalchemy.Integer, primary_key=True)
    loading_plan_id = Column(sqlalchemy.Integer, 
                          ForeignKey(LoadingPlanObj.loading_plan_id))
    dataset_id = Column(sqlalchemy.Integer, ForeignKey(DatasetObj.dataset_id))
    __table_args__ = (UniqueConstraint(loading_plan_id, dataset_id),
        )
    loading_plan = relationship(
        LoadingPlanObj, 
        primaryjoin=loading_plan_id == LoadingPlanObj.loading_plan_id)
    dataset = relationship(DatasetObj,
                           primaryjoin=dataset_id == DatasetObj.dataset_id)

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
            try:
                VolumeObj.create(self.engine)
            except:
                rh_logger.logger.report_exception()
                rh_logger.logger.report_event(
                    "Support for spatial querying disabled")
            Base.metadata.create_all(self.engine)
        self.loading_plan_ids = {}
        self.all_volumes = {}
        
    def __enter__(self):
        return self
    
    def __exit__(self, err_type, value, traceback):
        if value is not None:
            self.session.rollback()
        else:
            self.session.commit()
        self.session.close()
        return value is None
    
    def cleanup(self):
        '''Get rid of any cache / state
        
        This should be called after any sustained operation to clear out
        cached objects.
        '''
        self.session.expunge_all()
        self.all_volumes = {}
    
    def copy_db(self, dest_url):
        '''Copy the database schema and content to another database
        
        :param dest_url: the SQLAlchemy URL of the other database
        '''
        dest_engine = sqlalchemy.create_engine(dest_url)
        Session = sessionmaker()
        Session.configure(bind=self.engine)
        session=Session()
        Base.metadata.drop_all(dest_engine)
        Base.metadata.create_all(dest_engine)
        
        for table in DatasetTypeObj, DatasetIDObj, VolumeObj, DatasetObj, \
            DatasetSubvolumeObj, SubvolumeLocationObj, SubvolumeLinkObj, \
            LoadingPlanIDObj, LoadingPlanObj, SubvolumeDependentObj:
            objs = self.session.query(table).all()
            map(self.session.expunge, objs)
            map(session.merge, objs)
        session.commit()
        
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
    
    def register_dataset_type(self, dataset_name, persistence, datatype,
                              doc=None):
        '''Register a data type in the database
        
        :param dataset_name: the name of the data type, e.g. "image"
        :param persistence: one of the Persistence enums indicating whether
        the data should be ephemeral (possibly deleted after last use) or
        persistent
        :param datatype: the Numpy datatype, suitable for 
        getattr(numpy, datatype) to retrieve the dataset's datatype
        :param doc: documentation describing the "meaning" of the data type
        '''
        dataset_type = DatasetTypeObj(name=dataset_name,
                                      persistence=persistence,
                                      datatype=datatype)
        if doc is not None:
            dataset_type.doc=doc
        self.session.add(dataset_type)
        self.session.commit()
    
    def get_datatype_root(self, dataset_name):
        '''Given a dataset's name, return the root of its file system
        
        Based on how it was registered, a dataset type will be stored in the
        temporary or permanent filesystem hierarchy, so retreive one or the
        other based on the dataset_name
        
        :param dataset_name: the name of the dataset, e.g. "image"
        '''
        persistence = self.session.query(DatasetTypeObj.persistence).filter(
            DatasetTypeObj.name == dataset_name).first()[0]
        return self.target_dir if persistence == Persistence.Permanent \
               else self.temp_dir
    
    def get_dataset_type(self, dataset_name):
        '''Get a dataset type from the database'''
        return self.session.query(DatasetTypeObj).filter(
            DatasetTypeObj.name==dataset_name).first()
    
    def get_dataset_name_by_dataset_id(self, dataset_id):
        '''Get the the dataset's type name from the dataset
        
        :param dataset_id: the dataset_id of the dataset to fetch
        :returns: the name for the dataset's dataset_type
        '''
        return self.session.query(DatasetTypeObj.name).filter(
            DatasetTypeObj.dataset_type_id == DatasetObj.dataset_type_id and
            DatasetObj.dataset_id == dataset_id).first()[0]
    
    def get_dataset_dtype_by_dataset_id(self, dataset_id):
        '''Get the Numpy dtype of a dataset via the dataset_id of the record
        
        :param dataset_id: the dataset_id of the dataset being written
        :returns: the name of the Numpy dtype, suitable for getattr(np, dtype)
        to fetch it, e.g. "uint8"
        '''
        datatype = self.session.query(DatasetTypeObj.datatype).filter(
            DatasetTypeObj.dataset_type_id == DatasetObj.dataset_type_id and
            DatasetObj.dataset_id == dataset_id).first()[0]
        return datatype
    
    def get_or_create_task(self, task, commit=True):
        '''Register a task with the database
        
        :param task: a Luigi task
        :param commit: True to commit if created, False to keep building
                       the transaction.
        :returns: a TaskObj representation of the task
        '''
        assert isinstance(task, luigi.Task)
        task_obj = self.session.query(TaskObj).filter_by(luigi_id=task.task_id)\
            .first()
        if task_obj is not None:
            return task_obj
        task_obj = TaskObj(luigi_id=task.task_id,
                           task_class=task.task_family)
        self.session.add(task_obj)
        for name, value in task.get_params():
            value = getattr(task, name)
            param = TaskParameterObj(name=name,
                                     value=repr(value),
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
        key = (x0, x1, y0, y1, z0, z1)
        if key in self.all_volumes:
            return self.all_volumes[key]
        
        volume_objs = self.session.query(VolumeObj).filter_by(
                x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1).all()
        if len(volume_objs) > 0:
            self.all_volumes[key] = volume_objs[0]
            return volume_objs[0]
        volume_obj = VolumeObj(x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1)
        self.session.add(volume_obj)
        if commit:
            self.session.commit()
        self.all_volumes[key] = volume_obj
        return volume_obj
    
    def get_dataset_id(self):
        '''Get a dataset ID in preparation for registering a dataset'''
        dataset_id_obj = DatasetIDObj()
        self.session.add(dataset_id_obj)
        self.session.commit()
        return dataset_id_obj.dataset_id
            
    def register_dataset(self, dataset_id, task, dataset_name, volume):
        '''Register that a task will produce a dataset over a volume
        
        :param dataset_id: the dataset ID for the dataset as fetched by
        get_dataset_id()
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
        dataset_obj = DatasetObj(
            dataset_id=dataset_id,
            task=task_obj,
            volume=volume_obj,
            dataset_type=dataset_type_obj)
        self.session.add(dataset_obj)
        self.session.commit()
        return dataset_obj.dataset_id
    
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
    
    def find_loading_plan_id_by_type_and_volume(self, dataset_name, volume):
        '''Find the ID of the loading plan that loads the given dataset volume
        
        This can be used either to retrieve a known loading plan or to search
        for a possibly precreated one, e.g. the membrane loading plan created
        by BorderMaskTask for FindSeedsTask and subsequent.
        
        There should only be one loading plan per unique dataset & volume.
        
        :param dataset_name: the name of the dataset to be loaded
        :param volume: the volume to be loaded from the dataset
        :returns: None if there is no such loading plan, otherwise the loading
        plan ID of the already-existing loading plan
        '''
        key = self._get_loading_plan_key(dataset_name, volume)
        if key in self.loading_plan_ids:
            return self.loading_plan_ids[key]
        result = self.session.query(LoadingPlanObj.loading_plan_id).filter(
            sqlalchemy.and_(
                LoadingPlanObj.volume_id == VolumeObj.volume_id,
                VolumeObj.x0 == volume.x,
                VolumeObj.y0 == volume.y,
                VolumeObj.z0 == volume.z,
                VolumeObj.x1 == volume.x1,
                VolumeObj.y1 == volume.y1,
                VolumeObj.z1 == volume.z1,
                LoadingPlanObj.dataset_type_id == 
                DatasetTypeObj.dataset_type_id,
                DatasetTypeObj.name == dataset_name
                )).first()
        if result is None:
            return result
        self.loading_plan_ids[key] = result[0]
        return result[0]
    
    @staticmethod
    def _get_loading_plan_key(dataset_name, volume):
        key = (dataset_name, volume.x, volume.y, volume.z,
               volume.width, volume.height, volume.depth)
        return key
    
    def get_loading_plan_id(self):
        '''Get a loading_plan_id in preparation for requesting a dataset
        '''
        loading_plan_id_obj = LoadingPlanIDObj()
        self.session.add(loading_plan_id_obj)
        self.session.commit()
        return loading_plan_id_obj.loading_plan_id
    
    def register_dataset_dependent(
        self, loading_plan_id, task, dataset_name, volume, src_task = None):
        '''Register all dependencies of a task
        
        :loading_plan_id: the loading plan ID used to refer to the load plan
        for retrieving the volume, e.g. as retrieved from get_loading_plan_id()
        :param task: a task that has a dataset as a requirement
        :param dataset_name: the name of the dataset, e.g. "image"
        :param volume: the required volume from the dataset
        :param src_task: the task that's the source of the dataset. By default,
        any task will do, but if specified, make sure to choose only that one
        for the case where there are overlapping datasets.
        :returns: the loading plan ID which can be used to fetch the subvolumes
        '''
        task_obj = self.get_or_create_task(task)
        volume_obj = self.get_or_create_volume_obj(volume)
        dataset_type_obj = self.get_dataset_type(dataset_name)
        loading_plan = LoadingPlanObj(
            loading_plan_id = loading_plan_id,
            task = task_obj,
            volume = volume_obj,
            dataset_type = dataset_type_obj)
        if src_task is not None:
            src_task_obj = self.get_or_create_task(src_task)
            loading_plan.src_task_id = src_task_obj.task_id
        self.session.add(loading_plan)
        self.session.commit()
        key = self._get_loading_plan_key(dataset_name, volume)
        self.loading_plan_ids[key] = loading_plan_id
        return loading_plan.loading_plan_id
    
    def compute_subvolumes(self):
        '''Figure out how to break datasets into subvolumes
        
        For each dataset, find its dependents. We shard the volume at
        each intersection - for instance given a volume of 0:10, 0:10, 0:10
        and two dependents, one that requires the whole volume and one that
        requires 0:10, 0:10, 5:10, we shard the volume into two pieces:
        0:10, 0:10, 0:5 and 0:10, 0:10, 5:10.
        '''
        for loading_plan in self.session.query(LoadingPlanObj):
            volume = loading_plan.volume.volume()
            for dataset_obj in self.find_datasets_by_type_and_volume(
                loading_plan.dataset_type.name, volume):
                if loading_plan.src_task is not None and \
                   loading_plan.src_task.task_id != dataset_obj.task.task_id:
                    continue
                ddo = DatasetDependentObj(dataset=dataset_obj,
                                          loading_plan=loading_plan)
                self.session.add(ddo)
        self.session.commit()
        self.cleanup()
        
        for dataset_obj in self.session.query(DatasetObj):
            assert isinstance(dataset_obj, DatasetObj)
            volume = dataset_obj.volume.volume()
            x0 = volume.x
            x1 = volume.x1
            x = set([x0, x1])
            y0 = volume.y
            y1 = volume.y1
            y = set([y0, y1])
            z0 = volume.z
            z1 = volume.z1
            z = set([z0, z1])
            #
            # Find the shard points
            #
            for ddo in dataset_obj.dependents:
                volume = ddo.loading_plan.volume.volume()
                x0a = volume.x
                x1a = volume.x1
                y0a = volume.y
                y1a = volume.y1
                z0a = volume.z
                z1a = volume.z1
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
                            volumeb = ddo.loading_plan.volume.volume()
                            x0b = volumeb.x
                            x1b = volumeb.x1
                            y0b = volumeb.y
                            y1b = volumeb.y1
                            z0b = volumeb.z
                            z1b = volumeb.z1
                            key = (x0b, x1b, y0b, y1b, z0b, z1b)
                            if x0a >= x0b and x1a <= x1b and \
                               y0a >= y0b and y1a <= y1b and \
                               z0a >= z0b and z1a <= z1b:
                                if key not in volumes:
                                    link = SubvolumeLinkObj(
                                        subvolume=subvolume,
                                        loading_plan_id=ddo.loading_plan_id)
                                    self.session.add(link)
                                    volumes.add(key)
                                self.session.add(SubvolumeDependentObj(
                                    subvolume=subvolume,
                                    task_id=ddo.loading_plan.task_id))
        self.cleanup()
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
            svolume = subvolume.volume.volume()
            leaf_dir = "%s_%09d-%09d_%09d-%09d_%09d-%09d_%d.tif" % (
                dataset_name, svolume.x, svolume.x1,
                svolume.y, svolume.y1,
                svolume.z, svolume.z1,
                subvolume.subvolume_id)
            location = os.path.join(
                root, str(svolume.x), str(svolume.y),
                str(svolume.z), leaf_dir)
            self.session.add(SubvolumeLocationObj(
                subvolume=subvolume,
                location=location,
                persistence=persistence))
        self.cleanup()
        self.session.commit()
    
    def get_loading_plan_path(self, loading_plan_id):
        '''Get the canonical loading plan path from the loading plan ID
        
        :param loading_plan_id: the loading plan ID of the loading plan
        whose path we want.
        '''
        loading_plan = self.session.query(LoadingPlanObj).filter(
            LoadingPlanObj.loading_plan_id == loading_plan_id).first()
        assert isinstance(loading_plan, LoadingPlanObj)
        loading_plan_path = get_loading_plan_path(
            self.get_datatype_root(loading_plan.dataset_type.name),
            loading_plan_id, loading_plan.volume.volume(),
            loading_plan.dataset_type.name)
        return loading_plan_path
    
    def get_storage_plan_path(self, dataset_id):
        '''Get the canonical storage plan path from the dataset id
        
        :param dataset_id: the dataset ID for the dataset whose storage plan
        we are referencing.
        '''
        dataset = self.session.query(DatasetObj).filter(
            DatasetObj.dataset_id == dataset_id).first()
        assert isinstance(dataset, DatasetObj)
        return get_storage_plan_path(
            self.get_datatype_root(dataset.dataset_type.name),
            dataset_id,
            dataset.volume.volume(),
            dataset.dataset_type.name)
    
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
            DatasetDependentObj.loading_plan_id == 
            LoadingPlanObj.loading_plan_id,
            LoadingPlanObj.task_id == task2.c.task_id,
            task2.c.luigi_id == task.task_id))
        return [_[0] for _ in stmt.all()]
    
    def get_subvolume_locations(self, task, dataset_name, src_task_id=None):
        '''Get the locations and volumes of datasets needed by a task
        
        :param task: the task requesting its dependent
        :param dataset_name: the dataset type name of the dataset to fetch
        :param src_task_id: only get datasets produced by this source task.
                            Default is to get data from wherever.
        :returns: a list of two-tuples - location and volume of the
        subvolumes needed by the task.
        '''
        #
        # From subvolume location to subvolume to subvolume link to
        # loading plan
        # From subvolume to volume
        # From loading plan to task, dataset type and source task
        clauses = [
            VolumeObj.volume_id == DatasetSubvolumeObj.volume_id,
            
            DatasetSubvolumeObj.subvolume_id == 
            SubvolumeLinkObj.subvolume_id,
            
            SubvolumeLocationObj.subvolume_id == 
            SubvolumeLinkObj.subvolume_id,
            
            SubvolumeLinkObj.loading_plan_id == LoadingPlanObj.loading_plan_id,
            
            LoadingPlanObj.task_id == TaskObj.task_id,
            
            TaskObj.luigi_id == task.task_id,
            
            LoadingPlanObj.dataset_type_id == DatasetTypeObj.dataset_type_id, 
            
            DatasetTypeObj.name == dataset_name
        ]
        if src_task_id is not None:
            src_task_alias = sqlalchemy.alias(TaskObj)
            clauses.append(
                src_task_alias.c.task_id == LoadingPlanObj.src_task_id)
            clauses.append(src_task_alias.c.luigi_id == src_task_id)
        return self._make_location_volume_result(clauses)
    
    def get_subvolume_locations_by_loading_plan_id(self, loading_plan_id):
        '''Fetch locations for loading via the loading plan
        
        :param loading_plan_id: the loading_plan_id, e.g. as returned by
        register_dataset_dependent
        '''
        clauses = [
            VolumeObj.volume_id == DatasetSubvolumeObj.volume_id,
            
            DatasetSubvolumeObj.subvolume_id == 
            SubvolumeLinkObj.subvolume_id,
            
            SubvolumeLocationObj.subvolume_id == 
            SubvolumeLinkObj.subvolume_id,
            
            SubvolumeLinkObj.loading_plan_id == loading_plan_id]
        result = self._make_location_volume_result(clauses)
        return result

    def _make_location_volume_result(self, clauses):
        '''Issue a query over subvolume_locations and volume
        
        Given clauses to a "where" clause, extract joined rows from the
        subvolume_locations and volumes tables and package them into
        two-tuples of locations and volumes
        
        :param clauses: a list of boolean clauses that make up the join
        selecting the rows
        
        '''
        result = []
        for subvolume_location, volume in \
            self.session.query(SubvolumeLocationObj, VolumeObj).filter(
                sqlalchemy.and_(*clauses)):
            result.append((subvolume_location.location, volume.volume()))
        return result
    
    def get_subvolume_locations_by_dataset_id(self, dataset_id):
        '''Fetch locations for writing via the dataset
        
        :param dataset_id: the dataset ID of the volume to be written, e.g.
        from the call to register_dataset()
        :returns: a list of locations and volumes
        '''
        #
        # Go from dataset_id to subvolume to subvolume location
        #
        clauses = [
            DatasetSubvolumeObj.dataset_id == dataset_id,
            
            DatasetSubvolumeObj.volume_id == VolumeObj.volume_id,
           
            SubvolumeLocationObj.subvolume_id == 
            DatasetSubvolumeObj.subvolume_id
        ]
        return self._make_location_volume_result(clauses)
    
    def get_loading_plan_ids(self):
        '''Return a sequence of all of the loading plan IDs
        
        '''
        return map(lambda _:_[0],
                   self.session.query(LoadingPlanObj.loading_plan_id))
    
    def get_dataset_ids(self):
        '''Return a sequence of all of the dataset IDs'''
        
        return map(lambda _:_[0],
                   self.session.query(DatasetObj.dataset_id))
    
    def imread(self, loading_plan_id):
        '''Read a volume, using a loading plan
        
        :param loading_plan_id: the ID of a loading plan, e.g. from
        register_dataset_dependent
        '''
        t0 = time.time()
        loading_plan = self.session.query(LoadingPlanObj).filter(
            LoadingPlanObj.loading_plan_id == loading_plan_id).first()
        assert isinstance(loading_plan, LoadingPlanObj)
        volume = loading_plan.volume.volume()
        x0 = volume.x
        x1 = volume.x1
        y0 = volume.y
        y1 = volume.y1
        z0 = volume.z
        z1 = volume.z1
        rh_logger.logger.report_event(
            "Loading %s: %d:%d, %d:%d, %d:%d" % (
                loading_plan.dataset_type.name,
                x0, x1, y0, y1, z0, z1))

        result = None
        datatype = getattr(np, loading_plan.dataset_type.datatype)
        
        clauses = [
            VolumeObj.volume_id == DatasetSubvolumeObj.volume_id,
            
            DatasetSubvolumeObj.subvolume_id == 
            SubvolumeLinkObj.subvolume_id,
            
            SubvolumeLocationObj.subvolume_id == 
            SubvolumeLinkObj.subvolume_id,
            
            SubvolumeLinkObj.loading_plan_id == loading_plan_id]
        
        for subvolume_location, svolume in self.session.query(
            SubvolumeLocationObj, VolumeObj).filter(
                sqlalchemy.and_(*clauses)):
            tif_path = subvolume_location.location
            svolume = svolume.volume()
            if svolume.x >= x1 or\
               svolume.y >= y1 or\
               svolume.z >= z1 or \
               svolume.x1 <= x0 or \
               svolume.y1 <= y0 or \
               svolume.z1 <= z0:
                rh_logger.logger.report_event(
                    "Ignoring block %d:%d, %d:%d, %d:%d from load plan" %
                (svolume.x, svolume.x1, svolume.y, svolume.y1,
                 svolume.z, svolume.z1))
                continue
                
            with tifffile.TiffFile(tif_path) as fd:
                block = fd.asarray()
                if svolume.x == x0 and \
                   svolume.x1 == x1 and \
                   svolume.y == y0 and \
                   svolume.y1 == y1 and \
                   svolume.z == z0 and \
                   svolume.z1 == z1:
                    # Cheap win, return the block
                    rh_logger.logger.report_metric("Dataset load time (sec)",
                                                       time.time() - t0)
                    return block.astype(datatype)
                if result is None:
                    result = np.zeros((z1-z0, y1-y0, x1-x0), datatype)
                #
                # Defensively trim the block to within x0:x1, y0:y1, z0:z1
                #
                if svolume.z < z0:
                    block = block[z0-svolume.z:]
                    sz0 = z0
                else:
                    sz0 = svolume.z
                if svolume.z1 > z1:
                    block = block[:z1 - sz0]
                    sz1 = z1
                else:
                    sz1 = svolume.z1
                if svolume.y < y0:
                    block = block[:, y0-svolume.y:]
                    sy0 = y0
                else:
                    sy0 = svolume.y
                if svolume.y1 > y1:
                    block = block[:, :y1 - sy0]
                    sy1 = y1
                else:
                    sy1 = svolume.y1
                if svolume.x < x0:
                    block = block[:, :, x0-svolume.x:]
                    sx0 = x0
                else:
                    sx0 = svolume.x
                if svolume.x1 > x1:
                    block = block[:, :, :x1 - sx0]
                    sx1 = x1
                else:
                    sx1 = svolume.x1
                result[sz0 - z0:sz1 - z0,
                       sy0 - y0:sy1 - y0,
                       sx0 - x0:sx1 - x0] = block
        rh_logger.logger.report_metric("Dataset load time (sec)",
                                       time.time() - t0)
        return result
    
    def imwrite(self, dataset_id, data, compression=0):
        '''Write the block of data to the dataset
        
        :param dataset_id: The ID of the dataset to be written
        :param data: a 3d Numpy array to be written
        '''
        t0 = time.time()
        dataset = self.session.query(DatasetObj).filter(
            DatasetObj.dataset_id == dataset_id).first()
        assert isinstance(dataset, DatasetObj)
        x0 = dataset.volume.volume().x
        x1 = dataset.volume.volume().x1
        y0 = dataset.volume.volume().y
        y1 = dataset.volume.volume().y1
        z0 = dataset.volume.volume().z
        z1 = dataset.volume.volume().z1
        datatype = getattr(np, dataset.dataset_type.datatype)
        
        rh_logger.logger.report_event(
            "Writing %s: %d:%d, %d:%d, %d:%d" %
            (dataset.dataset_type.name, x0, x1, y0, y1, z0, z1))
        
        clauses = [
            DatasetSubvolumeObj.dataset_id == dataset_id,
            
            DatasetSubvolumeObj.volume_id == VolumeObj.volume_id,
           
            SubvolumeLocationObj.subvolume_id == 
            DatasetSubvolumeObj.subvolume_id
        ]
        for subvolume_location, svolume in self.session.query(
            SubvolumeLocationObj, VolumeObj).filter(
                sqlalchemy.and_(*clauses)):
            tif_path = subvolume_location.location
            svolume = svolume.volume()
            sx0 = svolume.x
            sx1 = svolume.x1
            sy0 = svolume.y
            sy1 = svolume.y1
            sz0 = svolume.z
            sz1 = svolume.z1
            tif_dir = os.path.dirname(tif_path)
            if not os.path.isdir(tif_dir):
                os.makedirs(tif_dir)
            with tifffile.TiffWriter(tif_path, bigtiff=True) as fd:
                metadata = dict(x0=sx0, x1=sx1, y0=sy0, y1=sy1, z0=sz0, z1=sz1,
                                dataset_name = dataset.dataset_type.name,
                                dataset_id = dataset_id)
                block = data[sz0 - z0: sz1 - z0,
                             sy0 - y0: sy1 - y0,
                             sx0 - x0: sx1 - x0].astype(datatype)
                fd.save(block, 
                        photometric='minisblack',
                        compress=compression,
                        description=dataset.dataset_type.name,
                        metadata=metadata)
        rh_logger.logger.report_metric("Dataset store time (sec)",
                                       time.time() - t0)
    
    def get_dataset_path(self, dataset_id):
        '''Get the canonical path to the dataset's "done" file'''
        dataset = self.session.query(DatasetObj).filter(
            DatasetObj.dataset_id == dataset_id).first()
        return self._get_dataset_path_by_dataset(dataset)
    
    def _get_dataset_path_by_dataset(self, dataset):
        persistence = dataset.dataset_type.persistence
        dataset_name = dataset.dataset_type.name
        if persistence == Persistence.Permanent:
            root = self.target_dir
        else:
            root = self.temp_dir
        volume = dataset.volume.volume()
        return get_storage_plan_path(root, dataset.dataset_id, volume, 
                                    dataset_name)

    def get_dataset_volume(self, dataset_id):
        '''Get the volume encompassed by this dataset
        
        :param dataset_id: the ID of the dataset to be retrieved
        :returns: a ariadne_microns_pipeline.parameters.volume
        '''
        volume = self.session.query(DatasetObj).filter(
            DatasetObj.dataset_id == dataset_id).first().volume
        return volume.volume()
    
    def get_dataset_paths_by_loading_plan_id(self, loading_plan_id):
        '''Get the .done file locations for a loading plan's datasets
        
        :param loading_plan_id: the loading plan for the dataset
        :returns: a list of .done file locations
        '''
        datasets = self.session.query(DatasetObj).filter(
            sqlalchemy.and_(
                DatasetObj.dataset_id == DatasetDependentObj.dataset_id,
                DatasetDependentObj.loading_plan_id == loading_plan_id)).all()
        return map(self._get_dataset_path_by_dataset, datasets)
    
    def get_loading_plan_volume(self, loading_plan_id):
        '''Get the volume for a loading plan
        
        :param loading_plan_id: the loading plan for loading a dataset
        :returns: a ariadne_microns_pipeline.parameters.Volume describing
        the dataset's extent
        '''
        volume = self.session.query(VolumeObj).filter(
            sqlalchemy.and_(
                VolumeObj.volume_id == LoadingPlanObj.volume_id,
                LoadingPlanObj.loading_plan_id == loading_plan_id)).first()
        return volume.volume()
    
    def get_loading_plan_dataset_name(self, loading_plan_id):
        '''Get the loading plan's dataset name, e.g. "image"
        
        :param loading_plan_id: the ID of the loading plan record
        '''
        return self.session.query(DatasetTypeObj.name).filter(sqlalchemy.and_(
            DatasetTypeObj.dataset_type_id == LoadingPlanObj.dataset_type_id,
            LoadingPlanObj.loading_plan_id == loading_plan_id
            )).first()[0]
    
    def get_loading_plan_dataset_type(self, loading_plan_id):
        '''Get the loading plan's Numpy dtype, e.g. "uint8"
        
        :param loading_plan_id: the ID of the loading plan record
        '''
        return self.session.query(DatasetTypeObj.datatype).filter(
            sqlalchemy.and_(
                DatasetTypeObj.dataset_type_id == 
                LoadingPlanObj.dataset_type_id,
                LoadingPlanObj.loading_plan_id == loading_plan_id
            )).first()[0]
    
    def get_loading_plan_dataset_ids(self, loading_plan_id):
        '''Get the dataset_ids for the datasets referenced by a loading plan
        
        :param loading_plan_id: the ID of the loading plan for reading a  volume
        '''
        return self.session.query(DatasetDependentObj.dataset_id).filter(
            DatasetDependentObj.loading_plan_id == loading_plan_id).all()

all = [VolumeDB, Persistence, get_loading_plan_path, get_storage_plan_path]