import csv
import dateutil
import rh_config
import rh_logger
import luigi
import matplotlib
import matplotlib.backends.backend_pdf
import numpy as np
import os
import sqlalchemy
import datetime

class PipelineRunReportMixin:
    '''This mixin adds a timing report generator to a pipeline
    
    The .rh_config.yaml should have the following section:
    
    luigid:
        db_connection=<sqlalchemy db connection>
    
    For instance, sqllite:////<path-to-luigi-task-hist.db>
        
    '''
    
    pipeline_report_location = luigi.Parameter(
        description="Location for the timing report .csv")

    def output(self):
        return luigi.LocalTarget(self.pipeline_report_location)
    
    def run(self):
        '''Compile the task history'''
        if hasattr(self, "ariadne_run"):
            self.ariadne_run()
        matplotlib.use("Pdf")
        conn_params = rh_config.config["luigid"]["db_connection"]
        rh_logger.logger.report_event("Fetching task history")
        engine = sqlalchemy.create_engine(conn_params)
        d = self.get_task_history(self, engine)
        rh_logger.logger.report_event("%d events retrieved" % len(d))
        tasks = {}
        tmin = datetime.datetime.now()
        for task_id, (delta, t0, t1) in d.items():
            if task_id.startswith("ariadne_microns_pipeline."):
                task_id = task_id.split(".", 1)[1]
            task = task_id.split("_", 1)[0]
            if task not in tasks:
                tasks[task] = []
            tasks[task].append(delta.total_seconds())
            if t0 < tmin:
                tmin = t0
        task_names = sorted(tasks.keys())
        timings = map(lambda k:tasks[k], task_names)
        self.write_pdf_report(task_names, timings)
            
        with self.output().open("w") as fd:
            writer = csv.writer(fd)
            writer.writerow(["task_id", "run_time", "start", "end"])
            for task_id, (delta, t0, t1) in d.items():
                writer.writerow((task_id, 
                                 str(delta), 
                                 str((t0 - tmin).total_seconds()), 
                                 str((t1 - tmin).total_seconds())))

    def write_pdf_report(self, task_names, timings):
        pdf_path = os.path.splitext(self.pipeline_report_location)[0] + ".pdf"
        with matplotlib.backends.backend_pdf.PdfPages(pdf_path) as pdf:
            #
            # boxplots of individual timings
            #
            rh_logger.logger.report_event(
                "Plotting boxplot of timing variances per task.")
            figure = matplotlib.pyplot.figure()
            ax = figure.add_axes([0.1, 0.3, 0.8, 0.65])
            ax.boxplot(timings, labels=task_names)
            labels = ax.get_xmajorticklabels()
            matplotlib.pyplot.setp(labels, 
                                   rotation=45, 
                                   horizontalalignment='right',
                                   size=8)
            ax.set_title("Task runtime (sec)")
            pdf.savefig(figure)
            #
            # bar plot of total timing
            #
            rh_logger.logger.report_event("Plotting bar plot of total sec/task")
            figure = matplotlib.pyplot.figure()
            ax = figure.add_axes([0.1, 0.3, 0.8, 0.65])
            colors = ((0, 0, 1.0), (0, .4, .6))
            for i in range(np.max(map(len, timings))):
                idx = [_ for _ in range(len(timings))
                       if len(timings[_]) > i]
                if i == 0:
                    bottom = [0] * len(idx)
                else:
                    bottom = [np.sum(timings[_][:i]) for _ in idx]
                ax.bar(idx, 
                       [timings[_][i] for _ in idx], 
                       color=colors[i % len(colors)],
                       bottom=bottom)
            ax.set_xticks(np.arange(len(timings)) + .5)
            ax.set_xticklabels(task_names,
                               rotation=45, 
                               horizontalalignment='right',
                               size=8)
            total_runtime = np.sum([np.sum(_) for _ in timings])
            voxels = self.volume.width * self.volume.height * self.volume.depth
            mvoxel_per_sec = float(voxels) / 1000 / 1000 / total_runtime
            ax.set_title(
                "Total runtime by task (sec): %.2f sec, %.2f M voxels/sec" %
                (total_runtime, mvoxel_per_sec))
            pdf.savefig(figure)
            rh_logger.logger.report_event("Finished plotting %s" % pdf_path)
    
    def get_task_history(self, task, engine):
        '''Get the run history of the task graph
        
        :param task: the root of the dependency graph
        :param engine: a sqlalchemy engine for the Luigi database.
        :returns: a dictionary of task_id to runtime in seconds, start time
        and end time
        '''
        stack = [task]
        all_tasks = set()
        while len(stack) > 0:
            t = stack.pop()
            if t.task_id in all_tasks:
                continue
            all_tasks.add(t.task_id)
            stack += list(t.requires())
        sql = """
        SELECT t.task_id, te.event_name, te.ts
        FROM tasks t JOIN task_events te
        ON t.id = te.task_id
        WHERE te.event_name != 'PENDING'
        AND t.task_id in ('%s') order by t.task_id, te.ts""" %\
            "','".join(list(all_tasks))
        result = engine.execute(sql)
        t0 = {}
        t1 = {}
        for task_id, event_name, ts in result:
            if event_name == luigi.task_status.RUNNING:
                t0[task_id] = dateutil.parser.parse(ts)
                if task_id in t1:
                    del t1[task_id]
            elif event_name == luigi.task_status.DONE and task_id not in t1:
                t1[task_id] = dateutil.parser.parse(ts)
        d = {}
        for task_id in t0:
            if task_id in t1:
                delta = t1[task_id] - t0[task_id]
                d[task_id] = (delta, t0[task_id], t1[task_id])
        return d
        
        
        