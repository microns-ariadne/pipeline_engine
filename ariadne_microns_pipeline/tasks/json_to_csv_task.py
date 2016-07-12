import csv
import json
import luigi
from .utilities import RequiresMixin

class JSONToCSVTaskMixin:
    
    json_paths = luigi.ListParameter(
        description="The paths of the JSON files to be combined")
    output_path = luigi.Parameter(
        description="The path to the CSV file to be created")
    
    def input(self):
        for path in self.json_paths:
            yield luigi.LocalTarget(path)
    
    def output(self):
        return luigi.LocalTarget(self.output_path)

class JSONToCSVRunMixin:
    def ariadne_run(self):
        dd = [json.load(inp.open("r")) for inp in self.input()]
        keys = sorted(set.union(*[set(_.keys()) for _ in dd]))
        with self.output().open("w") as fd:
            writer = csv.writer(fd)
            writer.writerow(keys)
            for d in dd:
                row = [d.get(k, "") for k in keys]
                writer.writerow(row)

class JSONToCSVTask(JSONToCSVTaskMixin, JSONToCSVRunMixin,
                    RequiresMixin, luigi.Task):
    '''This task compiles a series of separate JSON files into one CSV
    
    Each of the inputs to the task is the filename of a JSON file containing
    a dictionary. The task creates a CSV file where the columns are the
    keys (all keys in all JSON files) and the rows are the values for each
    JSON file
    '''
    
    task_namespace="ariadne_microns_pipeline"

    def run(self):
        self.ariadne_run()