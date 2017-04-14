# Gannt chart for Luigi
# Takes two arguments
#    the URL for sqlalchemy, e.g. sqlite:////path/to/luigi-task-hist.db
#    the name of the .PDF to generate
import json
import numpy as np
import sqlalchemy
import matplotlib
matplotlib.use("agg")
import matplotlib.backends.backend_pdf
from matplotlib.patches import Patch
from matplotlib.cm import ScalarMappable
from pylab import *
import dateutil
import sys
import os

def draw_gantt(database_url, pdf_file, xy_scale, z_scale):
    stmt = """
    select t.id, t.name, te1.ts as start_time, te2.ts as end_time, 
       tp.value from tasks t 
       join task_events te1 on t.id = te1.task_id 
       join task_events te2 on t.id = te2.task_id 
       left outer join (select task_id, max(value) as value from task_parameters 
                         where value like "%.plan" group by task_id) tp
                         on t.id = tp.task_id
       where te1.event_name = "RUNNING" and te2.event_name="DONE"
             and t.name != "ariadne_microns_pipeline.StitchSegmentationTask"
       order by te1.ts
    """
    
    if xy_scale == 0:
        by_time = True
    else:
        by_time = False
    engine = sqlalchemy.create_engine(database_url)
    result = engine.execute(stmt)
    data = result.fetchall()
    s0 = dateutil.parser.parse(data[0]['start_time'])
    #
    # Build a dictionary of volume string to x, y, z for sorting
    #
    od = { None:0}
    def octree_encode(x, y, z):
        accumulator = 0
        addend = 1
        n = 1
        while n < x or n < y or n < z:
            if n & x:
                accumulator += addend
            addend += addend
            if n & y:
                accumulator += addend
            addend += addend
            if n & z:
                accumulator += addend
            addend += addend
            n += n
        return accumulator
    
    if not by_time:
        for item in data:
            storage_plan = item['value']
            if storage_plan not in od and storage_plan is not None:
                spd = os.path.dirname(storage_plan)
                spd, z = os.path.split(spd)
                spd, y = os.path.split(spd)
                spd, x = os.path.split(spd)
                x = int(x) * xy_scale
                y = int(y) * xy_scale
                z = int(z) * z_scale
                od[storage_plan] = octree_encode(x, y, z)
    
            def compare(a, b):
                av = a['value']
                bv = b['value']
                if av == bv:
                    return cmp(dateutil.parser.parse(a['start_time']),
                               dateutil.parser.parse(b['start_time']))
                return cmp(od[av], od[bv])
        
            
        data = sorted(data, cmp=compare)
    
    sm = ScalarMappable(cmap="jet")
    names = set([_['name'] for _ in data])
    colors = sm.to_rgba(range(len(names)))
    colors = dict([(name, colors[i]) for i, name in enumerate(names)])
    rcParams['figure.figsize'] = (100, 100)
    rcParams['legend.fontsize'] = 60
    rcParams['axes.labelsize'] = 60
    rcParams['axes.titlesize'] = 60
    rcParams['xtick.labelsize'] = 30
    if by_time:
        smin = dateutil.parser.parse(data[0]['start_time'])
        emax = dateutil.parser.parse(data[0]['end_time'])
        for d in data:
            s = dateutil.parser.parse(d['start_time'])
            if s < smin:
                smin = s
            e = dateutil.parser.parse(d['end_time'])
            if e > emax:
                emax = e
        bar_height = (emax - smin).total_seconds() / 100
    else:
        bar_height = np.max(od.values()) / 100
    bottom = []
    width = []
    height = bar_height
    left = []
    color = []
    for i, d in enumerate(data):
        s = dateutil.parser.parse(d['start_time'])
        e = dateutil.parser.parse(d['end_time'])
        delta = e-s
        volume = d['value']
        if by_time:
            ht = (s - smin).total_seconds()
        else:
            ht = float(od[volume])
        bottom.append(ht)
        width.append(delta.total_seconds())
        left.append((s-s0).total_seconds())
        color.append(colors[d['name']])
    barh(bottom, width, height=height, left=left, color=color)
    legend(handles=[Patch(color=color, label=name) for name, color in colors.items()], loc=0)
    gca().set_xlabel("Exection time (sec)")
    gca().set_yticks([])
    gcf().savefig(pdf_file)
