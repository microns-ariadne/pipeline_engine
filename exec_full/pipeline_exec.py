
import os
import sys
import argparse

from pipeline_common import *
from pipeline_engine import *

###############################################################################
# Phases
###############################################################################    

PHASE_FUNCS = {
    
    'ALIGN_GENERATE_TILES' : ALIGN_GENERATE_TILES_execute,
    'ALIGN_GENERATE_TILES_TXT' : ALIGN_GENERATE_TILES_TXT_execute,
    'ALIGN_COMPUTE_KPS_AND_MATCH' : ALIGN_COMPUTE_KPS_AND_MATCH_execute,
    'ALIGN_COMPUTE_TRANSFORMS' : ALIGN_COMPUTE_TRANSFORMS_execute,
    'ALIGN_COMPUTE_WARPS' : ALIGN_COMPUTE_WARPS_execute,
    
    'GENERATE_BLOCKS' : GENERATE_BLOCKS_execute,
    
    'ANALYZE_BLOCKS' : ANALYZE_BLOCKS_execute,

    'CNN' : CNN_execute,
    
    'WS_NP_PREPARE' : WS_NP_PREPARE_execute,
    
    'WS' : WS_execute,

    'NP_PREPARE' : NP_PREPARE_execute,
    
    'NP' : NP_execute,
    
    'MERGE_PREPROCESS' : MERGE_PREPROCESS_exec,
    'MERGE_BLOCK' : MERGE_BLOCK_exec,
    'MERGE_COMBINE' : MERGE_COMBINE_exec,
    'MERGE_RELABEL' : MERGE_RELABEL_exec,
    
    'SCATTER_POINTS' : SCATTER_POINTS_execute,
    
    'SKELETONS' : SKELETONS_execute,
    
    'DEBUG_GENERATE' : DEBUG_GENERATE_exec,
}

def parse_range(in_str):
    try:
        parts = in_str.split('-')
        start_num = int(parts[0])
        finish_num = int(parts[1])
    except ValueError:
        raise ArgumentTypeError('unexpected range = %s (must be [N1]-[N2])' % (in_str,))
    
    return (start_num, finish_num)
    
def get_sections(args):
    
    sec_start, sec_finish = parse_range(args.section_range)
    
    sections = [ sec_id for sec_id in xrange(sec_start, sec_finish)]
    
    return sections

def get_blocks(args):
    
    z_start, z_finish = parse_range(args.z_range)
    x_start, x_finish = parse_range(args.x_range)
    y_start, y_finish = parse_range(args.y_range)
    
    blocks = [ \
        (z,x,y) for z in xrange(z_start,z_finish) \
        for x in xrange(x_start,x_finish) \
        for y in xrange(y_start,y_finish) ]
    
    return blocks
    
def pipeline_execute(args, ctx):
    
    assert (ctx.phase in PHASE_FUNCS.keys()), 'unexpected phase = %s' % (ctx.phase,)
    
    func = PHASE_FUNCS[ctx.phase]
    
    q_msg = 'Execute phase = %s' % (ctx.phase,)
    
    if ctx.phase == 'GENERATE_BLOCKS':
        q_msg += ' on sections = [%s]' % (args.section_range,)
    else:
        q_msg += ' on blocks Z-X-Y=[%s][%s][%s]' % (
            args.z_range,
            args.x_range,
            args.y_range)
    
    assert_user_verify(ctx, q_msg)
    
    func(ctx)
    
if '__main__' == __name__:
    parser = argparse.ArgumentParser('Pipeline execute')
    
    parser.add_argument(
        '--force',
        dest='is_force',
        action="store_true",
        help='No interactive questions')
    
    parser.add_argument(
        '--phase',
        dest='phase',
        type=str,
        required=True,
        help='Pipeline phase to execute')
        
    parser.add_argument(
        '--Z-range',
        dest='z_range',
        type=str,
        required=True,
        help='Z range for blocks')
    
    parser.add_argument(
        '--X-range',
        dest='x_range',
        type=str,
        required=True,
        help='X range for blocks')
    
    parser.add_argument(
        '--Y-range',
        dest='y_range',
        type=str,
        required=True,
        help='Y range for blocks')
    
    #
    parser.add_argument(
        '--section-range',
        dest='section_range',
        type=str,
        required=True,
        help='section range for pre-block pipeline phases')
    
    args = parser.parse_args()
    
    sections_to_process = get_sections(args)
    blocks_to_process = get_blocks(args)
        
    ctx = PipelineContext()
    
    ctx.set_input_params(
        args.is_force,
        args.phase,
        sections_to_process,
        blocks_to_process)
    
    ctx.set_data_dirs(
        DATA_DIR_LIST)
        
    pipeline_execute(args, ctx)
    
