/////////////////////////////////////////////////////////////////////////////////////////
//
//
//
/////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////
// INCLUDES
/////////////////////////////////////////////////////////////////////////////////////////
#include "common.h"
#include "interweave.h"

/////////////////////////////////////////////////////////////////////////////////////////
// DEFINES
/////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////
// TYPES
/////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////
// GLOBALS
/////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////
// INTERNAL FUNCTIONS
/////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////
// EXTERNAL FUNCTIONS
/////////////////////////////////////////////////////////////////////////////////////////
void interweave(float * __restrict__ p_in_matrix,
                int n_matrices,
                int m_d1,
                int m_d2,
                int m_d3,
                int channel_id, 
                int shift_d1,
                int shift_d2,
                int n_depth_shift,
                float * __restrict__ p_out_matrix,
                int out_d1,
                int out_d2) {
                     
    DNN_TRACE_4("start %s\n","");
    DNN_TRACE_4(" p_in_matrix: %p\n", p_in_matrix);
    DNN_TRACE_4(" n_matrices: %d\n", n_matrices);
    DNN_TRACE_4(" m_d1,m_d2,m_d3: [%d,%d,%d]\n", m_d1, m_d2, m_d3);
    DNN_TRACE_4(" channel_id: %d\n", channel_id);
    DNN_TRACE_4(" shift_d1,shift_d2: [%d,%d]\n", shift_d1, shift_d2);
    DNN_TRACE_4(" n_depth_shift: %d\n", n_depth_shift);
    DNN_TRACE_4(" p_out_matrix: %p\n", p_out_matrix);
    DNN_TRACE_4(" out_d1,out_d2: [%d,%d]\n", out_d1, out_d2);
    
    //DNN_ASSERT(out_d1 == (m_d1 * n_depth_shift));
    //DNN_ASSERT(out_d2 == (m_d2 * n_depth_shift));
    
    int cur_out_d1 = shift_d1;
    for (int cur_m_d1 = 0; cur_m_d1 < m_d1; cur_m_d1++) {
        
        if (cur_out_d1 >= out_d1) {
            continue;
        }
        
        int offset_d1 = D1_TO_OFFSET(cur_m_d1, m_d2, m_d3);
        
        int offset_out_d1 = D1_TO_OFFSET(cur_out_d1, out_d2, 1);
        
        int cur_out_d2 = shift_d2;
        for (int cur_m_d2 = 0; cur_m_d2 < m_d2; cur_m_d2++) {
            
            if (cur_out_d2 >= out_d2) {
                continue;
            }
            
            int offset_d2 = D2_TO_OFFSET(cur_m_d2, m_d3);
            
            int offset_out_d2 = D2_TO_OFFSET(cur_out_d2, 1);
            
            for (int m_id = 0; m_id < n_matrices; m_id++) {
                int offset_matrix = DIMS_TO_SIZE(m_d1, m_d2, m_d3) * m_id;
                
                float num = p_in_matrix[offset_matrix + offset_d1 + offset_d2 + channel_id];
                
                p_out_matrix[offset_out_d1 + offset_out_d2 + m_id] = num;
            }
            
            cur_out_d2 += n_depth_shift;
        }
        
        cur_out_d1 += n_depth_shift;
    }
    
    DNN_TRACE_4("finish %s\n","");
}

void recursive_interweave(float * __restrict__ p_in_matrices,
                          int n_matrices,
                          int m_d1,
                          int m_d2,
                          int m_d3,
                          int channel_id,
                          int in_shift_d1,
                          int in_shift_d2,
                          int n_depth_shift,
                          float * __restrict__ p_out_matrix,
                          int out_d1,
                          int out_d2) {
    
    /*if (n_matrices == 1) {
        interweave(p_in_matrices,
                   1,
                   m_d1,
                   m_d2,
                   m_d3,
                   channel_id, 
                   in_shift_d1,
                   in_shift_d2,
                   n_depth_shift,
                   p_out_matrix,
                   out_d1,
                   out_d2);
        return;
    }*/
    
    CILK_FOR_M (int shift_d1 = 0; shift_d1 < 2; shift_d1++) {
        CILK_FOR_M (int shift_d2 = 0; shift_d2 < 2; shift_d2++) {

            int block_id = shift_d1 * 2 + shift_d2;
            int n_matrices_in_block = n_matrices / 4;
            int matrix_size = DIMS_TO_SIZE(m_d1, m_d2, m_d3);
            
            if (n_matrices_in_block == 1) {
                interweave(p_in_matrices + ((matrix_size * n_matrices_in_block) * block_id),
                           1,
                           m_d1,
                           m_d2,
                           m_d3,
                           channel_id, 
                           in_shift_d1 + shift_d1 * n_depth_shift,
                           in_shift_d2 + shift_d2 * n_depth_shift,
                           n_depth_shift * 2,
                           p_out_matrix,
                           out_d1,
                           out_d2);
                
            } else {
                recursive_interweave(p_in_matrices + ((matrix_size * n_matrices_in_block) * block_id),
                                    n_matrices_in_block,
                                    m_d1,
                                    m_d2,
                                    m_d3,
                                    channel_id,
                                    in_shift_d1 + shift_d1 * n_depth_shift,
                                    in_shift_d2 + shift_d2 * n_depth_shift,
                                    n_depth_shift * 2,
                                    p_out_matrix,
                                    out_d1,
                                    out_d2);
            }
        }
    }
}