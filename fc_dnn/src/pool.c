/////////////////////////////////////////////////////////////////////////////////////////
//
//
//
/////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////
// INCLUDES
/////////////////////////////////////////////////////////////////////////////////////////
#include <immintrin.h>

#include "common.h"
#include "pool.h"


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
void max_pool_AVX_8ch(
    float * __restrict__ p_in_matrix, 
    int m_d1,
    int m_d2,
    int m_d1_start,
    int m_d2_start,
    int m_d1_finish,
    int m_d2_finish,
    int k_d1,
    int k_d2,
    int n_channels,
    int n_stride,
    float * __restrict__ p_out_matrix)
{    
    int out_d2;
    
    DNN_TRACE_4("start %s\n","");
    DNN_TRACE_4(" p_in_matrix: %p\n", p_in_matrix);
    DNN_TRACE_4(" m_d1,m_d2: [%d,%d]\n", m_d1, m_d2);
    DNN_TRACE_4(" s/e_d1,s/e_d2: [%d,%d] <=> [%d,%d]\n", m_d1_start, m_d2_start, m_d1_finish, m_d2_finish);
    DNN_TRACE_4(" k_d1,k_d2: [%d,%d]\n", k_d1, k_d2);
    DNN_TRACE_4(" n_channels,n_stride: [%d,%d]\n", n_channels, n_stride);
    DNN_TRACE_4(" p_out_matrix: %p\n", p_out_matrix);
    
    out_d2 = OUT_DIM(m_d2, k_d2, n_stride);
    
    //int cur_out_d1 = -1;
    for (int cur_m_d1 = m_d1_start; (cur_m_d1 + k_d1) <= m_d1_finish; cur_m_d1 += n_stride) {
        //cur_out_d1++;
        
        int cur_out_d1 = cur_m_d1 / n_stride;
        int out_offset_d1 = D1_TO_OFFSET(cur_out_d1, out_d2, n_channels);
        
        //int cur_out_d2 = -1;
        for (int cur_m_d2 = m_d2_start; (cur_m_d2 + k_d2) <= m_d2_finish; cur_m_d2 += n_stride) {
            //cur_out_d2++;
            
            int is_window_outside = ((cur_m_d1 + k_d1) > m_d1) || ((cur_m_d2 + k_d2) > m_d2);
            
            int cur_out_d2 = cur_m_d2 / n_stride;
            int out_offset_d2 = D2_TO_OFFSET(cur_out_d2, n_channels);
            
            __m256 ymm0;
            __m256 ymm1;
            __m256 ymm2;
            __m256 ymm15 = _mm256_xor_ps(ymm15, ymm15); // init to 0s
                        
            int in_offset_init = D1_TO_OFFSET(cur_m_d1, m_d2, n_channels) + D2_TO_OFFSET(cur_m_d2, n_channels);
            
            for (int k = 0; k < n_channels; k += 8) {
                ymm0 = _mm256_load_ps(&p_in_matrix[in_offset_init + k]);
                
                if (!is_window_outside) {
                    _mm256_store_ps(&p_out_matrix[out_offset_d1 + out_offset_d2 + k], ymm0); // init to first numbers
                } else {
                    _mm256_store_ps(&p_out_matrix[out_offset_d1 + out_offset_d2 + k], ymm15); // init to 0s
                }
            }
            
            if (is_window_outside) {
                continue;
            }
            
            for (int i = 0; i < k_d1; i++) {
                
                int in_offset_d1 = D1_TO_OFFSET(cur_m_d1 + i, m_d2, n_channels);
                
                for (int j = 0; j < k_d2; j += 2) {
                    
                    int in_offset_d2 = D2_TO_OFFSET(cur_m_d2 + j, n_channels);
                    
                    for (int k = 0; k < n_channels; k += 8) {
                        ymm0 = _mm256_load_ps(&p_in_matrix[in_offset_d1 + in_offset_d2 + k]);
                        
                        ymm1 = _mm256_load_ps(&p_in_matrix[in_offset_d1 + in_offset_d2 + k + n_channels]);
                        
                        ymm0 = _mm256_max_ps(ymm0, ymm1);
                        
                        ymm2 = _mm256_load_ps(&p_out_matrix[out_offset_d1 + out_offset_d2 + k]);
                        
                        ymm0 = _mm256_max_ps(ymm0, ymm2);
                        
                        _mm256_store_ps(&p_out_matrix[out_offset_d1 + out_offset_d2 + k], ymm0);
                    }                        
                    
                }
            }
            
        }
        
    }
    
    DNN_TRACE_4("finish %s\n","");
}

void max_pool_AVX_8ch_exact_k2_k2(
    float * __restrict__ p_in_matrix, 
    int m_d1,
    int m_d2,
    int m_d1_start,
    int m_d2_start,
    int m_d1_finish,
    int m_d2_finish,
    int k_d1,
    int k_d2,
    int n_channels,
    int n_stride,
    float * __restrict__ p_out_matrix)
{    
    int out_d2;
    
    DNN_TRACE_4("start %s\n","");
    DNN_TRACE_4(" p_in_matrix: %p\n", p_in_matrix);
    DNN_TRACE_4(" m_d1,m_d2: [%d,%d]\n", m_d1, m_d2);
    DNN_TRACE_4(" s/e_d1,s/e_d2: [%d,%d] <=> [%d,%d]\n", m_d1_start, m_d2_start, m_d1_finish, m_d2_finish);
    DNN_TRACE_4(" k_d1,k_d2: [%d,%d]\n", k_d1, k_d2);
    DNN_TRACE_4(" n_channels,n_stride: [%d,%d]\n", n_channels, n_stride);
    DNN_TRACE_4(" p_out_matrix: %p\n", p_out_matrix);
    
    DNN_ASSERT(n_channels == N_CHANNELS_8);
    DNN_ASSERT(k_d1 == K_D1_2);
    DNN_ASSERT(k_d2 == K_D2_2);
    
    out_d2 = OUT_DIM(m_d2, K_D2_2, n_stride);
    
    //int cur_out_d1 = -1;
    for (int cur_m_d1 = m_d1_start; (cur_m_d1 + K_D1_2) <= m_d1_finish; cur_m_d1 += n_stride) {
        //cur_out_d1++;
        
        int cur_out_d1 = cur_m_d1 / n_stride;
        int out_offset_d1 = D1_TO_OFFSET(cur_out_d1, out_d2, N_CHANNELS_8);
        
        //int cur_out_d2 = -1;
        for (int cur_m_d2 = m_d2_start; (cur_m_d2 + K_D2_2) <= m_d2_finish; cur_m_d2 += n_stride) {
            //cur_out_d2++;
            
            int is_window_outside = ((cur_m_d1 + K_D1_2) > m_d1) || ((cur_m_d2 + K_D2_2) > m_d2);
            
            int cur_out_d2 = cur_m_d2 / n_stride;
            int out_offset_d2 = D2_TO_OFFSET(cur_out_d2, n_channels);
            
            __m256 ymm0;
            __m256 ymm1;
            __m256 ymm2;
            __m256 ymm15 = _mm256_xor_ps(ymm15, ymm15); // init to 0s
                        
            int in_offset_init = D1_TO_OFFSET(cur_m_d1, m_d2, N_CHANNELS_8) + D2_TO_OFFSET(cur_m_d2, N_CHANNELS_8);
            
            for (int k = 0; k < N_CHANNELS_8; k += 8) {
                ymm0 = _mm256_load_ps(&p_in_matrix[in_offset_init + k]);
                
                if (!is_window_outside) {
                    _mm256_store_ps(&p_out_matrix[out_offset_d1 + out_offset_d2 + k], ymm0); // init to first numbers
                } else {
                    _mm256_store_ps(&p_out_matrix[out_offset_d1 + out_offset_d2 + k], ymm15); // init to 0s
                }
            }
            
            if (is_window_outside) {
                continue;
            }
            
            for (int i = 0; i < K_D1_2; i++) {
                
                int in_offset_d1 = D1_TO_OFFSET(cur_m_d1 + i, m_d2, N_CHANNELS_8);
                
                for (int j = 0; j < K_D2_2; j += 2) {
                    
                    int in_offset_d2 = D2_TO_OFFSET(cur_m_d2 + j, N_CHANNELS_8);
                    
                    for (int k = 0; k < N_CHANNELS_8; k += 8) {
                        ymm0 = _mm256_load_ps(&p_in_matrix[in_offset_d1 + in_offset_d2 + k]);
                        
                        ymm1 = _mm256_load_ps(&p_in_matrix[in_offset_d1 + in_offset_d2 + k + N_CHANNELS_8]);
                        
                        ymm0 = _mm256_max_ps(ymm0, ymm1);
                        
                        ymm2 = _mm256_load_ps(&p_out_matrix[out_offset_d1 + out_offset_d2 + k]);
                        
                        ymm0 = _mm256_max_ps(ymm0, ymm2);
                        
                        _mm256_store_ps(&p_out_matrix[out_offset_d1 + out_offset_d2 + k], ymm0);
                    }                        
                    
                }
            }
            
        }
        
    }
    
    DNN_TRACE_4("finish %s\n","");
}

void max_pool_AVX_1ch(float * __restrict__ p_in_matrix, 
              int m_shift_d1,
              int m_shift_d2,
              int m_d1,
              int m_d2,
              int m_pad_d1,
              int m_pad_d2,
              int k_d1,
              int k_d2,
              int n_channels,
              int n_stride,
              float * __restrict__ p_out_matrix)
{    
    int out_d2;
    
    DNN_TRACE_4("start %s\n","");
    DNN_TRACE_4(" p_in_matrix: %p\n", p_in_matrix);
    DNN_TRACE_4(" s_d1,s_d2: [%d,%d]\n", m_shift_d1, m_shift_d2);
    DNN_TRACE_4(" m_d1,m_d2: [%d,%d]\n", m_d1, m_d2);
    DNN_TRACE_4(" p_d1,p_d2: [%d,%d]\n", m_pad_d1, m_pad_d2);
    DNN_TRACE_4(" k_d1,k_d2: [%d,%d]\n", k_d1, k_d2);
    DNN_TRACE_4(" n_channels,n_stride: [%d,%d]\n", n_channels, n_stride);
    DNN_TRACE_4(" p_out_matrix: %p\n", p_out_matrix);
    
    out_d2 = OUT_DIM(m_d2 + m_pad_d2, k_d2, n_stride);
    
    int cur_out_d1 = -1;
    for (int cur_m_d1 = m_shift_d1; (cur_m_d1 + k_d1) <= (m_d1 + m_pad_d1); cur_m_d1 += n_stride) {
        cur_out_d1++;
        
        int out_offset_d1 = D1_TO_OFFSET(cur_out_d1, out_d2, n_channels);
        
        int cur_out_d2 = -1;
        for (int cur_m_d2 = m_shift_d2; (cur_m_d2 + k_d2) <= (m_d2 + m_pad_d2); cur_m_d2 += n_stride) {
            cur_out_d2++;
            
            int is_window_outside = ((cur_m_d1 + k_d1) > m_d1) || ((cur_m_d2 + k_d2) > m_d2);
            
            int out_offset_d2 = D2_TO_OFFSET(cur_out_d2, n_channels);
            
            int in_offset_init = D1_TO_OFFSET(cur_m_d1, m_d2, n_channels) + D2_TO_OFFSET(cur_m_d2, n_channels);
            
            for (int k = 0; k < n_channels; k += 1) {
                if (!is_window_outside) {
                    p_out_matrix[out_offset_d1 + out_offset_d2 + k] = p_in_matrix[in_offset_init + k];
                } else {
                    p_out_matrix[out_offset_d1 + out_offset_d2 + k] = 0;
                }
            }
            
            if (is_window_outside) {
                continue;
            }
            
            for (int i = 0; i < k_d1; i++) {
                
                int in_offset_d1 = D1_TO_OFFSET(cur_m_d1 + i, m_d2, n_channels);
                
                for (int j = 0; j < k_d2; j += 2) {
                    
                    int in_offset_d2 = D2_TO_OFFSET(cur_m_d2 + j, n_channels);
                    
                    for (int k = 0; k < n_channels; k += 1) {
                        if (p_in_matrix[in_offset_d1 + in_offset_d2 + k] > p_out_matrix[out_offset_d1 + out_offset_d2 + k]) {
                            p_out_matrix[out_offset_d1 + out_offset_d2 + k] = p_in_matrix[in_offset_d1 + in_offset_d2 + k];
                        }
                    }                        
                    
                }
            }
            
        }
        
    }
    
    DNN_TRACE_4("finish %s\n","");
}

void max_pool_1ch(
    float * __restrict__ p_in_matrix, 
    int m_d1,
    int m_d2,
    int m_d1_start,
    int m_d2_start,
    int m_d1_finish,
    int m_d2_finish,
    int k_d1,
    int k_d2,
    int n_channels,
    int n_stride,
    float * __restrict__ p_out_matrix)
{    
    int out_d2;
    
    DNN_TRACE_4("start %s\n","");
    DNN_TRACE_4(" p_in_matrix: %p\n", p_in_matrix);
    DNN_TRACE_4(" m_d1,m_d2: [%d,%d]\n", m_d1, m_d2);
    DNN_TRACE_4(" s/e_d1,s/e_d2: [%d,%d] <=> [%d,%d]\n", m_d1_start, m_d2_start, m_d1_finish, m_d2_finish);
    DNN_TRACE_4(" k_d1,k_d2: [%d,%d]\n", k_d1, k_d2);
    DNN_TRACE_4(" n_channels,n_stride: [%d,%d]\n", n_channels, n_stride);
    DNN_TRACE_4(" p_out_matrix: %p\n", p_out_matrix);
    
    out_d2 = OUT_DIM(m_d2, k_d2, n_stride);
    
    for (int cur_m_d1 = m_d1_start; (cur_m_d1 + k_d1) <= m_d1_finish; cur_m_d1 += n_stride) {
        
        int out_offset_d1 = D1_TO_OFFSET(cur_m_d1, out_d2, n_channels);
        
        for (int cur_m_d2 = m_d2_start; (cur_m_d2 + k_d2) <= m_d1_finish; cur_m_d2 += n_stride) {
            
            int out_offset_d2 = D2_TO_OFFSET(cur_m_d2, n_channels);
                        
            for (int k = 0; k < n_channels; k += 1) {
                p_out_matrix[out_offset_d1 + out_offset_d2 + k] = 0;
            }
            
            if ((cur_m_d1 > m_d1) || (cur_m_d2 > m_d2)) {
                continue;
            }
            
            for (int i = 0; i < k_d1; i++) {
                
                int in_offset_d1 = D1_TO_OFFSET(cur_m_d1 + i, m_d2, n_channels);
                
                for (int j = 0; j < k_d2; j += 2) {
                    
                    int in_offset_d2 = D2_TO_OFFSET(cur_m_d2 + j, n_channels);
                    
                    for (int k = 0; k < n_channels; k += 1) {
                        if (p_in_matrix[in_offset_d1 + in_offset_d2 + k] > p_out_matrix[out_offset_d1 + out_offset_d2 + k]) {
                            p_out_matrix[out_offset_d1 + out_offset_d2 + k] = p_in_matrix[in_offset_d1 + in_offset_d2 + k];
                        }
                    }                        
                    
                }
            }
            
        }
    }
    
    DNN_TRACE_4("finish %s\n","");
}

void max_pool(float * __restrict__ p_in_matrix, 
                  int m_d1,
                  int m_d2,
                  int m_d1_start,
                  int m_d2_start,
                  int m_d1_finish,
                  int m_d2_finish,
                  int k_d1,
                  int k_d2,
                  int n_channels,
                  int n_stride,
                  float * __restrict__ p_out_matrix)
{    
    if ((n_channels == N_CHANNELS_8) && (k_d1 == K_D1_2) && (k_d2 == K_D2_2)) {
        max_pool_AVX_8ch_exact_k2_k2(
            p_in_matrix, 
            m_d1,
            m_d2,
            m_d1_start,
            m_d2_start,
            m_d1_finish,
            m_d2_finish,
            k_d1,
            k_d2,
            n_channels,
            n_stride,
            p_out_matrix);
    
    } else if (n_channels % 8 == 0) {
        max_pool_AVX_8ch(p_in_matrix, 
                         m_d1,
                         m_d2,
                         m_d1_start,
                         m_d2_start,
                         m_d1_finish,
                         m_d2_finish,
                         k_d1,
                         k_d2,
                         n_channels,
                         n_stride,
                         p_out_matrix);
    } else {
        max_pool_1ch(p_in_matrix, 
                     m_d1,
                     m_d2,
                     m_d1_start,
                     m_d2_start,
                     m_d1_finish,
                     m_d2_finish,
                     k_d1,
                     k_d2,
                     n_channels,
                     n_stride,
                     p_out_matrix);
    }
    
}

