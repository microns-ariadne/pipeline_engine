/////////////////////////////////////////////////////////////////////////////////////////
//
//
//
/////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////
// INCLUDES
/////////////////////////////////////////////////////////////////////////////////////////
#include <immintrin.h>
#include <float.h>
#include "common.h"
#include "conv.h"

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
void conv3D_AVX_32ch_depth(
    const float * __restrict__ p_in_matrix, 
    int m_d1, 
    int m_d2,
    const float * __restrict__ p_kernel,
    const float * __restrict__ p_biases,
    int n_kernels,
    int n_maxout_kernels,
    int k_d1, 
    int k_d2,
    int n_channels,
    int n_stride,
    float * __restrict__ p_out_matrix) 
{    
    float * __restrict__ matrix = (float *)__builtin_assume_aligned(p_in_matrix, 64);
    float * __restrict__ kernel = (float *)__builtin_assume_aligned(p_kernel, 64);
    float * __restrict__ res_matrix = (float *)__builtin_assume_aligned(p_out_matrix, 64);
    
	float sum;
    float sum_arr[8];
	
    DNN_TRACE_4("start %s\n","");
    DNN_TRACE_4(" p_in_matrix: %p\n", p_in_matrix);
    DNN_TRACE_4(" m_d1,m_d2: [%d,%d]\n", m_d1, m_d2);
    DNN_TRACE_4(" p_kernel: %p\n", p_kernel);
    DNN_TRACE_4(" p_biases: %p\n", p_biases);
    DNN_TRACE_4(" n_kernels: %d\n", n_kernels);
    DNN_TRACE_4(" n_maxout_kernels: %d\n", n_maxout_kernels);
    DNN_TRACE_4(" k_d1,k_d2: [%d,%d]\n", k_d1, k_d2);
    DNN_TRACE_4(" n_channels,n_stride: [%d,%d]\n", n_channels, n_stride);
    DNN_TRACE_4(" p_out_matrix: %p\n", p_out_matrix);
    
    int res_idx = 0;
    
    for (int cur_m_d1 = 0; (cur_m_d1 + k_d1) <= m_d1; cur_m_d1 += n_stride) {

        for (int cur_m_d2 = 0; (cur_m_d2 + k_d2) <= m_d2; cur_m_d2 += n_stride) {
            
            for (int maxout_id = 0; maxout_id < n_kernels; maxout_id += n_maxout_kernels) {

                int is_first_result = 1;
                float cur_result = 0;
                
                for (int kernel_id = maxout_id; kernel_id < (maxout_id + n_maxout_kernels); kernel_id++) {
                    __m256 ymm0;
                    __m256 ymm1;
                    __m256 ymm2;
                    __m256 ymm3;
                    __m256 ymm4;
                    __m256 ymm5;
                    __m256 ymm6;
                    __m256 ymm7;

                    __m256 ymm15 = _mm256_xor_ps(ymm15, ymm15); // init to 0s
                    __m256 ymm14 = _mm256_xor_ps(ymm14, ymm14); // init to 0s
                    __m256 ymm13 = _mm256_xor_ps(ymm14, ymm14); // init to 0s
                    __m256 ymm12 = _mm256_xor_ps(ymm14, ymm14); // init to 0s

                    int kernel_shift = DIMS_TO_SIZE(k_d1, k_d2, n_channels) * kernel_id;
                
                    for (int i = 0; i < k_d1; i++) {

                        int m_offset_d1 = D1_TO_OFFSET(cur_m_d1 + i, m_d2, n_channels);

                        int k_offset_d1 = D1_TO_OFFSET(i, k_d2, n_channels);

                        for (int j = 0; j < k_d2; j++) {

                            int m_offset_d2 = D2_TO_OFFSET(cur_m_d2 + j, n_channels);

                            int k_offset_d2 = D2_TO_OFFSET(j, n_channels);

                            int m_shift = m_offset_d1 + m_offset_d2;
                            int k_shift = k_offset_d1 + k_offset_d2;

                            for (int k = 0 ; k < n_channels; k += 32) {       
                                ymm0 = _mm256_load_ps(&matrix[m_shift + k]);
                                ymm1 = _mm256_load_ps(&kernel[kernel_shift + k_shift + k]);

                                ymm15 = _mm256_fmadd_ps(ymm0, ymm1, ymm15);

                                ymm2 = _mm256_load_ps(&matrix[m_shift + k + 8]);
                                ymm3 = _mm256_load_ps(&kernel[kernel_shift + k_shift + k + 8]);

                                ymm14 = _mm256_fmadd_ps(ymm2, ymm3, ymm14);

                                ymm4 = _mm256_load_ps(&matrix[m_shift + k + 16]);
                                ymm5 = _mm256_load_ps(&kernel[kernel_shift + k_shift + k + 16]);

                                ymm13 = _mm256_fmadd_ps(ymm4, ymm5, ymm13);

                                ymm6 = _mm256_load_ps(&matrix[m_shift + k + 24]);
                                ymm7 = _mm256_load_ps(&kernel[kernel_shift + k_shift + k + 24]);

                                ymm12 = _mm256_fmadd_ps(ymm6, ymm7, ymm12);
                            }
                        }
                    }

                    ymm12 = _mm256_add_ps(ymm12, ymm13);
                    ymm14 = _mm256_add_ps(ymm14, ymm15);
                    ymm12 = _mm256_add_ps(ymm12, ymm14);

                    sum = p_biases[kernel_id];

                    _mm256_store_ps(sum_arr, ymm12);
                    for (int m = 0; m < 8; m++) {
                        sum += sum_arr[m];
                    }

                    if (is_first_result) {
                        cur_result = sum;
                        is_first_result = 0;
                    } else {
                        if (sum > cur_result) { // MAX
                            cur_result = sum;
                        }
                    }

                }

                res_matrix[res_idx] = cur_result;
                res_idx++;
           }
            
        }
        
    }
    
    DNN_TRACE_4("finish %s\n","");
}

void conv3D_AVX_32ch_depth_exact(
    const float * __restrict__ p_in_matrix, 
    int m_d1, 
    int m_d2,
    const float * __restrict__ p_kernel,
    const float * __restrict__ p_biases,
    int n_kernels,
    int n_maxout_kernels,
    int k_d1, 
    int k_d2,
    int n_channels,
    int n_stride,
    float * __restrict__ p_out_matrix) 
{    
    float * __restrict__ matrix = (float *)__builtin_assume_aligned(p_in_matrix, 64);
    float * __restrict__ kernel = (float *)__builtin_assume_aligned(p_kernel, 64);
    float * __restrict__ res_matrix = (float *)__builtin_assume_aligned(p_out_matrix, 64);
    
	float sum;
    float sum_arr[8];
	
    DNN_TRACE_4("start %s\n","");
    DNN_TRACE_4(" p_in_matrix: %p\n", p_in_matrix);
    DNN_TRACE_4(" m_d1,m_d2: [%d,%d]\n", m_d1, m_d2);
    DNN_TRACE_4(" p_kernel: %p\n", p_kernel);
    DNN_TRACE_4(" p_biases: %p\n", p_biases);
    DNN_TRACE_4(" n_kernels: %d\n", n_kernels);
    DNN_TRACE_4(" n_maxout_kernels: %d\n", n_maxout_kernels);
    DNN_TRACE_4(" k_d1,k_d2: [%d,%d]\n", k_d1, k_d2);
    DNN_TRACE_4(" n_channels,n_stride: [%d,%d]\n", n_channels, n_stride);
    DNN_TRACE_4(" p_out_matrix: %p\n", p_out_matrix);
    
    int res_idx = 0;
    
    for (int cur_m_d1 = 0; (cur_m_d1 + K_D1_4) <= m_d1; cur_m_d1 += n_stride) {

        for (int cur_m_d2 = 0; (cur_m_d2 + K_D2_4) <= m_d2; cur_m_d2 += n_stride) {
            
            for (int maxout_id = 0; maxout_id < N_KERNELS; maxout_id += N_MAXOUT_KERNELS) {

                int is_first_result = 1;
                float cur_result = 0;
                
                for (int kernel_id = maxout_id; kernel_id < (maxout_id + N_MAXOUT_KERNELS); kernel_id++) {
                    __m256 ymm0;
                    __m256 ymm1;
                    __m256 ymm2;
                    __m256 ymm3;
                    __m256 ymm4;
                    __m256 ymm5;
                    __m256 ymm6;
                    __m256 ymm7;
                    __m256 ymm10;
                    __m256 ymm11;
                    
                    __m256 ymm15 = _mm256_xor_ps(ymm15, ymm15); // init to 0s
                    __m256 ymm14 = _mm256_xor_ps(ymm14, ymm14); // init to 0s
                    __m256 ymm13 = _mm256_xor_ps(ymm14, ymm14); // init to 0s
                    __m256 ymm12 = _mm256_xor_ps(ymm14, ymm14); // init to 0s
                    
                    int kernel_shift = DIMS_TO_SIZE(K_D1_4, K_D2_4, N_CHANNELS_32) * kernel_id;
                
                    for (int i = 0; i < K_D1_4; i++) {

                        int m_offset_d1 = D1_TO_OFFSET(cur_m_d1 + i, m_d2, N_CHANNELS_32);

                        int k_offset_d1 = D1_TO_OFFSET(i, K_D2_4, n_channels);

                        for (int j = 0; j < K_D2_4; j++) {

                            int m_offset_d2 = D2_TO_OFFSET(cur_m_d2 + j, N_CHANNELS_32);

                            int k_offset_d2 = D2_TO_OFFSET(j, N_CHANNELS_32);

                            int m_shift = m_offset_d1 + m_offset_d2;
                            int k_shift = k_offset_d1 + k_offset_d2;
                            
                            int k = 0;
                            //for (int k = 0 ; k < N_CHANNELS_32; k += 32) {       
                                ymm0 = _mm256_load_ps(&matrix[m_shift + k]);
                                ymm1 = _mm256_load_ps(&kernel[kernel_shift + k_shift + k]);

                                ymm15 = _mm256_fmadd_ps(ymm0, ymm1, ymm15);

                                ymm2 = _mm256_load_ps(&matrix[m_shift + k + 8]);
                                ymm3 = _mm256_load_ps(&kernel[kernel_shift + k_shift + k + 8]);

                                ymm14 = _mm256_fmadd_ps(ymm2, ymm3, ymm14);

                                ymm4 = _mm256_load_ps(&matrix[m_shift + k + 16]);
                                ymm5 = _mm256_load_ps(&kernel[kernel_shift + k_shift + k + 16]);

                                ymm13 = _mm256_fmadd_ps(ymm4, ymm5, ymm13);

                                ymm6 = _mm256_load_ps(&matrix[m_shift + k + 24]);
                                ymm7 = _mm256_load_ps(&kernel[kernel_shift + k_shift + k + 24]);

                                ymm12 = _mm256_fmadd_ps(ymm6, ymm7, ymm12);
                            //}
                        }
                    }
                    
                    //
                    ymm12 = _mm256_add_ps(ymm12, ymm13);
                    ymm14 = _mm256_add_ps(ymm14, ymm15);
                    ymm11 = _mm256_add_ps(ymm12, ymm14);

                    ymm10 = _mm256_permute2f128_ps(ymm11 , ymm11 , 1);
                    ymm11 = _mm256_add_ps(ymm11, ymm10);
                    ymm11 = _mm256_hadd_ps(ymm11, ymm11);
                    ymm11 = _mm256_hadd_ps(ymm11, ymm11);

                    sum = p_biases[kernel_id];

                    _mm256_store_ps(sum_arr, ymm11);
                    //for (int m = 0; m < 1; m++) {
                        sum += sum_arr[0];
                    //}
                    
                    if (is_first_result) {
                        cur_result = sum;
                        is_first_result = 0;
                    } else {
                        if (sum > cur_result) { // MAX
                            cur_result = sum;
                        }
                    }

                }

                res_matrix[res_idx] = cur_result;
                res_idx++;
           }
            
        }
        
    }
    
    DNN_TRACE_4("finish %s\n","");
}

void conv3D_AVX_32ch(
    const float * __restrict__ p_in_matrix, 
    int m_d1, 
    int m_d2,
    const float * __restrict__ p_kernel,
    const float * __restrict__ p_biases,
    int n_kernels,
    int k_d1, 
    int k_d2,
    int n_channels,
    int n_stride,
    float * __restrict__ p_out_matrix) 
{    
    float * __restrict__ matrix = (float *)__builtin_assume_aligned(p_in_matrix, 64);
    float * __restrict__ kernel = (float *)__builtin_assume_aligned(p_kernel, 64);
    float * __restrict__ res_matrix = (float *)__builtin_assume_aligned(p_out_matrix, 64);
    
	float sum;
    float sum_arr[8];
	
    DNN_TRACE_4("start %s\n","");
    DNN_TRACE_4(" p_in_matrix: %p\n", p_in_matrix);
    DNN_TRACE_4(" m_d1,m_d2: [%d,%d]\n", m_d1, m_d2);
    DNN_TRACE_4(" p_kernel: %p\n", p_kernel);
    DNN_TRACE_4(" p_biases: %p\n", p_biases);
    DNN_TRACE_4(" n_kernels: %d\n", n_kernels);
    DNN_TRACE_4(" k_d1,k_d2: [%d,%d]\n", k_d1, k_d2);
    DNN_TRACE_4(" n_channels,n_stride: [%d,%d]\n", n_channels, n_stride);
    DNN_TRACE_4(" p_out_matrix: %p\n", p_out_matrix);
    
    int res_idx = 0;
    
    for (int cur_m_d1 = 0; (cur_m_d1 + k_d1) <= m_d1; cur_m_d1 += n_stride) {

        for (int cur_m_d2 = 0; (cur_m_d2 + k_d2) <= m_d2; cur_m_d2 += n_stride) {
            
            int is_first_result = 1;
            float cur_result = 0;
            
            for (int kernel_id = 0; kernel_id < n_kernels; kernel_id++) {
                __m256 ymm0;
                __m256 ymm1;
                __m256 ymm2;
                __m256 ymm3;
                __m256 ymm4;
                __m256 ymm5;
                __m256 ymm6;
                __m256 ymm7;

                __m256 ymm15 = _mm256_xor_ps(ymm15, ymm15); // init to 0s
                __m256 ymm14 = _mm256_xor_ps(ymm14, ymm14); // init to 0s
                __m256 ymm13 = _mm256_xor_ps(ymm14, ymm14); // init to 0s
                __m256 ymm12 = _mm256_xor_ps(ymm14, ymm14); // init to 0s

                int kernel_shift = DIMS_TO_SIZE(k_d1, k_d2, n_channels) * kernel_id;
                
                for (int i = 0; i < k_d1; i++) {

                    int m_offset_d1 = D1_TO_OFFSET(cur_m_d1 + i, m_d2, n_channels);

                    int k_offset_d1 = D1_TO_OFFSET(i, k_d2, n_channels);

                    for (int j = 0; j < k_d2; j++) {

                        int m_offset_d2 = D2_TO_OFFSET(cur_m_d2 + j, n_channels);

                        int k_offset_d2 = D2_TO_OFFSET(j, n_channels);

                        int m_shift = m_offset_d1 + m_offset_d2;
                        int k_shift = k_offset_d1 + k_offset_d2;

                        for (int k = 0 ; k < n_channels; k += 32) {       
                            ymm0 = _mm256_load_ps(&matrix[m_shift + k]);
                            ymm1 = _mm256_load_ps(&kernel[kernel_shift + k_shift + k]);
                            
                            ymm15 = _mm256_fmadd_ps(ymm0, ymm1, ymm15);

                            ymm2 = _mm256_load_ps(&matrix[m_shift + k + 8]);
                            ymm3 = _mm256_load_ps(&kernel[kernel_shift + k_shift + k + 8]);

                            ymm14 = _mm256_fmadd_ps(ymm2, ymm3, ymm14);

                            ymm4 = _mm256_load_ps(&matrix[m_shift + k + 16]);
                            ymm5 = _mm256_load_ps(&kernel[kernel_shift + k_shift + k + 16]);

                            ymm13 = _mm256_fmadd_ps(ymm4, ymm5, ymm13);

                            ymm6 = _mm256_load_ps(&matrix[m_shift + k + 24]);
                            ymm7 = _mm256_load_ps(&kernel[kernel_shift + k_shift + k + 24]);

                            ymm12 = _mm256_fmadd_ps(ymm6, ymm7, ymm12);
                        }
                    }
                }

                ymm12 = _mm256_add_ps(ymm12, ymm13);
                ymm14 = _mm256_add_ps(ymm14, ymm15);
                ymm12 = _mm256_add_ps(ymm12, ymm14);

                sum = p_biases[kernel_id];
                
                _mm256_store_ps(sum_arr, ymm12);
                for (int m = 0; m < 8; m++) {
                    sum += sum_arr[m];
                }
                
                if (is_first_result) {
                    cur_result = sum;
                    is_first_result = 0;
                } else {
                    if (sum > cur_result) { // MAX
                        cur_result = sum;
                    }
                }
                
            }
            
            res_matrix[res_idx] = cur_result;
            res_idx++;
            
        }
        
    }
    
    DNN_TRACE_4("finish %s\n","");
}

void conv3D_AVX_32ch_exact(
    const float * __restrict__ p_in_matrix, 
    int m_d1, 
    int m_d2,
    const float * __restrict__ p_kernel,
    const float * __restrict__ p_biases,
    int n_kernels,
    int k_d1, 
    int k_d2,
    int n_channels,
    int n_stride,
    float * __restrict__ p_out_matrix) 
{    
    float * __restrict__ matrix = (float * __restrict__)__builtin_assume_aligned(p_in_matrix, 32);
    float * __restrict__ kernel = (float * __restrict__)__builtin_assume_aligned(p_kernel, 32);
    float * __restrict__ res_matrix = (float * __restrict__)__builtin_assume_aligned(p_out_matrix, 32);
    
	float sum;
    float sum_arr[8];
	
    DNN_TRACE_4("start %s\n","");
    DNN_TRACE_4(" p_in_matrix: %p\n", p_in_matrix);
    DNN_TRACE_4(" m_d1,m_d2: [%d,%d]\n", m_d1, m_d2);
    DNN_TRACE_4(" p_kernel: %p\n", p_kernel);
    DNN_TRACE_4(" p_biases: %p\n", p_biases);
    DNN_TRACE_4(" n_kernels: %d\n", n_kernels);
    DNN_TRACE_4(" k_d1,k_d2: [%d,%d]\n", k_d1, k_d2);
    DNN_TRACE_4(" n_channels,n_stride: [%d,%d]\n", n_channels, n_stride);
    DNN_TRACE_4(" p_out_matrix: %p\n", p_out_matrix);
    
    DNN_ASSERT(n_channels == N_CHANNELS_32);
    DNN_ASSERT(k_d1 == K_D1_4);
    DNN_ASSERT(k_d2 == K_D2_4);
    
    int res_idx = 0;
    
    for (int cur_m_d1 = 0; (cur_m_d1 + K_D1_4) <= m_d1; cur_m_d1 += n_stride) {

        for (int cur_m_d2 = 0; (cur_m_d2 + K_D2_4) <= m_d2; cur_m_d2 += n_stride) {
            
            int is_first_result = 1;
            float cur_result = 0;
            
            for (int kernel_id = 0; kernel_id < n_kernels; kernel_id++) {
                __m256 ymm0;
                __m256 ymm1;
                __m256 ymm2;
                __m256 ymm3;
                __m256 ymm4;
                __m256 ymm5;
                __m256 ymm6;
                __m256 ymm7;
                
                __m256 ymm10;
                __m256 ymm11;
                __m256 ymm15 = _mm256_xor_ps(ymm15, ymm15); // init to 0s
                __m256 ymm14 = _mm256_xor_ps(ymm14, ymm14); // init to 0s
                __m256 ymm13 = _mm256_xor_ps(ymm14, ymm14); // init to 0s
                __m256 ymm12 = _mm256_xor_ps(ymm14, ymm14); // init to 0s

                int kernel_shift = DIMS_TO_SIZE(K_D1_4, K_D2_4, N_CHANNELS_32) * kernel_id;
                
                int m_offset_d1 = D1_TO_OFFSET(cur_m_d1, m_d2, N_CHANNELS_32);
                
                int k_offset_d1 = 0;
                
                for (int i = 0; i < K_D1_4; i++) {

                    int m_offset_d2 = D2_TO_OFFSET(cur_m_d2, N_CHANNELS_32);

                    int k_offset_d2 = 0;

                    for (int j = 0; j < K_D2_4; j++) {
                        
                        int m_shift = m_offset_d1 + m_offset_d2;
                        int k_shift = k_offset_d1 + k_offset_d2;
                        
                        ymm0 = _mm256_load_ps(&matrix[m_shift]);
                        ymm1 = _mm256_load_ps(&kernel[kernel_shift + k_shift]);
                            
                        ymm15 = _mm256_fmadd_ps(ymm0, ymm1, ymm15);
                        
                        ymm2 = _mm256_load_ps(&matrix[m_shift + 8]);
                        ymm3 = _mm256_load_ps(&kernel[kernel_shift + k_shift + 8]);

                        ymm14 = _mm256_fmadd_ps(ymm2, ymm3, ymm14);
                        
                        ymm4 = _mm256_load_ps(&matrix[m_shift + 16]);
                        ymm5 = _mm256_load_ps(&kernel[kernel_shift + k_shift + 16]);

                        ymm13 = _mm256_fmadd_ps(ymm4, ymm5, ymm13);
                        
                        ymm6 = _mm256_load_ps(&matrix[m_shift + 24]);
                        ymm7 = _mm256_load_ps(&kernel[kernel_shift + k_shift + 24]);

                        ymm12 = _mm256_fmadd_ps(ymm6, ymm7, ymm12);
                        
                        k_offset_d2 += N_CHANNELS_32;
                        
                        m_offset_d2 += N_CHANNELS_32;
                    }
                    
                    m_offset_d1 += m_d2 * N_CHANNELS_32;
                    
                    k_offset_d1 += K_D2_4 * N_CHANNELS_32;
                }
                
                ymm12 = _mm256_add_ps(ymm12, ymm13);
                ymm14 = _mm256_add_ps(ymm14, ymm15);
                ymm11 = _mm256_add_ps(ymm12, ymm14);
                
                ymm10 = _mm256_permute2f128_ps(ymm11 , ymm11 , 1);
                ymm11 = _mm256_add_ps(ymm11, ymm10);
                ymm11 = _mm256_hadd_ps(ymm11, ymm11);
                ymm11 = _mm256_hadd_ps(ymm11, ymm11);
                
                sum = p_biases[kernel_id];
                
                _mm256_store_ps(sum_arr, ymm11);
                //for (int m = 0; m < 1; m++) {
                    sum += sum_arr[0];
                //}
                
                if (is_first_result) {
                    cur_result = sum;
                    is_first_result = 0;
                } else {
                    if (sum > cur_result) { // MAX
                        cur_result = sum;
                    }
                }
                
            }
                        
            res_matrix[res_idx] = cur_result;
            res_idx++;
            
        }
        
    }
    
    DNN_TRACE_4("finish %s\n","");
}

void conv3D_AVX_32ch_exact_2k(
    const float * __restrict__ p_in_matrix, 
    int m_d1, 
    int m_d2,
    const float * __restrict__ p_kernel,
    const float * __restrict__ p_biases,
    int n_kernels,
    int k_d1, 
    int k_d2,
    int n_channels,
    int n_stride,
    float * __restrict__ p_out_matrix) 
{    
    float * __restrict__ matrix = (float * __restrict__)__builtin_assume_aligned(p_in_matrix, 32);
    float * __restrict__ kernel = (float * __restrict__)__builtin_assume_aligned(p_kernel, 32);
    float * __restrict__ res_matrix = (float * __restrict__)__builtin_assume_aligned(p_out_matrix, 32);
    
	//float sum;
    float sum_arr_1[8];
    float sum_arr_2[8];
	
    DNN_TRACE_4("start %s\n","");
    DNN_TRACE_4(" p_in_matrix: %p\n", p_in_matrix);
    DNN_TRACE_4(" m_d1,m_d2: [%d,%d]\n", m_d1, m_d2);
    DNN_TRACE_4(" p_kernel: %p\n", p_kernel);
    DNN_TRACE_4(" p_biases: %p\n", p_biases);
    DNN_TRACE_4(" n_kernels: %d\n", n_kernels);
    DNN_TRACE_4(" k_d1,k_d2: [%d,%d]\n", k_d1, k_d2);
    DNN_TRACE_4(" n_channels,n_stride: [%d,%d]\n", n_channels, n_stride);
    DNN_TRACE_4(" p_out_matrix: %p\n", p_out_matrix);
    
    DNN_ASSERT(n_channels == N_CHANNELS_32);
    DNN_ASSERT(k_d1 == K_D1_4);
    DNN_ASSERT(k_d2 == K_D2_4);
    DNN_ASSERT(n_kernels == 2);
    
    int res_idx = 0;
    
    for (int cur_m_d1 = 0; (cur_m_d1 + K_D1_4) <= m_d1; cur_m_d1 += n_stride) {

        for (int cur_m_d2 = 0; (cur_m_d2 + K_D2_4) <= m_d2; cur_m_d2 += n_stride) {
            
            //int is_first_result = 1;
            float cur_result = 0;
            
            //for (int kernel_id = 0; kernel_id < n_kernels; kernel_id++) {
                __m256 ymm0;
                __m256 ymm1;
                __m256 ymm2;
                __m256 ymm3;
                __m256 ymm4;
                __m256 ymm5;
                __m256 ymm6;
                __m256 ymm7;
                __m256 ymm8;
                __m256 ymm9;
                __m256 ymm10 = _mm256_xor_ps(ymm10, ymm10); // init to 0s
                __m256 ymm11 = _mm256_xor_ps(ymm11, ymm11); // init to 0s
                __m256 ymm15 = _mm256_xor_ps(ymm15, ymm15); // init to 0s
                __m256 ymm14 = _mm256_xor_ps(ymm14, ymm14); // init to 0s
                __m256 ymm13 = _mm256_xor_ps(ymm13, ymm13); // init to 0s
                __m256 ymm12 = _mm256_xor_ps(ymm12, ymm12); // init to 0s

                int kernel_shift = DIMS_TO_SIZE(K_D1_4, K_D2_4, N_CHANNELS_32);
                
                int m_offset_d1 = D1_TO_OFFSET(cur_m_d1, m_d2, N_CHANNELS_32);
                
                int k_offset_d1 = 0;
                
                for (int i = 0; i < K_D1_4; i++) {

                    int m_offset_d2 = D2_TO_OFFSET(cur_m_d2, N_CHANNELS_32);

                    int k_offset_d2 = 0;

                    for (int j = 0; j < K_D2_4; j++) {
                        
                        int m_shift = m_offset_d1 + m_offset_d2;
                        int k_shift = k_offset_d1 + k_offset_d2;
                        
                        ymm0 = _mm256_load_ps(&matrix[m_shift + 0]);
                        ymm1 = _mm256_load_ps(&kernel[k_shift + 0]);
                        ymm2 = _mm256_load_ps(&kernel[kernel_shift + k_shift + 0]);
                        
                        ymm15 = _mm256_fmadd_ps(ymm0, ymm1, ymm15);
                        ymm14 = _mm256_fmadd_ps(ymm0, ymm2, ymm14);
                        
                        ymm3 = _mm256_load_ps(&matrix[m_shift + 8]);
                        ymm4 = _mm256_load_ps(&kernel[k_shift + 8]);
                        ymm5 = _mm256_load_ps(&kernel[kernel_shift + k_shift + 8]);
                        
                        ymm13 = _mm256_fmadd_ps(ymm3, ymm4, ymm13);
                        ymm12 = _mm256_fmadd_ps(ymm3, ymm5, ymm12);
                        
                        ymm6 = _mm256_load_ps(&matrix[m_shift + 16]);
                        ymm7 = _mm256_load_ps(&kernel[k_shift + 16]);
                        ymm8 = _mm256_load_ps(&kernel[kernel_shift + k_shift + 16]);
                        
                        ymm11 = _mm256_fmadd_ps(ymm6, ymm7, ymm11);
                        ymm10 = _mm256_fmadd_ps(ymm6, ymm8, ymm10);
                        
                        ymm3 = _mm256_load_ps(&matrix[m_shift + 24]);
                        ymm4 = _mm256_load_ps(&kernel[k_shift + 24]);
                        ymm5 = _mm256_load_ps(&kernel[kernel_shift + k_shift + 24]);
                        
                        ymm13 = _mm256_fmadd_ps(ymm3, ymm4, ymm13);
                        ymm12 = _mm256_fmadd_ps(ymm3, ymm5, ymm12);
                        
                        k_offset_d2 += N_CHANNELS_32;
                        
                        m_offset_d2 += N_CHANNELS_32;
                    }
                    
                    m_offset_d1 += m_d2 * N_CHANNELS_32;
                    
                    k_offset_d1 += K_D2_4 * N_CHANNELS_32;
                }
                
                ymm12 = _mm256_add_ps(ymm10, ymm12);
                ymm14 = _mm256_add_ps(ymm12, ymm14);
                
                ymm13 = _mm256_add_ps(ymm11, ymm13);
                ymm15 = _mm256_add_ps(ymm13, ymm15);
                
                float sum_1 = p_biases[0];
                float sum_2 = p_biases[1];
                    
                ymm9 = _mm256_permute2f128_ps(ymm15 , ymm15 , 1);
                ymm15 = _mm256_add_ps(ymm15, ymm9);
                ymm15 = _mm256_hadd_ps(ymm15, ymm15);
                ymm15 = _mm256_hadd_ps(ymm15, ymm15);
                
                ymm8 = _mm256_permute2f128_ps(ymm14 , ymm14 , 1);
                ymm14 = _mm256_add_ps(ymm14, ymm8);
                ymm14 = _mm256_hadd_ps(ymm14, ymm14);
                ymm14 = _mm256_hadd_ps(ymm14, ymm14);
                
                _mm256_store_ps(sum_arr_1, ymm15);
                sum_1 += sum_arr_1[0];
                
                _mm256_store_ps(sum_arr_2, ymm14);
                sum_2 += sum_arr_2[0];
                
                if (sum_1 > sum_2) {
                    cur_result = sum_1;
                } else {
                    cur_result = sum_2;
                }
                
                
            //}
                        
            res_matrix[res_idx] = cur_result;
            res_idx++;
            
        }
        
    }
    
    DNN_TRACE_4("finish %s\n","");
}

void conv3D_AVX_32ch_exact_2k_2X(
    const float * __restrict__ p_in_matrix, 
    int m_d1, 
    int m_d2,
    int m_d1_start,
    int m_d2_start,
    int m_d1_finish,
    int m_d2_finish,
    const float * __restrict__ p_kernel,
    const float * __restrict__ p_biases,
    int n_kernels,
    int n_maxout_kernels,
    int k_d1, 
    int k_d2,
    int n_channels,
    int n_stride,
    float * __restrict__ p_out_matrix)
{ 
    float * __restrict__ matrix = (float * __restrict__)__builtin_assume_aligned(p_in_matrix, 32);
    float * __restrict__ kernel = (float * __restrict__)__builtin_assume_aligned(p_kernel, 32);
    float * __restrict__ res_matrix = (float * __restrict__)__builtin_assume_aligned(p_out_matrix, 32);
    
	//float sum;
    float sum_arr_1_1[8];
    float sum_arr_1_2[8];
    float sum_arr_1_3[8];
    float sum_arr_1_4[8];
    
    float sum_arr_2_1[8];
    float sum_arr_2_2[8];
    float sum_arr_2_3[8];
    float sum_arr_2_4[8];
	
    DNN_TRACE_4("start %s\n","");
    DNN_TRACE_4(" p_in_matrix: %p\n", p_in_matrix);
    DNN_TRACE_4(" m_d1,m_d2: [%d,%d]\n", m_d1, m_d2);
    DNN_TRACE_4(" s_d1,s_d2: [%d,%d]\n", m_d1_start, m_d2_start);
    DNN_TRACE_4(" e_d1,e_d2: [%d,%d]\n", m_d1_finish, m_d2_finish);    
    DNN_TRACE_4(" p_kernel: %p\n", p_kernel);
    DNN_TRACE_4(" p_biases: %p\n", p_biases);
    DNN_TRACE_4(" n_kernels: %d\n", n_kernels);
    DNN_TRACE_4(" k_d1,k_d2: [%d,%d]\n", k_d1, k_d2);
    DNN_TRACE_4(" n_channels,n_stride: [%d,%d]\n", n_channels, n_stride);
    DNN_TRACE_4(" p_out_matrix: %p\n", p_out_matrix);
    
    DNN_ASSERT(n_channels == N_CHANNELS_32);
    DNN_ASSERT(k_d1 == K_D1_4);
    DNN_ASSERT(k_d2 == K_D2_4);
    DNN_ASSERT(n_kernels == 2);
    DNN_ASSERT(n_stride == 1);
    DNN_ASSERT(m_d2 >= 4);
    
    int out_d2 = m_d2 - K_D2_4 + 1;
    int out_d3 = 1;
    //int res_idx = 0;
    
    for (int cur_m_d1 = m_d1_start; (cur_m_d1 + K_D1_4) <= m_d1_finish; cur_m_d1++) {
        
        int out_offset_d1 = D1_TO_OFFSET(cur_m_d1, out_d2, out_d3);
        
        for (int cur_m_d2 = m_d2_start; (cur_m_d2 + K_D2_4) <= m_d2_finish; cur_m_d2 += 4) {
            
            int fix_offset = ((cur_m_d2 + 3) + K_D2_4) - m_d2;
            if (fix_offset > 0) {
                cur_m_d2 -= fix_offset;
                DNN_ASSERT(cur_m_d2 >= 0);
            }
            
            int out_offset_d2 = D2_TO_OFFSET(cur_m_d2, out_d3);
            
            //int is_first_result = 1;
            float cur_result_1 = 0;
            float cur_result_2 = 0;
            float cur_result_3 = 0;
            float cur_result_4 = 0;
            
            //for (int kernel_id = 0; kernel_id < n_kernels; kernel_id++) {
                __m256 ymm0;
                __m256 ymm1;
                __m256 ymm2;
                __m256 ymm3;
                __m256 ymm4;
                __m256 ymm5;
                __m256 ymm6;
                __m256 ymm7;
                __m256 ymm8 = _mm256_xor_ps(ymm8, ymm8); // init to 0s
                __m256 ymm9 = _mm256_xor_ps(ymm9, ymm9); // init to 0s
                __m256 ymm10 = _mm256_xor_ps(ymm10, ymm10); // init to 0s
                __m256 ymm11 = _mm256_xor_ps(ymm11, ymm11); // init to 0s
                __m256 ymm15 = _mm256_xor_ps(ymm15, ymm15); // init to 0s
                __m256 ymm14 = _mm256_xor_ps(ymm14, ymm14); // init to 0s
                __m256 ymm13 = _mm256_xor_ps(ymm13, ymm13); // init to 0s
                __m256 ymm12 = _mm256_xor_ps(ymm12, ymm12); // init to 0s

                int kernel_shift = DIMS_TO_SIZE(K_D1_4, K_D2_4, N_CHANNELS_32);
                
                int m_offset_d1 = D1_TO_OFFSET(cur_m_d1, m_d2, N_CHANNELS_32);
                
                int k_offset_d1 = 0;
                
                for (int i = 0; i < K_D1_4; i++) {

                    int m_offset_d2 = D2_TO_OFFSET(cur_m_d2, N_CHANNELS_32);

                    int k_offset_d2 = 0;

                    for (int j = 0; j < K_D2_4; j++) {
                        
                        int m_shift = m_offset_d1 + m_offset_d2;
                        int k_shift = k_offset_d1 + k_offset_d2;
                        
                        /*for (int k = 0; k < N_CHANNELS_32; k += 16) {
                            
                            // M1
                            ymm0 = _mm256_load_ps(&matrix[m_shift + k]);
                            ymm1 = _mm256_load_ps(&matrix[m_shift + k + 8]);
                            
                            // K1
                            ymm4 = _mm256_load_ps(&kernel[k_shift + k]);
                            ymm5 = _mm256_load_ps(&kernel[k_shift + k + 8]);
                            
                            // M1 * K1
                            ymm15 = _mm256_fmadd_ps(ymm0, ymm4, ymm15);
                            ymm14 = _mm256_fmadd_ps(ymm1, ymm5, ymm14);
                            
                            // K2
                            ymm6 = _mm256_load_ps(&kernel[kernel_shift + k_shift + k]);
                            ymm7 = _mm256_load_ps(&kernel[kernel_shift + k_shift + k + 8]);
                            
                            // M1 * K2
                            ymm13 = _mm256_fmadd_ps(ymm0, ymm6, ymm13);
                            ymm12 = _mm256_fmadd_ps(ymm1, ymm7, ymm12);
                            
                            // M2
                            ymm2 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_32 + k ]);
                            ymm3 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_32 + k + 8]);
                            
                            // M2 * K1
                            ymm11 = _mm256_fmadd_ps(ymm2, ymm4, ymm11);
                            ymm10 = _mm256_fmadd_ps(ymm3, ymm5, ymm10);
                            
                            // M2 * K2
                            ymm9 = _mm256_fmadd_ps(ymm2, ymm6, ymm9);
                            ymm8 = _mm256_fmadd_ps(ymm3, ymm7, ymm8);
                        }*/
                        
                        
                        for (int k = 0; k < N_CHANNELS_32; k += 8) {
                            
                            // M1
                            ymm2 = _mm256_load_ps(&matrix[m_shift + k]);
                            
                            // K1
                            ymm0 = _mm256_load_ps(&kernel[k_shift + k]);
                            
                            // M1 * K1
                            ymm15 = _mm256_fmadd_ps(ymm0, ymm2, ymm15);
                            
                            // K2
                            ymm1 = _mm256_load_ps(&kernel[kernel_shift + k_shift + k]);
                            
                            // M1 * K2
                            ymm14 = _mm256_fmadd_ps(ymm1, ymm2, ymm14);
                            
                            // M2
                            ymm3 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_32 + k]);
                            
                            // M2 * K1
                            ymm13 = _mm256_fmadd_ps(ymm0, ymm3, ymm13);
                            
                            // M2 * K2
                            ymm12 = _mm256_fmadd_ps(ymm1, ymm3, ymm12);
                            
                            // M3
                            ymm4 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_32 * 2 + k]);
                            
                            // M3 * K1
                            ymm11 = _mm256_fmadd_ps(ymm0, ymm4, ymm11);
                            
                            // M3 * K2
                            ymm10 = _mm256_fmadd_ps(ymm1, ymm4, ymm10);
                            
                            // M4
                            ymm5 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_32 * 3 + k]);
                            
                            // M4 * K1
                            ymm9 = _mm256_fmadd_ps(ymm0, ymm5, ymm9);
                            
                            // M4 * K2
                            ymm8 = _mm256_fmadd_ps(ymm1, ymm5, ymm8);
                            
                        }
                        
                        k_offset_d2 += N_CHANNELS_32;
                    
                        m_offset_d2 += N_CHANNELS_32;
                    }
                    
                    m_offset_d1 += m_d2 * N_CHANNELS_32;
                    
                    k_offset_d1 += K_D2_4 * N_CHANNELS_32;
                }
                
                //ymm15 = _mm256_add_ps(ymm15, ymm14);
                //ymm13 = _mm256_add_ps(ymm13, ymm12);
                //ymm11 = _mm256_add_ps(ymm11, ymm10);
                //ymm9 = _mm256_add_ps(ymm9, ymm8);
                
                float sum_1_1 = p_biases[0];
                float sum_1_2 = p_biases[0];
                float sum_1_3 = p_biases[0];
                float sum_1_4 = p_biases[0];
                
                float sum_2_1 = p_biases[1];
                float sum_2_2 = p_biases[1];
                float sum_2_3 = p_biases[1];
                float sum_2_4 = p_biases[1];
                                    
                ymm7 = _mm256_permute2f128_ps(ymm15 , ymm15 , 1);
                ymm15 = _mm256_add_ps(ymm15, ymm7);
                ymm15 = _mm256_hadd_ps(ymm15, ymm15);
                ymm15 = _mm256_hadd_ps(ymm15, ymm15);
                
                ymm6 = _mm256_permute2f128_ps(ymm14 , ymm14 , 1);
                ymm14 = _mm256_add_ps(ymm14, ymm6);
                ymm14 = _mm256_hadd_ps(ymm14, ymm14);
                ymm14 = _mm256_hadd_ps(ymm14, ymm14);
                
                ymm5 = _mm256_permute2f128_ps(ymm13 , ymm13 , 1);
                ymm13 = _mm256_add_ps(ymm13, ymm5);
                ymm13 = _mm256_hadd_ps(ymm13, ymm13);
                ymm13 = _mm256_hadd_ps(ymm13, ymm13);
                
                ymm4 = _mm256_permute2f128_ps(ymm12 , ymm12 , 1);
                ymm12 = _mm256_add_ps(ymm12, ymm4);
                ymm12 = _mm256_hadd_ps(ymm12, ymm12);
                ymm12 = _mm256_hadd_ps(ymm12, ymm12);
                
                ymm3 = _mm256_permute2f128_ps(ymm11 , ymm11 , 1);
                ymm11 = _mm256_add_ps(ymm11, ymm3);
                ymm11 = _mm256_hadd_ps(ymm11, ymm11);
                ymm11 = _mm256_hadd_ps(ymm11, ymm11);
                
                ymm2 = _mm256_permute2f128_ps(ymm10 , ymm10 , 1);
                ymm10 = _mm256_add_ps(ymm10, ymm2);
                ymm10 = _mm256_hadd_ps(ymm10, ymm10);
                ymm10 = _mm256_hadd_ps(ymm10, ymm10);
                
                ymm1 = _mm256_permute2f128_ps(ymm9 , ymm9 , 1);
                ymm9 = _mm256_add_ps(ymm9, ymm1);
                ymm9 = _mm256_hadd_ps(ymm9, ymm9);
                ymm9 = _mm256_hadd_ps(ymm9, ymm9);
                
                ymm0 = _mm256_permute2f128_ps(ymm8 , ymm8 , 1);
                ymm8 = _mm256_add_ps(ymm8, ymm0);
                ymm8 = _mm256_hadd_ps(ymm8, ymm8);
                ymm8 = _mm256_hadd_ps(ymm8, ymm8);
                
                
                _mm256_store_ps(sum_arr_1_1, ymm15);
                sum_1_1 += sum_arr_1_1[0];
                
                _mm256_store_ps(sum_arr_2_1, ymm14);
                sum_2_1 += sum_arr_2_1[0];
                
                if (sum_1_1 > sum_2_1) {
                    cur_result_1 = sum_1_1;
                } else {
                    cur_result_1 = sum_2_1;
                }
                
                _mm256_store_ps(sum_arr_1_2, ymm13);
                sum_1_2 += sum_arr_1_2[0];
                
                _mm256_store_ps(sum_arr_2_2, ymm12);
                sum_2_2 += sum_arr_2_2[0];
                
                if (sum_1_2 > sum_2_2) {
                    cur_result_2 = sum_1_2;
                } else {
                    cur_result_2 = sum_2_2;
                }
                
                _mm256_store_ps(sum_arr_1_3, ymm11);
                sum_1_3 += sum_arr_1_3[0];
                
                _mm256_store_ps(sum_arr_2_3, ymm10);
                sum_2_3 += sum_arr_2_3[0];
                
                if (sum_1_3 > sum_2_3) {
                    cur_result_3 = sum_1_3;
                } else {
                    cur_result_3 = sum_2_3;
                }
                
                _mm256_store_ps(sum_arr_1_4, ymm9);
                sum_1_4 += sum_arr_1_4[0];
                
                _mm256_store_ps(sum_arr_2_4, ymm8);
                sum_2_4 += sum_arr_2_4[0];
                
                if (sum_1_4 > sum_2_4) {
                    cur_result_4 = sum_1_4;
                } else {
                    cur_result_4 = sum_2_4;
                }
                
                
            //}
                        
            res_matrix[out_offset_d1 + out_offset_d2 + 0] = cur_result_1;
            res_matrix[out_offset_d1 + out_offset_d2 + 1] = cur_result_2;
            res_matrix[out_offset_d1 + out_offset_d2 + 2] = cur_result_3;
            res_matrix[out_offset_d1 + out_offset_d2 + 3] = cur_result_4;
            
        }
        
    }
    
    DNN_TRACE_4("finish %s\n","");
}

void conv3D_AVX_32ch_exact_2k_2X_NEW(
    const float * __restrict__ p_in_matrix, 
    int m_d1, 
    int m_d2,
    int m_d1_start,
    int m_d2_start,
    int m_d1_finish,
    int m_d2_finish,
    const float * __restrict__ p_kernel,
    const float * __restrict__ p_biases,
    int n_kernels,
    int n_maxout_kernels,
    int k_d1, 
    int k_d2,
    int n_channels,
    int n_stride,
    float * __restrict__ p_out_matrix)
{ 
    const float * __restrict__ matrix = (const float * __restrict__)__builtin_assume_aligned(p_in_matrix, 32);
    const float * __restrict__ kernel = (const float * __restrict__)__builtin_assume_aligned(p_kernel, 32);
    float * __restrict__ res_matrix = (float * __restrict__)__builtin_assume_aligned(p_out_matrix, 32);
    
	//float sum;
    float sum_arr_1_1[8];
    float sum_arr_1_2[8];
    float sum_arr_1_3[8];
    float sum_arr_1_4[8];
    float sum_arr_1_5[8];
    float sum_arr_1_6[8];
    
    float sum_arr_2_1[8];
    float sum_arr_2_2[8];
    float sum_arr_2_3[8];
    float sum_arr_2_4[8];
    float sum_arr_2_5[8];
    float sum_arr_2_6[8];
	
    DNN_TRACE_4("start %s\n","");
    DNN_TRACE_4(" p_in_matrix: %p\n", p_in_matrix);
    DNN_TRACE_4(" m_d1,m_d2: [%d,%d]\n", m_d1, m_d2);
    DNN_TRACE_4(" s_d1,s_d2: [%d,%d]\n", m_d1_start, m_d2_start);
    DNN_TRACE_4(" e_d1,e_d2: [%d,%d]\n", m_d1_finish, m_d2_finish);    
    DNN_TRACE_4(" p_kernel: %p\n", p_kernel);
    DNN_TRACE_4(" p_biases: %p\n", p_biases);
    DNN_TRACE_4(" n_kernels: %d\n", n_kernels);
    DNN_TRACE_4(" k_d1,k_d2: [%d,%d]\n", k_d1, k_d2);
    DNN_TRACE_4(" n_channels,n_stride: [%d,%d]\n", n_channels, n_stride);
    DNN_TRACE_4(" p_out_matrix: %p\n", p_out_matrix);
    
    DNN_ASSERT(n_channels == N_CHANNELS_32);
    DNN_ASSERT(k_d1 == K_D1_4);
    DNN_ASSERT(k_d2 == K_D2_4);
    DNN_ASSERT(n_kernels == 2);
    DNN_ASSERT(n_stride == 1);
    DNN_ASSERT(m_d2 >= 4);
    
    const int out_d2 = m_d2 - K_D2_4 + 1;
    const int out_d3 = 1;
    //int res_idx = 0;
    
    for (int cur_m_d1 = m_d1_start; (cur_m_d1 + K_D1_4) <= m_d1_finish; cur_m_d1 += 1) {
        
        int fix_offset_1 = ((cur_m_d1) + K_D1_4) - m_d1;
        if (unlikely(fix_offset_1 > 0)) {
            cur_m_d1 -= fix_offset_1;
            DNN_ASSERT(cur_m_d1 >= 0);
        }
        
        int out_offset_d1 = D1_TO_OFFSET(cur_m_d1, out_d2, out_d3);
        
        for (int cur_m_d2 = m_d2_start; (cur_m_d2 + K_D2_4) <= m_d2_finish; cur_m_d2 += 6) {
            
            int fix_offset_2 = ((cur_m_d2 + 5) + K_D2_4) - m_d2;
            if (unlikely(fix_offset_2 > 0)) {
                cur_m_d2 -= fix_offset_2;
                DNN_ASSERT(cur_m_d2 >= 0);
            }
            
            int out_offset_d2 = D2_TO_OFFSET(cur_m_d2, out_d3);
            
            //int is_first_result = 1;
            float cur_result_1 = 0;
            float cur_result_2 = 0;
            float cur_result_3 = 0;
            float cur_result_4 = 0;
            float cur_result_5 = 0;
            float cur_result_6 = 0;
            
            //for (int kernel_id = 0; kernel_id < n_kernels; kernel_id++) {
                __m256 ymm0;
                __m256 ymm1;
                __m256 ymm2;
                __m256 ymm3;
                __m256 ymm4 = _mm256_xor_ps(ymm4, ymm4); // init to 0s
                __m256 ymm5 = _mm256_xor_ps(ymm5, ymm5); // init to 0s
                __m256 ymm6 = _mm256_xor_ps(ymm6, ymm6); // init to 0s
                __m256 ymm7 = _mm256_xor_ps(ymm7, ymm7); // init to 0s
                __m256 ymm8 = _mm256_xor_ps(ymm8, ymm8); // init to 0s
                __m256 ymm9 = _mm256_xor_ps(ymm9, ymm9); // init to 0s
                __m256 ymm10 = _mm256_xor_ps(ymm10, ymm10); // init to 0s
                __m256 ymm11 = _mm256_xor_ps(ymm11, ymm11); // init to 0s
                __m256 ymm15 = _mm256_xor_ps(ymm15, ymm15); // init to 0s
                __m256 ymm14 = _mm256_xor_ps(ymm14, ymm14); // init to 0s
                __m256 ymm13 = _mm256_xor_ps(ymm13, ymm13); // init to 0s
                __m256 ymm12 = _mm256_xor_ps(ymm12, ymm12); // init to 0s

                int kernel_shift = DIMS_TO_SIZE(K_D1_4, K_D2_4, N_CHANNELS_32);
                
                int m_offset_d1 = D1_TO_OFFSET(cur_m_d1, m_d2, N_CHANNELS_32);
                
                int k_offset_d1 = 0;
                
                for (int i = 0; i < K_D1_4; i++) {

                    int m_offset_d2 = D2_TO_OFFSET(cur_m_d2, N_CHANNELS_32);

                    int k_offset_d2 = 0;

                    for (int j = 0; j < K_D2_4; j++) {
                        
                        int m_shift = m_offset_d1 + m_offset_d2;
                        int k_shift = k_offset_d1 + k_offset_d2;
                        
                        for (int k = 0; k < N_CHANNELS_32; k += 8) {
                            
                            // M1-1
                            ymm2 = _mm256_load_ps(&matrix[m_shift + k]);
                            
                            // K1
                            ymm0 = _mm256_load_ps(&kernel[k_shift + k]);
                            
                            // M1-1 * K1
                            ymm15 = _mm256_fmadd_ps(ymm0, ymm2, ymm15);
                                    
                            // K2
                            ymm1 = _mm256_load_ps(&kernel[kernel_shift + k_shift + k]);
                            
                            // M1-1 * K2
                            ymm14 = _mm256_fmadd_ps(ymm1, ymm2, ymm14);
                            
                            /////
                            // M1-3
                            ymm3 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_32 * 2 + k]);
                            
                            // M1-3 * K1
                            ymm7 = _mm256_fmadd_ps(ymm0, ymm3, ymm7);
                            
                            // M1-3 * K2
                            ymm6 = _mm256_fmadd_ps(ymm1, ymm3, ymm6);
                            
                            /////
                            // M1-2
                            ymm2 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_32 + k]);
                            
                            // M1-2 * K1
                            ymm13 = _mm256_fmadd_ps(ymm0, ymm2, ymm13);
                            
                            // M1-2 * K2
                            ymm12 = _mm256_fmadd_ps(ymm1, ymm2, ymm12);
                            
                            /////
                            // M1-4
                            ymm3 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_32 * 3 + k]);
                            
                            // M1-4 * K1
                            ymm5 = _mm256_fmadd_ps(ymm0, ymm3, ymm5);
                            
                            // M1-4 * K2
                            ymm4 = _mm256_fmadd_ps(ymm1, ymm3, ymm4);
                            
                            // M1-5
                            ymm2 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_32 * 4 + k]);
                            
                            // M1-5 * K1
                            ymm11 = _mm256_fmadd_ps(ymm0, ymm2, ymm11);
                            
                            // M1-5 * K2
                            ymm10 = _mm256_fmadd_ps(ymm1, ymm2, ymm10);
                            
                            // M1-6
                            ymm2 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_32 * 5 + k]);
                            
                            // M1-6 * K1
                            ymm9 = _mm256_fmadd_ps(ymm0, ymm2, ymm9);
                            
                            // M1-6 * K2
                            ymm8 = _mm256_fmadd_ps(ymm1, ymm2, ymm8);
                            
                        }
                        
                        k_offset_d2 += N_CHANNELS_32;
                    
                        m_offset_d2 += N_CHANNELS_32;
                    }
                    
                    m_offset_d1 += m_d2 * N_CHANNELS_32;
                    
                    k_offset_d1 += K_D2_4 * N_CHANNELS_32;
                }
                
                //ymm15 = _mm256_add_ps(ymm15, ymm14);
                //ymm13 = _mm256_add_ps(ymm13, ymm12);
                //ymm11 = _mm256_add_ps(ymm11, ymm10);
                //ymm9 = _mm256_add_ps(ymm9, ymm8);
                
                float sum_1_1 = p_biases[0];
                float sum_1_2 = p_biases[0];
                float sum_1_3 = p_biases[0];
                float sum_1_4 = p_biases[0];
                float sum_1_5 = p_biases[0];
                float sum_1_6 = p_biases[0];
                
                float sum_2_1 = p_biases[1];
                float sum_2_2 = p_biases[1];
                float sum_2_3 = p_biases[1];
                float sum_2_4 = p_biases[1];
                float sum_2_5 = p_biases[1];
                float sum_2_6 = p_biases[1];
                                    
                ymm0 = _mm256_permute2f128_ps(ymm15 , ymm15 , 1);
                ymm15 = _mm256_add_ps(ymm15, ymm0);
                ymm15 = _mm256_hadd_ps(ymm15, ymm15);
                ymm15 = _mm256_hadd_ps(ymm15, ymm15);
                
                ymm1 = _mm256_permute2f128_ps(ymm14 , ymm14 , 1);
                ymm14 = _mm256_add_ps(ymm14, ymm1);
                ymm14 = _mm256_hadd_ps(ymm14, ymm14);
                ymm14 = _mm256_hadd_ps(ymm14, ymm14);
                
                ymm2 = _mm256_permute2f128_ps(ymm13 , ymm13 , 1);
                ymm13 = _mm256_add_ps(ymm13, ymm2);
                ymm13 = _mm256_hadd_ps(ymm13, ymm13);
                ymm13 = _mm256_hadd_ps(ymm13, ymm13);
                
                ymm3 = _mm256_permute2f128_ps(ymm12 , ymm12 , 1);
                ymm12 = _mm256_add_ps(ymm12, ymm3);
                ymm12 = _mm256_hadd_ps(ymm12, ymm12);
                ymm12 = _mm256_hadd_ps(ymm12, ymm12);
                
                ymm0 = _mm256_permute2f128_ps(ymm11 , ymm11 , 1);
                ymm11 = _mm256_add_ps(ymm11, ymm0);
                ymm11 = _mm256_hadd_ps(ymm11, ymm11);
                ymm11 = _mm256_hadd_ps(ymm11, ymm11);
                
                ymm1 = _mm256_permute2f128_ps(ymm10 , ymm10 , 1);
                ymm10 = _mm256_add_ps(ymm10, ymm1);
                ymm10 = _mm256_hadd_ps(ymm10, ymm10);
                ymm10 = _mm256_hadd_ps(ymm10, ymm10);
                
                ymm2 = _mm256_permute2f128_ps(ymm9 , ymm9 , 1);
                ymm9 = _mm256_add_ps(ymm9, ymm2);
                ymm9 = _mm256_hadd_ps(ymm9, ymm9);
                ymm9 = _mm256_hadd_ps(ymm9, ymm9);
                
                ymm3 = _mm256_permute2f128_ps(ymm8 , ymm8 , 1);
                ymm8 = _mm256_add_ps(ymm8, ymm3);
                ymm8 = _mm256_hadd_ps(ymm8, ymm8);
                ymm8 = _mm256_hadd_ps(ymm8, ymm8);
                
                ymm0 = _mm256_permute2f128_ps(ymm7 , ymm7 , 1);
                ymm7 = _mm256_add_ps(ymm7, ymm0);
                ymm7 = _mm256_hadd_ps(ymm7, ymm7);
                ymm7 = _mm256_hadd_ps(ymm7, ymm7);
                
                ymm1 = _mm256_permute2f128_ps(ymm6 , ymm6 , 1);
                ymm6 = _mm256_add_ps(ymm6, ymm1);
                ymm6 = _mm256_hadd_ps(ymm6, ymm6);
                ymm6 = _mm256_hadd_ps(ymm6, ymm6);
                
                ymm2 = _mm256_permute2f128_ps(ymm5 , ymm5 , 1);
                ymm5 = _mm256_add_ps(ymm5, ymm2);
                ymm5 = _mm256_hadd_ps(ymm5, ymm5);
                ymm5 = _mm256_hadd_ps(ymm5, ymm5);
                
                ymm3 = _mm256_permute2f128_ps(ymm4 , ymm4 , 1);
                ymm4 = _mm256_add_ps(ymm4, ymm3);
                ymm4 = _mm256_hadd_ps(ymm4, ymm4);
                ymm4 = _mm256_hadd_ps(ymm4, ymm4);
                
                _mm256_store_ps(sum_arr_1_1, ymm15);
                sum_1_1 += sum_arr_1_1[0];
                
                _mm256_store_ps(sum_arr_2_1, ymm14);
                sum_2_1 += sum_arr_2_1[0];
                
                if (sum_1_1 > sum_2_1) {
                    cur_result_1 = sum_1_1;
                } else {
                    cur_result_1 = sum_2_1;
                }
                
                _mm256_store_ps(sum_arr_1_2, ymm13);
                sum_1_2 += sum_arr_1_2[0];
                
                _mm256_store_ps(sum_arr_2_2, ymm12);
                sum_2_2 += sum_arr_2_2[0];
                
                if (sum_1_2 > sum_2_2) {
                    cur_result_2 = sum_1_2;
                } else {
                    cur_result_2 = sum_2_2;
                }
                
                _mm256_store_ps(sum_arr_1_3, ymm11);
                sum_1_3 += sum_arr_1_3[0];
                
                _mm256_store_ps(sum_arr_2_3, ymm10);
                sum_2_3 += sum_arr_2_3[0];
                
                if (sum_1_3 > sum_2_3) {
                    cur_result_3 = sum_1_3;
                } else {
                    cur_result_3 = sum_2_3;
                }
                
                _mm256_store_ps(sum_arr_1_4, ymm9);
                sum_1_4 += sum_arr_1_4[0];
                
                _mm256_store_ps(sum_arr_2_4, ymm8);
                sum_2_4 += sum_arr_2_4[0];
                
                if (sum_1_4 > sum_2_4) {
                    cur_result_4 = sum_1_4;
                } else {
                    cur_result_4 = sum_2_4;
                }
                
                _mm256_store_ps(sum_arr_1_5, ymm7);
                sum_1_5 += sum_arr_1_5[0];
                
                _mm256_store_ps(sum_arr_2_5, ymm6);
                sum_2_5 += sum_arr_2_5[0];
                
                if (sum_1_5 > sum_2_5) {
                    cur_result_5 = sum_1_5;
                } else {
                    cur_result_5 = sum_2_5;
                }
                
                _mm256_store_ps(sum_arr_1_6, ymm5);
                sum_1_6 += sum_arr_1_6[0];
                
                _mm256_store_ps(sum_arr_2_6, ymm4);
                sum_2_6 += sum_arr_2_6[0];
                
                if (sum_1_6 > sum_2_6) {
                    cur_result_6 = sum_1_6;
                } else {
                    cur_result_6 = sum_2_6;
                }
                
                
            //}
                        
            res_matrix[out_offset_d1 + out_offset_d2 + 0] = cur_result_1;
            res_matrix[out_offset_d1 + out_offset_d2 + 1] = cur_result_2;
            res_matrix[out_offset_d1 + out_offset_d2 + 2] = cur_result_5;
            res_matrix[out_offset_d1 + out_offset_d2 + 3] = cur_result_6;
            res_matrix[out_offset_d1 + out_offset_d2 + 4] = cur_result_3;
            res_matrix[out_offset_d1 + out_offset_d2 + 5] = cur_result_4;
            
        }
        
    }
    
    DNN_TRACE_4("finish %s\n","");
}

void conv3D_AVX_32ch_exact_2k_2X_NEW_k2_k2(
    const float * __restrict__ p_in_matrix, 
    int m_d1, 
    int m_d2,
    int m_d1_start,
    int m_d2_start,
    int m_d1_finish,
    int m_d2_finish,
    const float * __restrict__ p_kernel,
    const float * __restrict__ p_biases,
    int n_kernels,
    int n_maxout_kernels,
    int k_d1, 
    int k_d2,
    int n_channels,
    int n_stride,
    float * __restrict__ p_out_matrix)
{ 
    float * __restrict__ matrix = (float * __restrict__)__builtin_assume_aligned(p_in_matrix, 32);
    float * __restrict__ kernel = (float * __restrict__)__builtin_assume_aligned(p_kernel, 32);
    float * __restrict__ res_matrix = (float * __restrict__)__builtin_assume_aligned(p_out_matrix, 32);
    
	//float sum;
    float sum_arr_1_1[8];
    float sum_arr_1_2[8];
    float sum_arr_1_3[8];
    float sum_arr_1_4[8];
    float sum_arr_1_5[8];
    float sum_arr_1_6[8];
    
    float sum_arr_2_1[8];
    float sum_arr_2_2[8];
    float sum_arr_2_3[8];
    float sum_arr_2_4[8];
    float sum_arr_2_5[8];
    float sum_arr_2_6[8];
	
    DNN_TRACE_4("start %s\n","");
    DNN_TRACE_4(" p_in_matrix: %p\n", p_in_matrix);
    DNN_TRACE_4(" m_d1,m_d2: [%d,%d]\n", m_d1, m_d2);
    DNN_TRACE_4(" s_d1,s_d2: [%d,%d]\n", m_d1_start, m_d2_start);
    DNN_TRACE_4(" e_d1,e_d2: [%d,%d]\n", m_d1_finish, m_d2_finish);    
    DNN_TRACE_4(" p_kernel: %p\n", p_kernel);
    DNN_TRACE_4(" p_biases: %p\n", p_biases);
    DNN_TRACE_4(" n_kernels: %d\n", n_kernels);
    DNN_TRACE_4(" k_d1,k_d2: [%d,%d]\n", k_d1, k_d2);
    DNN_TRACE_4(" n_channels,n_stride: [%d,%d]\n", n_channels, n_stride);
    DNN_TRACE_4(" p_out_matrix: %p\n", p_out_matrix);
    
    DNN_ASSERT(n_channels == N_CHANNELS_32);
    DNN_ASSERT(k_d1 == K_D1_2);
    DNN_ASSERT(k_d2 == K_D2_2);
    DNN_ASSERT(n_kernels == 2);
    DNN_ASSERT(n_stride == 1);
    DNN_ASSERT(m_d2 >= 4);
    
    int out_d2 = m_d2 - K_D2_2 + 1;
    int out_d3 = 1;
    //int res_idx = 0;
    
    for (int cur_m_d1 = m_d1_start; (cur_m_d1 + K_D1_2) <= m_d1_finish; cur_m_d1 += 1) {
        
        int fix_offset_1 = ((cur_m_d1) + K_D1_2) - m_d1;
        if (fix_offset_1 > 0) {
            cur_m_d1 -= fix_offset_1;
            DNN_ASSERT(cur_m_d1 >= 0);
        }
        
        int out_offset_d1 = D1_TO_OFFSET(cur_m_d1, out_d2, out_d3);
        
        for (int cur_m_d2 = m_d2_start; (cur_m_d2 + K_D2_2) <= m_d2_finish; cur_m_d2 += 6) {
            
            int fix_offset_2 = ((cur_m_d2 + 5) + K_D2_2) - m_d2;
            if (fix_offset_2 > 0) {
                cur_m_d2 -= fix_offset_2;
                DNN_ASSERT(cur_m_d2 >= 0);
            }
            
            int out_offset_d2 = D2_TO_OFFSET(cur_m_d2, out_d3);
            
            //int is_first_result = 1;
            float cur_result_1 = 0;
            float cur_result_2 = 0;
            float cur_result_3 = 0;
            float cur_result_4 = 0;
            float cur_result_5 = 0;
            float cur_result_6 = 0;
            
            //for (int kernel_id = 0; kernel_id < n_kernels; kernel_id++) {
                __m256 ymm0;
                __m256 ymm1;
                __m256 ymm2;
                __m256 ymm3;
                __m256 ymm4 = _mm256_xor_ps(ymm4, ymm4); // init to 0s
                __m256 ymm5 = _mm256_xor_ps(ymm5, ymm5); // init to 0s
                __m256 ymm6 = _mm256_xor_ps(ymm6, ymm6); // init to 0s
                __m256 ymm7 = _mm256_xor_ps(ymm7, ymm7); // init to 0s
                __m256 ymm8 = _mm256_xor_ps(ymm8, ymm8); // init to 0s
                __m256 ymm9 = _mm256_xor_ps(ymm9, ymm9); // init to 0s
                __m256 ymm10 = _mm256_xor_ps(ymm10, ymm10); // init to 0s
                __m256 ymm11 = _mm256_xor_ps(ymm11, ymm11); // init to 0s
                __m256 ymm15 = _mm256_xor_ps(ymm15, ymm15); // init to 0s
                __m256 ymm14 = _mm256_xor_ps(ymm14, ymm14); // init to 0s
                __m256 ymm13 = _mm256_xor_ps(ymm13, ymm13); // init to 0s
                __m256 ymm12 = _mm256_xor_ps(ymm12, ymm12); // init to 0s

                int kernel_shift = DIMS_TO_SIZE(K_D1_2, K_D2_2, N_CHANNELS_32);
                
                int m_offset_d1 = D1_TO_OFFSET(cur_m_d1, m_d2, N_CHANNELS_32);
                
                int k_offset_d1 = 0;
                
                for (int i = 0; i < K_D1_2; i++) {

                    int m_offset_d2 = D2_TO_OFFSET(cur_m_d2, N_CHANNELS_32);

                    int k_offset_d2 = 0;

                    for (int j = 0; j < K_D2_2; j++) {
                        
                        int m_shift = m_offset_d1 + m_offset_d2;
                        int k_shift = k_offset_d1 + k_offset_d2;
                        
                        for (int k = 0; k < N_CHANNELS_32; k += 8) {
                            
                            // M1-1
                            ymm2 = _mm256_load_ps(&matrix[m_shift + k]);
                            
                            // K1
                            ymm0 = _mm256_load_ps(&kernel[k_shift + k]);
                            
                            // M1-1 * K1
                            ymm15 = _mm256_fmadd_ps(ymm0, ymm2, ymm15);
                                    
                            // K2
                            ymm1 = _mm256_load_ps(&kernel[kernel_shift + k_shift + k]);
                            
                            // M1-1 * K2
                            ymm14 = _mm256_fmadd_ps(ymm1, ymm2, ymm14);
                            
                            /////
                            // M1-3
                            ymm3 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_32 * 2 + k]);
                            
                            // M1-3 * K1
                            ymm7 = _mm256_fmadd_ps(ymm0, ymm3, ymm7);
                            
                            // M1-3 * K2
                            ymm6 = _mm256_fmadd_ps(ymm1, ymm3, ymm6);
                            
                            /////
                            // M1-2
                            ymm2 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_32 + k]);
                            
                            // M1-2 * K1
                            ymm13 = _mm256_fmadd_ps(ymm0, ymm2, ymm13);
                            
                            // M1-2 * K2
                            ymm12 = _mm256_fmadd_ps(ymm1, ymm2, ymm12);
                            
                            /////
                            // M1-4
                            ymm3 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_32 * 3 + k]);
                            
                            // M1-4 * K1
                            ymm5 = _mm256_fmadd_ps(ymm0, ymm3, ymm5);
                            
                            // M1-4 * K2
                            ymm4 = _mm256_fmadd_ps(ymm1, ymm3, ymm4);
                            
                            // M1-5
                            ymm2 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_32 * 4 + k]);
                            
                            // M1-5 * K1
                            ymm11 = _mm256_fmadd_ps(ymm0, ymm2, ymm11);
                            
                            // M1-5 * K2
                            ymm10 = _mm256_fmadd_ps(ymm1, ymm2, ymm10);
                            
                            // M1-6
                            ymm2 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_32 * 5 + k]);
                            
                            // M1-6 * K1
                            ymm9 = _mm256_fmadd_ps(ymm0, ymm2, ymm9);
                            
                            // M1-6 * K2
                            ymm8 = _mm256_fmadd_ps(ymm1, ymm2, ymm8);
                            
                        }
                        
                        k_offset_d2 += N_CHANNELS_32;
                    
                        m_offset_d2 += N_CHANNELS_32;
                    }
                    
                    m_offset_d1 += m_d2 * N_CHANNELS_32;
                    
                    k_offset_d1 += K_D2_2 * N_CHANNELS_32;
                }
                
                //ymm15 = _mm256_add_ps(ymm15, ymm14);
                //ymm13 = _mm256_add_ps(ymm13, ymm12);
                //ymm11 = _mm256_add_ps(ymm11, ymm10);
                //ymm9 = _mm256_add_ps(ymm9, ymm8);
                
                float sum_1_1 = p_biases[0];
                float sum_1_2 = p_biases[0];
                float sum_1_3 = p_biases[0];
                float sum_1_4 = p_biases[0];
                float sum_1_5 = p_biases[0];
                float sum_1_6 = p_biases[0];
                
                float sum_2_1 = p_biases[1];
                float sum_2_2 = p_biases[1];
                float sum_2_3 = p_biases[1];
                float sum_2_4 = p_biases[1];
                float sum_2_5 = p_biases[1];
                float sum_2_6 = p_biases[1];
                                    
                ymm0 = _mm256_permute2f128_ps(ymm15 , ymm15 , 1);
                ymm15 = _mm256_add_ps(ymm15, ymm0);
                ymm15 = _mm256_hadd_ps(ymm15, ymm15);
                ymm15 = _mm256_hadd_ps(ymm15, ymm15);
                
                ymm1 = _mm256_permute2f128_ps(ymm14 , ymm14 , 1);
                ymm14 = _mm256_add_ps(ymm14, ymm1);
                ymm14 = _mm256_hadd_ps(ymm14, ymm14);
                ymm14 = _mm256_hadd_ps(ymm14, ymm14);
                
                ymm2 = _mm256_permute2f128_ps(ymm13 , ymm13 , 1);
                ymm13 = _mm256_add_ps(ymm13, ymm2);
                ymm13 = _mm256_hadd_ps(ymm13, ymm13);
                ymm13 = _mm256_hadd_ps(ymm13, ymm13);
                
                ymm3 = _mm256_permute2f128_ps(ymm12 , ymm12 , 1);
                ymm12 = _mm256_add_ps(ymm12, ymm3);
                ymm12 = _mm256_hadd_ps(ymm12, ymm12);
                ymm12 = _mm256_hadd_ps(ymm12, ymm12);
                
                ymm0 = _mm256_permute2f128_ps(ymm11 , ymm11 , 1);
                ymm11 = _mm256_add_ps(ymm11, ymm0);
                ymm11 = _mm256_hadd_ps(ymm11, ymm11);
                ymm11 = _mm256_hadd_ps(ymm11, ymm11);
                
                ymm1 = _mm256_permute2f128_ps(ymm10 , ymm10 , 1);
                ymm10 = _mm256_add_ps(ymm10, ymm1);
                ymm10 = _mm256_hadd_ps(ymm10, ymm10);
                ymm10 = _mm256_hadd_ps(ymm10, ymm10);
                
                ymm2 = _mm256_permute2f128_ps(ymm9 , ymm9 , 1);
                ymm9 = _mm256_add_ps(ymm9, ymm2);
                ymm9 = _mm256_hadd_ps(ymm9, ymm9);
                ymm9 = _mm256_hadd_ps(ymm9, ymm9);
                
                ymm3 = _mm256_permute2f128_ps(ymm8 , ymm8 , 1);
                ymm8 = _mm256_add_ps(ymm8, ymm3);
                ymm8 = _mm256_hadd_ps(ymm8, ymm8);
                ymm8 = _mm256_hadd_ps(ymm8, ymm8);
                
                ymm0 = _mm256_permute2f128_ps(ymm7 , ymm7 , 1);
                ymm7 = _mm256_add_ps(ymm7, ymm0);
                ymm7 = _mm256_hadd_ps(ymm7, ymm7);
                ymm7 = _mm256_hadd_ps(ymm7, ymm7);
                
                ymm1 = _mm256_permute2f128_ps(ymm6 , ymm6 , 1);
                ymm6 = _mm256_add_ps(ymm6, ymm1);
                ymm6 = _mm256_hadd_ps(ymm6, ymm6);
                ymm6 = _mm256_hadd_ps(ymm6, ymm6);
                
                ymm2 = _mm256_permute2f128_ps(ymm5 , ymm5 , 1);
                ymm5 = _mm256_add_ps(ymm5, ymm2);
                ymm5 = _mm256_hadd_ps(ymm5, ymm5);
                ymm5 = _mm256_hadd_ps(ymm5, ymm5);
                
                ymm3 = _mm256_permute2f128_ps(ymm4 , ymm4 , 1);
                ymm4 = _mm256_add_ps(ymm4, ymm3);
                ymm4 = _mm256_hadd_ps(ymm4, ymm4);
                ymm4 = _mm256_hadd_ps(ymm4, ymm4);
                
                _mm256_store_ps(sum_arr_1_1, ymm15);
                sum_1_1 += sum_arr_1_1[0];
                
                _mm256_store_ps(sum_arr_2_1, ymm14);
                sum_2_1 += sum_arr_2_1[0];
                
                if (sum_1_1 > sum_2_1) {
                    cur_result_1 = sum_1_1;
                } else {
                    cur_result_1 = sum_2_1;
                }
                
                _mm256_store_ps(sum_arr_1_2, ymm13);
                sum_1_2 += sum_arr_1_2[0];
                
                _mm256_store_ps(sum_arr_2_2, ymm12);
                sum_2_2 += sum_arr_2_2[0];
                
                if (sum_1_2 > sum_2_2) {
                    cur_result_2 = sum_1_2;
                } else {
                    cur_result_2 = sum_2_2;
                }
                
                _mm256_store_ps(sum_arr_1_3, ymm11);
                sum_1_3 += sum_arr_1_3[0];
                
                _mm256_store_ps(sum_arr_2_3, ymm10);
                sum_2_3 += sum_arr_2_3[0];
                
                if (sum_1_3 > sum_2_3) {
                    cur_result_3 = sum_1_3;
                } else {
                    cur_result_3 = sum_2_3;
                }
                
                _mm256_store_ps(sum_arr_1_4, ymm9);
                sum_1_4 += sum_arr_1_4[0];
                
                _mm256_store_ps(sum_arr_2_4, ymm8);
                sum_2_4 += sum_arr_2_4[0];
                
                if (sum_1_4 > sum_2_4) {
                    cur_result_4 = sum_1_4;
                } else {
                    cur_result_4 = sum_2_4;
                }
                
                _mm256_store_ps(sum_arr_1_5, ymm7);
                sum_1_5 += sum_arr_1_5[0];
                
                _mm256_store_ps(sum_arr_2_5, ymm6);
                sum_2_5 += sum_arr_2_5[0];
                
                if (sum_1_5 > sum_2_5) {
                    cur_result_5 = sum_1_5;
                } else {
                    cur_result_5 = sum_2_5;
                }
                
                _mm256_store_ps(sum_arr_1_6, ymm5);
                sum_1_6 += sum_arr_1_6[0];
                
                _mm256_store_ps(sum_arr_2_6, ymm4);
                sum_2_6 += sum_arr_2_6[0];
                
                if (sum_1_6 > sum_2_6) {
                    cur_result_6 = sum_1_6;
                } else {
                    cur_result_6 = sum_2_6;
                }
                
                
            //}
                        
            res_matrix[out_offset_d1 + out_offset_d2 + 0] = cur_result_1;
            res_matrix[out_offset_d1 + out_offset_d2 + 1] = cur_result_2;
            res_matrix[out_offset_d1 + out_offset_d2 + 2] = cur_result_5;
            res_matrix[out_offset_d1 + out_offset_d2 + 3] = cur_result_6;
            res_matrix[out_offset_d1 + out_offset_d2 + 4] = cur_result_3;
            res_matrix[out_offset_d1 + out_offset_d2 + 5] = cur_result_4;
            
        }
        
    }
    
    DNN_TRACE_4("finish %s\n","");
}

void conv3D_AVX_32ch_exact_2k_2X_NEW_QTD(
    const float * __restrict__ p_in_matrix, 
    const uint32_t * __restrict__ p_sp_out_matrix,
    int m_d1, 
    int m_d2,
    int m_d1_start,
    int m_d2_start,
    int m_d1_finish,
    int m_d2_finish,
    const float * __restrict__ p_kernel,
    const float * __restrict__ p_biases,
    int n_kernels,
    int n_maxout_kernels,
    int k_d1, 
    int k_d2,
    int n_channels,
    int n_stride,
    float * __restrict__ p_out_matrix)
{ 
    float * __restrict__ matrix = (float * __restrict__)__builtin_assume_aligned(p_in_matrix, 32);
    float * __restrict__ kernel = (float * __restrict__)__builtin_assume_aligned(p_kernel, 32);
    float * __restrict__ res_matrix = (float * __restrict__)__builtin_assume_aligned(p_out_matrix, 32);
    
	//float sum;
    float sum_arr_1_1[8];
    float sum_arr_1_2[8];
    float sum_arr_1_3[8];
    float sum_arr_1_4[8];
    float sum_arr_1_5[8];
    float sum_arr_1_6[8];
    
    float sum_arr_2_1[8];
    float sum_arr_2_2[8];
    float sum_arr_2_3[8];
    float sum_arr_2_4[8];
    float sum_arr_2_5[8];
    float sum_arr_2_6[8];
	
    DNN_TRACE_4("start %s\n","");
    DNN_TRACE_4(" p_in_matrix: %p\n", p_in_matrix);
    DNN_TRACE_4(" m_d1,m_d2: [%d,%d]\n", m_d1, m_d2);
    DNN_TRACE_4(" s_d1,s_d2: [%d,%d]\n", m_d1_start, m_d2_start);
    DNN_TRACE_4(" e_d1,e_d2: [%d,%d]\n", m_d1_finish, m_d2_finish);    
    DNN_TRACE_4(" p_kernel: %p\n", p_kernel);
    DNN_TRACE_4(" p_biases: %p\n", p_biases);
    DNN_TRACE_4(" n_kernels: %d\n", n_kernels);
    DNN_TRACE_4(" k_d1,k_d2: [%d,%d]\n", k_d1, k_d2);
    DNN_TRACE_4(" n_channels,n_stride: [%d,%d]\n", n_channels, n_stride);
    DNN_TRACE_4(" p_out_matrix: %p\n", p_out_matrix);
    
    DNN_ASSERT(n_channels == N_CHANNELS_32);
    DNN_ASSERT(k_d1 == K_D1_4);
    DNN_ASSERT(k_d2 == K_D2_4);
    DNN_ASSERT(n_kernels == 2);
    DNN_ASSERT(n_stride == 1);
    DNN_ASSERT(m_d2 >= 4);
    
    int out_d2 = m_d2 - K_D2_4 + 1;
    int out_d3 = 1;
    //int res_idx = 0;
    
    int n_qtd = 0;
    
    for (int cur_m_d1 = m_d1_start; (cur_m_d1 + K_D1_4) <= m_d1_finish; cur_m_d1 += 1) {
        
        int fix_offset_1 = ((cur_m_d1) + K_D1_4) - m_d1;
        if (fix_offset_1 > 0) {
            cur_m_d1 -= fix_offset_1;
            DNN_ASSERT(cur_m_d1 >= 0);
        }
        
        int out_offset_d1 = D1_TO_OFFSET(cur_m_d1, out_d2, out_d3);
        
        int sp_offset_d1 = D1_TO_OFFSET(cur_m_d1, m_d1, 1);
        
        for (int cur_m_d2 = m_d2_start; (cur_m_d2 + K_D2_4) <= m_d2_finish; ) {
            
            int out_offset_d2 = D2_TO_OFFSET(cur_m_d2, out_d3);
            
            int sp_offset_d2 = D2_TO_OFFSET(cur_m_d2, 1);
            
            if ((p_sp_out_matrix[sp_offset_d1 + sp_offset_d2] == 3) ||
                (p_sp_out_matrix[sp_offset_d1 + sp_offset_d2] == 1)) {
                res_matrix[out_offset_d1 + out_offset_d2] = res_matrix[out_offset_d1 + out_offset_d2 - 1];
                n_qtd++;
                cur_m_d2 += 1;
                continue;
            }
            
            if (p_sp_out_matrix[sp_offset_d1 + sp_offset_d2] == 2) {
                res_matrix[out_offset_d1 + out_offset_d2] = res_matrix[out_offset_d1 + out_offset_d2 - m_d1];
                n_qtd++;
                cur_m_d2 += 1;
                continue;
            }
            
            int fix_offset_2 = ((cur_m_d2 + 5) + K_D2_4) - m_d2;
            if (fix_offset_2 > 0) {
                cur_m_d2 -= fix_offset_2;
                DNN_ASSERT(cur_m_d2 >= 0);
            }
            
            out_offset_d2 = D2_TO_OFFSET(cur_m_d2, out_d3);
            
            //int is_first_result = 1;
            float cur_result_1 = 0;
            float cur_result_2 = 0;
            float cur_result_3 = 0;
            float cur_result_4 = 0;
            float cur_result_5 = 0;
            float cur_result_6 = 0;
            
            //for (int kernel_id = 0; kernel_id < n_kernels; kernel_id++) {
                __m256 ymm0;
                __m256 ymm1;
                __m256 ymm2;
                __m256 ymm3;
                __m256 ymm4 = _mm256_xor_ps(ymm4, ymm4); // init to 0s
                __m256 ymm5 = _mm256_xor_ps(ymm5, ymm5); // init to 0s
                __m256 ymm6 = _mm256_xor_ps(ymm6, ymm6); // init to 0s
                __m256 ymm7 = _mm256_xor_ps(ymm7, ymm7); // init to 0s
                __m256 ymm8 = _mm256_xor_ps(ymm8, ymm8); // init to 0s
                __m256 ymm9 = _mm256_xor_ps(ymm9, ymm9); // init to 0s
                __m256 ymm10 = _mm256_xor_ps(ymm10, ymm10); // init to 0s
                __m256 ymm11 = _mm256_xor_ps(ymm11, ymm11); // init to 0s
                __m256 ymm15 = _mm256_xor_ps(ymm15, ymm15); // init to 0s
                __m256 ymm14 = _mm256_xor_ps(ymm14, ymm14); // init to 0s
                __m256 ymm13 = _mm256_xor_ps(ymm13, ymm13); // init to 0s
                __m256 ymm12 = _mm256_xor_ps(ymm12, ymm12); // init to 0s

                int kernel_shift = DIMS_TO_SIZE(K_D1_4, K_D2_4, N_CHANNELS_32);
                
                int m_offset_d1 = D1_TO_OFFSET(cur_m_d1, m_d2, N_CHANNELS_32);
                
                int k_offset_d1 = 0;
                
                for (int i = 0; i < K_D1_4; i++) {

                    int m_offset_d2 = D2_TO_OFFSET(cur_m_d2, N_CHANNELS_32);

                    int k_offset_d2 = 0;

                    for (int j = 0; j < K_D2_4; j++) {
                        
                        int m_shift = m_offset_d1 + m_offset_d2;
                        int k_shift = k_offset_d1 + k_offset_d2;
                        
                        for (int k = 0; k < N_CHANNELS_32; k += 8) {
                            
                            // M1-1
                            ymm2 = _mm256_load_ps(&matrix[m_shift + k]);
                            
                            // K1
                            ymm0 = _mm256_load_ps(&kernel[k_shift + k]);
                            
                            // M1-1 * K1
                            ymm15 = _mm256_fmadd_ps(ymm0, ymm2, ymm15);
                                    
                            // K2
                            ymm1 = _mm256_load_ps(&kernel[kernel_shift + k_shift + k]);
                            
                            // M1-1 * K2
                            ymm14 = _mm256_fmadd_ps(ymm1, ymm2, ymm14);
                            
                            /////
                            // M1-3
                            ymm3 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_32 * 2 + k]);
                            
                            // M1-3 * K1
                            ymm7 = _mm256_fmadd_ps(ymm0, ymm3, ymm7);
                            
                            // M1-3 * K2
                            ymm6 = _mm256_fmadd_ps(ymm1, ymm3, ymm6);
                            
                            /////
                            // M1-2
                            ymm2 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_32 + k]);
                            
                            // M1-2 * K1
                            ymm13 = _mm256_fmadd_ps(ymm0, ymm2, ymm13);
                            
                            // M1-2 * K2
                            ymm12 = _mm256_fmadd_ps(ymm1, ymm2, ymm12);
                            
                            /////
                            // M1-4
                            ymm3 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_32 * 3 + k]);
                            
                            // M1-4 * K1
                            ymm5 = _mm256_fmadd_ps(ymm0, ymm3, ymm5);
                            
                            // M1-4 * K2
                            ymm4 = _mm256_fmadd_ps(ymm1, ymm3, ymm4);
                            
                            // M1-5
                            ymm2 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_32 * 4 + k]);
                            
                            // M1-5 * K1
                            ymm11 = _mm256_fmadd_ps(ymm0, ymm2, ymm11);
                            
                            // M1-5 * K2
                            ymm10 = _mm256_fmadd_ps(ymm1, ymm2, ymm10);
                            
                            // M1-6
                            ymm2 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_32 * 5 + k]);
                            
                            // M1-6 * K1
                            ymm9 = _mm256_fmadd_ps(ymm0, ymm2, ymm9);
                            
                            // M1-6 * K2
                            ymm8 = _mm256_fmadd_ps(ymm1, ymm2, ymm8);
                            
                        }
                        
                        k_offset_d2 += N_CHANNELS_32;
                    
                        m_offset_d2 += N_CHANNELS_32;
                    }
                    
                    m_offset_d1 += m_d2 * N_CHANNELS_32;
                    
                    k_offset_d1 += K_D2_4 * N_CHANNELS_32;
                }
                
                //ymm15 = _mm256_add_ps(ymm15, ymm14);
                //ymm13 = _mm256_add_ps(ymm13, ymm12);
                //ymm11 = _mm256_add_ps(ymm11, ymm10);
                //ymm9 = _mm256_add_ps(ymm9, ymm8);
                
                float sum_1_1 = p_biases[0];
                float sum_1_2 = p_biases[0];
                float sum_1_3 = p_biases[0];
                float sum_1_4 = p_biases[0];
                float sum_1_5 = p_biases[0];
                float sum_1_6 = p_biases[0];
                
                float sum_2_1 = p_biases[1];
                float sum_2_2 = p_biases[1];
                float sum_2_3 = p_biases[1];
                float sum_2_4 = p_biases[1];
                float sum_2_5 = p_biases[1];
                float sum_2_6 = p_biases[1];
                                    
                ymm0 = _mm256_permute2f128_ps(ymm15 , ymm15 , 1);
                ymm15 = _mm256_add_ps(ymm15, ymm0);
                ymm15 = _mm256_hadd_ps(ymm15, ymm15);
                ymm15 = _mm256_hadd_ps(ymm15, ymm15);
                
                ymm1 = _mm256_permute2f128_ps(ymm14 , ymm14 , 1);
                ymm14 = _mm256_add_ps(ymm14, ymm1);
                ymm14 = _mm256_hadd_ps(ymm14, ymm14);
                ymm14 = _mm256_hadd_ps(ymm14, ymm14);
                
                ymm2 = _mm256_permute2f128_ps(ymm13 , ymm13 , 1);
                ymm13 = _mm256_add_ps(ymm13, ymm2);
                ymm13 = _mm256_hadd_ps(ymm13, ymm13);
                ymm13 = _mm256_hadd_ps(ymm13, ymm13);
                
                ymm3 = _mm256_permute2f128_ps(ymm12 , ymm12 , 1);
                ymm12 = _mm256_add_ps(ymm12, ymm3);
                ymm12 = _mm256_hadd_ps(ymm12, ymm12);
                ymm12 = _mm256_hadd_ps(ymm12, ymm12);
                
                ymm0 = _mm256_permute2f128_ps(ymm11 , ymm11 , 1);
                ymm11 = _mm256_add_ps(ymm11, ymm0);
                ymm11 = _mm256_hadd_ps(ymm11, ymm11);
                ymm11 = _mm256_hadd_ps(ymm11, ymm11);
                
                ymm1 = _mm256_permute2f128_ps(ymm10 , ymm10 , 1);
                ymm10 = _mm256_add_ps(ymm10, ymm1);
                ymm10 = _mm256_hadd_ps(ymm10, ymm10);
                ymm10 = _mm256_hadd_ps(ymm10, ymm10);
                
                ymm2 = _mm256_permute2f128_ps(ymm9 , ymm9 , 1);
                ymm9 = _mm256_add_ps(ymm9, ymm2);
                ymm9 = _mm256_hadd_ps(ymm9, ymm9);
                ymm9 = _mm256_hadd_ps(ymm9, ymm9);
                
                ymm3 = _mm256_permute2f128_ps(ymm8 , ymm8 , 1);
                ymm8 = _mm256_add_ps(ymm8, ymm3);
                ymm8 = _mm256_hadd_ps(ymm8, ymm8);
                ymm8 = _mm256_hadd_ps(ymm8, ymm8);
                
                ymm0 = _mm256_permute2f128_ps(ymm7 , ymm7 , 1);
                ymm7 = _mm256_add_ps(ymm7, ymm0);
                ymm7 = _mm256_hadd_ps(ymm7, ymm7);
                ymm7 = _mm256_hadd_ps(ymm7, ymm7);
                
                ymm1 = _mm256_permute2f128_ps(ymm6 , ymm6 , 1);
                ymm6 = _mm256_add_ps(ymm6, ymm1);
                ymm6 = _mm256_hadd_ps(ymm6, ymm6);
                ymm6 = _mm256_hadd_ps(ymm6, ymm6);
                
                ymm2 = _mm256_permute2f128_ps(ymm5 , ymm5 , 1);
                ymm5 = _mm256_add_ps(ymm5, ymm2);
                ymm5 = _mm256_hadd_ps(ymm5, ymm5);
                ymm5 = _mm256_hadd_ps(ymm5, ymm5);
                
                ymm3 = _mm256_permute2f128_ps(ymm4 , ymm4 , 1);
                ymm4 = _mm256_add_ps(ymm4, ymm3);
                ymm4 = _mm256_hadd_ps(ymm4, ymm4);
                ymm4 = _mm256_hadd_ps(ymm4, ymm4);
                
                _mm256_store_ps(sum_arr_1_1, ymm15);
                sum_1_1 += sum_arr_1_1[0];
                
                _mm256_store_ps(sum_arr_2_1, ymm14);
                sum_2_1 += sum_arr_2_1[0];
                
                if (sum_1_1 > sum_2_1) {
                    cur_result_1 = sum_1_1;
                } else {
                    cur_result_1 = sum_2_1;
                }
                
                _mm256_store_ps(sum_arr_1_2, ymm13);
                sum_1_2 += sum_arr_1_2[0];
                
                _mm256_store_ps(sum_arr_2_2, ymm12);
                sum_2_2 += sum_arr_2_2[0];
                
                if (sum_1_2 > sum_2_2) {
                    cur_result_2 = sum_1_2;
                } else {
                    cur_result_2 = sum_2_2;
                }
                
                _mm256_store_ps(sum_arr_1_3, ymm11);
                sum_1_3 += sum_arr_1_3[0];
                
                _mm256_store_ps(sum_arr_2_3, ymm10);
                sum_2_3 += sum_arr_2_3[0];
                
                if (sum_1_3 > sum_2_3) {
                    cur_result_3 = sum_1_3;
                } else {
                    cur_result_3 = sum_2_3;
                }
                
                _mm256_store_ps(sum_arr_1_4, ymm9);
                sum_1_4 += sum_arr_1_4[0];
                
                _mm256_store_ps(sum_arr_2_4, ymm8);
                sum_2_4 += sum_arr_2_4[0];
                
                if (sum_1_4 > sum_2_4) {
                    cur_result_4 = sum_1_4;
                } else {
                    cur_result_4 = sum_2_4;
                }
                
                _mm256_store_ps(sum_arr_1_5, ymm7);
                sum_1_5 += sum_arr_1_5[0];
                
                _mm256_store_ps(sum_arr_2_5, ymm6);
                sum_2_5 += sum_arr_2_5[0];
                
                if (sum_1_5 > sum_2_5) {
                    cur_result_5 = sum_1_5;
                } else {
                    cur_result_5 = sum_2_5;
                }
                
                _mm256_store_ps(sum_arr_1_6, ymm5);
                sum_1_6 += sum_arr_1_6[0];
                
                _mm256_store_ps(sum_arr_2_6, ymm4);
                sum_2_6 += sum_arr_2_6[0];
                
                if (sum_1_6 > sum_2_6) {
                    cur_result_6 = sum_1_6;
                } else {
                    cur_result_6 = sum_2_6;
                }
                
                
            //}
            
            if (fabsf(cur_result_1) < RES_THREASHOLD) {
                cur_result_1 = 0;
            }
            
            if (fabsf(cur_result_2) < RES_THREASHOLD) {
                cur_result_2 = 0;
            }
            
            if (fabsf(cur_result_3) < RES_THREASHOLD) {
                cur_result_3 = 0;
            }
            
            if (fabsf(cur_result_4) < RES_THREASHOLD) {
                cur_result_4 = 0;
            }
            
            if (fabsf(cur_result_5) < RES_THREASHOLD) {
                cur_result_5 = 0;
            }
            
            if (fabsf(cur_result_6) < RES_THREASHOLD) {
                cur_result_6 = 0;
            }
            
            res_matrix[out_offset_d1 + out_offset_d2 + 0] = cur_result_1;
            res_matrix[out_offset_d1 + out_offset_d2 + 1] = cur_result_2;
            res_matrix[out_offset_d1 + out_offset_d2 + 2] = cur_result_5;
            res_matrix[out_offset_d1 + out_offset_d2 + 3] = cur_result_6;
            res_matrix[out_offset_d1 + out_offset_d2 + 4] = cur_result_3;
            res_matrix[out_offset_d1 + out_offset_d2 + 5] = cur_result_4;
            
            cur_m_d2 += 6;
        }
        
    }
    
    DNN_TRACE_4("finish [n_qtd = %d]\n", n_qtd);
}

void conv3D_AVX_32ch_exact_2X_NEW(
    const float * __restrict__ p_in_matrix, 
    int m_d1, 
    int m_d2,
    int m_d1_start,
    int m_d2_start,
    int m_d1_finish,
    int m_d2_finish,
    const float * __restrict__ p_kernel,
    const float * __restrict__ p_biases,
    int n_kernels,
    int n_maxout_kernels,
    int k_d1, 
    int k_d2,
    int n_channels,
    int n_stride,
    float * __restrict__ p_out_matrix)
{ 
    float * __restrict__ matrix = (float * __restrict__)__builtin_assume_aligned(p_in_matrix, 32);
    float * __restrict__ kernel = (float * __restrict__)__builtin_assume_aligned(p_kernel, 32);
    float * __restrict__ res_matrix = (float * __restrict__)__builtin_assume_aligned(p_out_matrix, 32);
    
	//float sum;
    float sum_arr_1_1[8];
    float sum_arr_1_2[8];
    float sum_arr_1_3[8];
    float sum_arr_1_4[8];
    float sum_arr_1_5[8];
    float sum_arr_1_6[8];
        
    DNN_TRACE_4("start %s\n","");
    DNN_TRACE_4(" p_in_matrix: %p\n", p_in_matrix);
    DNN_TRACE_4(" m_d1,m_d2: [%d,%d]\n", m_d1, m_d2);
    DNN_TRACE_4(" s_d1,s_d2: [%d,%d]\n", m_d1_start, m_d2_start);
    DNN_TRACE_4(" e_d1,e_d2: [%d,%d]\n", m_d1_finish, m_d2_finish);    
    DNN_TRACE_4(" p_kernel: %p\n", p_kernel);
    DNN_TRACE_4(" p_biases: %p\n", p_biases);
    DNN_TRACE_4(" n_kernels: %d\n", n_kernels);
    DNN_TRACE_4(" k_d1,k_d2: [%d,%d]\n", k_d1, k_d2);
    DNN_TRACE_4(" n_channels,n_stride: [%d,%d]\n", n_channels, n_stride);
    DNN_TRACE_4(" p_out_matrix: %p\n", p_out_matrix);
    
    DNN_ASSERT(n_channels == N_CHANNELS_32);
    DNN_ASSERT(k_d1 == K_D1_4);
    DNN_ASSERT(k_d2 == K_D2_4);
    //DNN_ASSERT(n_kernels == 2);
    DNN_ASSERT(n_stride == 1);
    DNN_ASSERT(m_d2 >= 6);
    
    int out_d2 = m_d2 - K_D2_4 + 1;
    int out_d3 = 1;
    //int res_idx = 0;
    
    for (int cur_m_d1 = m_d1_start; (cur_m_d1 + K_D1_4) <= m_d1_finish; cur_m_d1 += 1) {
        
        int fix_offset_1 = ((cur_m_d1) + K_D1_4) - m_d1;
        if (fix_offset_1 > 0) {
            cur_m_d1 -= fix_offset_1;
            DNN_ASSERT(cur_m_d1 >= 0);
        }
        
        int out_offset_d1 = D1_TO_OFFSET(cur_m_d1, out_d2, out_d3);
        
        for (int cur_m_d2 = m_d2_start; (cur_m_d2 + K_D2_4) <= m_d2_finish; cur_m_d2 += 6) {
            
            int fix_offset_2 = ((cur_m_d2 + 5) + K_D2_4) - m_d2;
            if (fix_offset_2 > 0) {
                cur_m_d2 -= fix_offset_2;
                DNN_ASSERT(cur_m_d2 >= 0);
            }
            
            int out_offset_d2 = D2_TO_OFFSET(cur_m_d2, out_d3);
            
            int is_first_result = 1;
            float cur_result_1 = 0;
            float cur_result_2 = 0;
            float cur_result_3 = 0;
            float cur_result_4 = 0;
            float cur_result_5 = 0;
            float cur_result_6 = 0;
            
            for (int kernel_id = 0; kernel_id < n_kernels; kernel_id++) {
                __m256 ymm0;
                __m256 ymm1;
                __m256 ymm2;
                __m256 ymm3;
                __m256 ymm4;
                __m256 ymm5;
                __m256 ymm6;
                __m256 ymm10 = _mm256_xor_ps(ymm10, ymm10); // init to 0s
                __m256 ymm11 = _mm256_xor_ps(ymm11, ymm11); // init to 0s
                __m256 ymm12 = _mm256_xor_ps(ymm12, ymm12); // init to 0s
                __m256 ymm13 = _mm256_xor_ps(ymm13, ymm13); // init to 0s
                __m256 ymm14 = _mm256_xor_ps(ymm14, ymm14); // init to 0s
                __m256 ymm15 = _mm256_xor_ps(ymm15, ymm15); // init to 0s

                int kernel_shift = DIMS_TO_SIZE(K_D1_4, K_D2_4, N_CHANNELS_32) * kernel_id;
                
                int m_offset_d1 = D1_TO_OFFSET(cur_m_d1, m_d2, N_CHANNELS_32);
                
                int k_offset_d1 = 0;
                
                for (int i = 0; i < K_D1_4; i++) {

                    int m_offset_d2 = D2_TO_OFFSET(cur_m_d2, N_CHANNELS_32);

                    int k_offset_d2 = 0;

                    for (int j = 0; j < K_D2_4; j++) {
                        
                        int m_shift = m_offset_d1 + m_offset_d2;
                        int k_shift = k_offset_d1 + k_offset_d2;
                        
                        for (int k = 0; k < N_CHANNELS_32; k += 8) {
                            
                            // K1
                            ymm0 = _mm256_load_ps(&kernel[kernel_shift + k_shift + k]);
                            
                            // M1-1
                            ymm1 = _mm256_load_ps(&matrix[m_shift + k]);
                            
                            // M1-1 * K1
                            ymm15 = _mm256_fmadd_ps(ymm0, ymm1, ymm15);
                            
                            // M1-2
                            ymm2 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_32 + k]);
                            
                            // M1-2 * K1
                            ymm14 = _mm256_fmadd_ps(ymm0, ymm2, ymm14);
                            
                            // M1-3
                            ymm3 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_32 * 2 + k]);
                            
                            // M1-3 * K1
                            ymm13 = _mm256_fmadd_ps(ymm0, ymm3, ymm13);
                            
                            // M1-4
                            ymm4 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_32 * 3 + k]);
                            
                            // M1-4 * K1
                            ymm12 = _mm256_fmadd_ps(ymm0, ymm4, ymm12);
                            
                            // M1-5
                            ymm5 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_32 * 4 + k]);
                            
                            // M1-5 * K1
                            ymm11 = _mm256_fmadd_ps(ymm0, ymm5, ymm11);
                            
                            // M1-6
                            ymm6 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_32 * 5 + k]);
                            
                            // M1-6 * K1
                            ymm10 = _mm256_fmadd_ps(ymm0, ymm6, ymm10);
                                                        
                        }
                        
                        k_offset_d2 += N_CHANNELS_32;
                    
                        m_offset_d2 += N_CHANNELS_32;
                    }
                    
                    m_offset_d1 += m_d2 * N_CHANNELS_32;
                    
                    k_offset_d1 += K_D2_4 * N_CHANNELS_32;
                }
                                
                float sum_1_1 = p_biases[kernel_id];
                float sum_1_2 = p_biases[kernel_id];
                float sum_1_3 = p_biases[kernel_id];
                float sum_1_4 = p_biases[kernel_id];
                float sum_1_5 = p_biases[kernel_id];
                float sum_1_6 = p_biases[kernel_id];
                                    
                ymm0 = _mm256_permute2f128_ps(ymm15 , ymm15 , 1);
                ymm15 = _mm256_add_ps(ymm15, ymm0);
                ymm15 = _mm256_hadd_ps(ymm15, ymm15);
                ymm15 = _mm256_hadd_ps(ymm15, ymm15);
                
                ymm1 = _mm256_permute2f128_ps(ymm14 , ymm14 , 1);
                ymm14 = _mm256_add_ps(ymm14, ymm1);
                ymm14 = _mm256_hadd_ps(ymm14, ymm14);
                ymm14 = _mm256_hadd_ps(ymm14, ymm14);
                
                ymm2 = _mm256_permute2f128_ps(ymm13 , ymm13 , 1);
                ymm13 = _mm256_add_ps(ymm13, ymm2);
                ymm13 = _mm256_hadd_ps(ymm13, ymm13);
                ymm13 = _mm256_hadd_ps(ymm13, ymm13);
                
                ymm3 = _mm256_permute2f128_ps(ymm12 , ymm12 , 1);
                ymm12 = _mm256_add_ps(ymm12, ymm3);
                ymm12 = _mm256_hadd_ps(ymm12, ymm12);
                ymm12 = _mm256_hadd_ps(ymm12, ymm12);
                
                ymm0 = _mm256_permute2f128_ps(ymm11 , ymm11 , 1);
                ymm11 = _mm256_add_ps(ymm11, ymm0);
                ymm11 = _mm256_hadd_ps(ymm11, ymm11);
                ymm11 = _mm256_hadd_ps(ymm11, ymm11);
                
                ymm1 = _mm256_permute2f128_ps(ymm10 , ymm10 , 1);
                ymm10 = _mm256_add_ps(ymm10, ymm1);
                ymm10 = _mm256_hadd_ps(ymm10, ymm10);
                ymm10 = _mm256_hadd_ps(ymm10, ymm10);
                
                _mm256_store_ps(sum_arr_1_1, ymm15);
                sum_1_1 += sum_arr_1_1[0];
                
                if (is_first_result) {
                    cur_result_1 = sum_1_1; 
                } else if (sum_1_1 > cur_result_1) {
                    cur_result_1 = sum_1_1; 
                }
                
                _mm256_store_ps(sum_arr_1_2, ymm14);
                sum_1_2 += sum_arr_1_2[0];
                
                if (is_first_result) {
                    cur_result_2 = sum_1_2; 
                } else if (sum_1_2 > cur_result_2) {
                    cur_result_2 = sum_1_2; 
                }
                
                _mm256_store_ps(sum_arr_1_3, ymm13);
                sum_1_3 += sum_arr_1_3[0];
                
                if (is_first_result) {
                    cur_result_3 = sum_1_3; 
                } else if (sum_1_3 > cur_result_3) {
                    cur_result_3 = sum_1_3; 
                }
                
                _mm256_store_ps(sum_arr_1_4, ymm12);
                sum_1_4 += sum_arr_1_4[0];
                
                if (is_first_result) {
                    cur_result_4 = sum_1_4; 
                } else if (sum_1_4 > cur_result_4) {
                    cur_result_4 = sum_1_4; 
                }
                
                _mm256_store_ps(sum_arr_1_5, ymm11);
                sum_1_5 += sum_arr_1_5[0];
                
                if (is_first_result) {
                    cur_result_5 = sum_1_5; 
                } else if (sum_1_5 > cur_result_5) {
                    cur_result_5 = sum_1_5; 
                }
                
                _mm256_store_ps(sum_arr_1_6, ymm10);
                sum_1_6 += sum_arr_1_6[0];
                
                if (is_first_result) {
                    cur_result_6 = sum_1_6; 
                } else if (sum_1_6 > cur_result_6) {
                    cur_result_6 = sum_1_6; 
                }
                
                is_first_result = 0;
            }
                        
            res_matrix[out_offset_d1 + out_offset_d2 + 0] = cur_result_1;
            res_matrix[out_offset_d1 + out_offset_d2 + 1] = cur_result_2;
            res_matrix[out_offset_d1 + out_offset_d2 + 2] = cur_result_3;
            res_matrix[out_offset_d1 + out_offset_d2 + 3] = cur_result_4;
            res_matrix[out_offset_d1 + out_offset_d2 + 4] = cur_result_5;
            res_matrix[out_offset_d1 + out_offset_d2 + 5] = cur_result_6;
            
        }
        
    }
    
    DNN_TRACE_4("finish %s\n","");
}

void conv3D_AVX_32ch_exact_2X_NEW_k6_k6(
    const float * __restrict__ p_in_matrix, 
    int m_d1, 
    int m_d2,
    int m_d1_start,
    int m_d2_start,
    int m_d1_finish,
    int m_d2_finish,
    const float * __restrict__ p_kernel,
    const float * __restrict__ p_biases,
    int n_kernels,
    int n_maxout_kernels,
    int k_d1, 
    int k_d2,
    int n_channels,
    int n_stride,
    float * __restrict__ p_out_matrix)
{ 
    float * __restrict__ matrix = (float * __restrict__)__builtin_assume_aligned(p_in_matrix, 32);
    float * __restrict__ kernel = (float * __restrict__)__builtin_assume_aligned(p_kernel, 32);
    float * __restrict__ res_matrix = (float * __restrict__)__builtin_assume_aligned(p_out_matrix, 32);
    
	//float sum;
    float sum_arr_1_1[8];
    float sum_arr_1_2[8];
    float sum_arr_1_3[8];
    float sum_arr_1_4[8];
    float sum_arr_1_5[8];
    float sum_arr_1_6[8];
        
    DNN_TRACE_4("start %s\n","");
    DNN_TRACE_4(" p_in_matrix: %p\n", p_in_matrix);
    DNN_TRACE_4(" m_d1,m_d2: [%d,%d]\n", m_d1, m_d2);
    DNN_TRACE_4(" s_d1,s_d2: [%d,%d]\n", m_d1_start, m_d2_start);
    DNN_TRACE_4(" e_d1,e_d2: [%d,%d]\n", m_d1_finish, m_d2_finish);    
    DNN_TRACE_4(" p_kernel: %p\n", p_kernel);
    DNN_TRACE_4(" p_biases: %p\n", p_biases);
    DNN_TRACE_4(" n_kernels: %d\n", n_kernels);
    DNN_TRACE_4(" k_d1,k_d2: [%d,%d]\n", k_d1, k_d2);
    DNN_TRACE_4(" n_channels,n_stride: [%d,%d]\n", n_channels, n_stride);
    DNN_TRACE_4(" p_out_matrix: %p\n", p_out_matrix);
    
    DNN_ASSERT(n_channels == N_CHANNELS_32);
    DNN_ASSERT(k_d1 == K_D1_6);
    DNN_ASSERT(k_d2 == K_D2_6);
    //DNN_ASSERT(n_kernels == 2);
    DNN_ASSERT(n_stride == 1);
    DNN_ASSERT(m_d2 >= 6);
    
    int out_d2 = m_d2 - K_D2_6 + 1;
    int out_d3 = 1;
    //int res_idx = 0;
    
    for (int cur_m_d1 = m_d1_start; (cur_m_d1 + K_D1_6) <= m_d1_finish; cur_m_d1 += 1) {
        
        int fix_offset_1 = ((cur_m_d1) + K_D1_6) - m_d1;
        if (fix_offset_1 > 0) {
            cur_m_d1 -= fix_offset_1;
            DNN_ASSERT(cur_m_d1 >= 0);
        }
        
        int out_offset_d1 = D1_TO_OFFSET(cur_m_d1, out_d2, out_d3);
        
        for (int cur_m_d2 = m_d2_start; (cur_m_d2 + K_D2_6) <= m_d2_finish; cur_m_d2 += 6) {
            
            int fix_offset_2 = ((cur_m_d2 + 5) + K_D2_6) - m_d2;
            if (fix_offset_2 > 0) {
                cur_m_d2 -= fix_offset_2;
                DNN_ASSERT(cur_m_d2 >= 0);
            }
            
            int out_offset_d2 = D2_TO_OFFSET(cur_m_d2, out_d3);
            
            int is_first_result = 1;
            float cur_result_1 = 0;
            float cur_result_2 = 0;
            float cur_result_3 = 0;
            float cur_result_4 = 0;
            float cur_result_5 = 0;
            float cur_result_6 = 0;
            
            for (int kernel_id = 0; kernel_id < n_kernels; kernel_id++) {
                __m256 ymm0;
                __m256 ymm1;
                __m256 ymm2;
                __m256 ymm3;
                __m256 ymm4;
                __m256 ymm5;
                __m256 ymm6;
                __m256 ymm10 = _mm256_xor_ps(ymm10, ymm10); // init to 0s
                __m256 ymm11 = _mm256_xor_ps(ymm11, ymm11); // init to 0s
                __m256 ymm12 = _mm256_xor_ps(ymm12, ymm12); // init to 0s
                __m256 ymm13 = _mm256_xor_ps(ymm13, ymm13); // init to 0s
                __m256 ymm14 = _mm256_xor_ps(ymm14, ymm14); // init to 0s
                __m256 ymm15 = _mm256_xor_ps(ymm15, ymm15); // init to 0s

                int kernel_shift = DIMS_TO_SIZE(K_D1_6, K_D2_6, N_CHANNELS_32) * kernel_id;
                
                int m_offset_d1 = D1_TO_OFFSET(cur_m_d1, m_d2, N_CHANNELS_32);
                
                int k_offset_d1 = 0;
                
                for (int i = 0; i < K_D1_6; i++) {

                    int m_offset_d2 = D2_TO_OFFSET(cur_m_d2, N_CHANNELS_32);

                    int k_offset_d2 = 0;

                    for (int j = 0; j < K_D2_6; j++) {
                        
                        int m_shift = m_offset_d1 + m_offset_d2;
                        int k_shift = k_offset_d1 + k_offset_d2;
                        
                        for (int k = 0; k < N_CHANNELS_32; k += 8) {
                            
                            // K1
                            ymm0 = _mm256_load_ps(&kernel[kernel_shift + k_shift + k]);
                            
                            // M1-1
                            ymm1 = _mm256_load_ps(&matrix[m_shift + k]);
                            
                            // M1-1 * K1
                            ymm15 = _mm256_fmadd_ps(ymm0, ymm1, ymm15);
                            
                            // M1-2
                            ymm2 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_32 + k]);
                            
                            // M1-2 * K1
                            ymm14 = _mm256_fmadd_ps(ymm0, ymm2, ymm14);
                            
                            // M1-3
                            ymm3 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_32 * 2 + k]);
                            
                            // M1-3 * K1
                            ymm13 = _mm256_fmadd_ps(ymm0, ymm3, ymm13);
                            
                            // M1-4
                            ymm4 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_32 * 3 + k]);
                            
                            // M1-4 * K1
                            ymm12 = _mm256_fmadd_ps(ymm0, ymm4, ymm12);
                            
                            // M1-5
                            ymm5 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_32 * 4 + k]);
                            
                            // M1-5 * K1
                            ymm11 = _mm256_fmadd_ps(ymm0, ymm5, ymm11);
                            
                            // M1-6
                            ymm6 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_32 * 5 + k]);
                            
                            // M1-6 * K1
                            ymm10 = _mm256_fmadd_ps(ymm0, ymm6, ymm10);
                                                        
                        }
                        
                        k_offset_d2 += N_CHANNELS_32;
                    
                        m_offset_d2 += N_CHANNELS_32;
                    }
                    
                    m_offset_d1 += m_d2 * N_CHANNELS_32;
                    
                    k_offset_d1 += K_D2_6 * N_CHANNELS_32;
                }
                                
                float sum_1_1 = p_biases[kernel_id];
                float sum_1_2 = p_biases[kernel_id];
                float sum_1_3 = p_biases[kernel_id];
                float sum_1_4 = p_biases[kernel_id];
                float sum_1_5 = p_biases[kernel_id];
                float sum_1_6 = p_biases[kernel_id];
                                    
                ymm0 = _mm256_permute2f128_ps(ymm15 , ymm15 , 1);
                ymm15 = _mm256_add_ps(ymm15, ymm0);
                ymm15 = _mm256_hadd_ps(ymm15, ymm15);
                ymm15 = _mm256_hadd_ps(ymm15, ymm15);
                
                ymm1 = _mm256_permute2f128_ps(ymm14 , ymm14 , 1);
                ymm14 = _mm256_add_ps(ymm14, ymm1);
                ymm14 = _mm256_hadd_ps(ymm14, ymm14);
                ymm14 = _mm256_hadd_ps(ymm14, ymm14);
                
                ymm2 = _mm256_permute2f128_ps(ymm13 , ymm13 , 1);
                ymm13 = _mm256_add_ps(ymm13, ymm2);
                ymm13 = _mm256_hadd_ps(ymm13, ymm13);
                ymm13 = _mm256_hadd_ps(ymm13, ymm13);
                
                ymm3 = _mm256_permute2f128_ps(ymm12 , ymm12 , 1);
                ymm12 = _mm256_add_ps(ymm12, ymm3);
                ymm12 = _mm256_hadd_ps(ymm12, ymm12);
                ymm12 = _mm256_hadd_ps(ymm12, ymm12);
                
                ymm0 = _mm256_permute2f128_ps(ymm11 , ymm11 , 1);
                ymm11 = _mm256_add_ps(ymm11, ymm0);
                ymm11 = _mm256_hadd_ps(ymm11, ymm11);
                ymm11 = _mm256_hadd_ps(ymm11, ymm11);
                
                ymm1 = _mm256_permute2f128_ps(ymm10 , ymm10 , 1);
                ymm10 = _mm256_add_ps(ymm10, ymm1);
                ymm10 = _mm256_hadd_ps(ymm10, ymm10);
                ymm10 = _mm256_hadd_ps(ymm10, ymm10);
                
                _mm256_store_ps(sum_arr_1_1, ymm15);
                sum_1_1 += sum_arr_1_1[0];
                
                if (is_first_result) {
                    cur_result_1 = sum_1_1; 
                } else if (sum_1_1 > cur_result_1) {
                    cur_result_1 = sum_1_1; 
                }
                
                _mm256_store_ps(sum_arr_1_2, ymm14);
                sum_1_2 += sum_arr_1_2[0];
                
                if (is_first_result) {
                    cur_result_2 = sum_1_2; 
                } else if (sum_1_2 > cur_result_2) {
                    cur_result_2 = sum_1_2; 
                }
                
                _mm256_store_ps(sum_arr_1_3, ymm13);
                sum_1_3 += sum_arr_1_3[0];
                
                if (is_first_result) {
                    cur_result_3 = sum_1_3; 
                } else if (sum_1_3 > cur_result_3) {
                    cur_result_3 = sum_1_3; 
                }
                
                _mm256_store_ps(sum_arr_1_4, ymm12);
                sum_1_4 += sum_arr_1_4[0];
                
                if (is_first_result) {
                    cur_result_4 = sum_1_4; 
                } else if (sum_1_4 > cur_result_4) {
                    cur_result_4 = sum_1_4; 
                }
                
                _mm256_store_ps(sum_arr_1_5, ymm11);
                sum_1_5 += sum_arr_1_5[0];
                
                if (is_first_result) {
                    cur_result_5 = sum_1_5; 
                } else if (sum_1_5 > cur_result_5) {
                    cur_result_5 = sum_1_5; 
                }
                
                _mm256_store_ps(sum_arr_1_6, ymm10);
                sum_1_6 += sum_arr_1_6[0];
                
                if (is_first_result) {
                    cur_result_6 = sum_1_6; 
                } else if (sum_1_6 > cur_result_6) {
                    cur_result_6 = sum_1_6; 
                }
                
                is_first_result = 0;
            }
                        
            res_matrix[out_offset_d1 + out_offset_d2 + 0] = cur_result_1;
            res_matrix[out_offset_d1 + out_offset_d2 + 1] = cur_result_2;
            res_matrix[out_offset_d1 + out_offset_d2 + 2] = cur_result_3;
            res_matrix[out_offset_d1 + out_offset_d2 + 3] = cur_result_4;
            res_matrix[out_offset_d1 + out_offset_d2 + 4] = cur_result_5;
            res_matrix[out_offset_d1 + out_offset_d2 + 5] = cur_result_6;
            
        }
        
    }
    
    DNN_TRACE_4("finish %s\n","");
}

void conv3D_AVX_8ch_exact_2X_NEW_k6_k6(
    const float * __restrict__ p_in_matrix, 
    int m_d1, 
    int m_d2,
    int m_d1_start,
    int m_d2_start,
    int m_d1_finish,
    int m_d2_finish,
    const float * __restrict__ p_kernel,
    const float * __restrict__ p_biases,
    int n_kernels,
    int n_maxout_kernels,
    int k_d1, 
    int k_d2,
    int n_channels,
    int n_stride,
    float * __restrict__ p_out_matrix)
{ 
    float * __restrict__ matrix = (float * __restrict__)__builtin_assume_aligned(p_in_matrix, 32);
    float * __restrict__ kernel = (float * __restrict__)__builtin_assume_aligned(p_kernel, 32);
    float * __restrict__ res_matrix = (float * __restrict__)__builtin_assume_aligned(p_out_matrix, 32);
    
	//float sum;
    float sum_arr_1_1[8];
    float sum_arr_1_2[8];
    float sum_arr_1_3[8];
    float sum_arr_1_4[8];
    float sum_arr_1_5[8];
    float sum_arr_1_6[8];
        
    DNN_TRACE_4("start %s\n","");
    DNN_TRACE_4(" p_in_matrix: %p\n", p_in_matrix);
    DNN_TRACE_4(" m_d1,m_d2: [%d,%d]\n", m_d1, m_d2);
    DNN_TRACE_4(" s_d1,s_d2: [%d,%d]\n", m_d1_start, m_d2_start);
    DNN_TRACE_4(" e_d1,e_d2: [%d,%d]\n", m_d1_finish, m_d2_finish);    
    DNN_TRACE_4(" p_kernel: %p\n", p_kernel);
    DNN_TRACE_4(" p_biases: %p\n", p_biases);
    DNN_TRACE_4(" n_kernels: %d\n", n_kernels);
    DNN_TRACE_4(" k_d1,k_d2: [%d,%d]\n", k_d1, k_d2);
    DNN_TRACE_4(" n_channels,n_stride: [%d,%d]\n", n_channels, n_stride);
    DNN_TRACE_4(" p_out_matrix: %p\n", p_out_matrix);
    
    DNN_ASSERT(n_channels == N_CHANNELS_8);
    DNN_ASSERT(k_d1 == K_D1_6);
    DNN_ASSERT(k_d2 == K_D2_6);
    //DNN_ASSERT(n_kernels == 2);
    DNN_ASSERT(n_stride == 1);
    DNN_ASSERT(m_d2 >= 6);
    
    int out_d2 = m_d2 - K_D2_6 + 1;
    int out_d3 = 1;
    //int res_idx = 0;
    
    for (int cur_m_d1 = m_d1_start; (cur_m_d1 + K_D1_6) <= m_d1_finish; cur_m_d1 += 1) {
        
        int fix_offset_1 = ((cur_m_d1) + K_D1_6) - m_d1;
        if (fix_offset_1 > 0) {
            cur_m_d1 -= fix_offset_1;
            DNN_ASSERT(cur_m_d1 >= 0);
        }
        
        int out_offset_d1 = D1_TO_OFFSET(cur_m_d1, out_d2, out_d3);
        
        for (int cur_m_d2 = m_d2_start; (cur_m_d2 + K_D2_6) <= m_d2_finish; cur_m_d2 += 6) {
            
            int fix_offset_2 = ((cur_m_d2 + 5) + K_D2_6) - m_d2;
            if (fix_offset_2 > 0) {
                cur_m_d2 -= fix_offset_2;
                DNN_ASSERT(cur_m_d2 >= 0);
            }
            
            int out_offset_d2 = D2_TO_OFFSET(cur_m_d2, out_d3);
            
            int is_first_result = 1;
            float cur_result_1 = 0;
            float cur_result_2 = 0;
            float cur_result_3 = 0;
            float cur_result_4 = 0;
            float cur_result_5 = 0;
            float cur_result_6 = 0;
            
            for (int kernel_id = 0; kernel_id < n_kernels; kernel_id++) {
                __m256 ymm0;
                __m256 ymm1;
                __m256 ymm2;
                __m256 ymm3;
                __m256 ymm4;
                __m256 ymm5;
                __m256 ymm6;
                __m256 ymm10 = _mm256_xor_ps(ymm10, ymm10); // init to 0s
                __m256 ymm11 = _mm256_xor_ps(ymm11, ymm11); // init to 0s
                __m256 ymm12 = _mm256_xor_ps(ymm12, ymm12); // init to 0s
                __m256 ymm13 = _mm256_xor_ps(ymm13, ymm13); // init to 0s
                __m256 ymm14 = _mm256_xor_ps(ymm14, ymm14); // init to 0s
                __m256 ymm15 = _mm256_xor_ps(ymm15, ymm15); // init to 0s

                int kernel_shift = DIMS_TO_SIZE(K_D1_6, K_D2_6, N_CHANNELS_8) * kernel_id;
                
                int m_offset_d1 = D1_TO_OFFSET(cur_m_d1, m_d2, N_CHANNELS_8);
                
                int k_offset_d1 = 0;
                
                for (int i = 0; i < K_D1_6; i++) {

                    int m_offset_d2 = D2_TO_OFFSET(cur_m_d2, N_CHANNELS_8);

                    int k_offset_d2 = 0;

                    for (int j = 0; j < K_D2_6; j++) {
                        
                        int m_shift = m_offset_d1 + m_offset_d2;
                        int k_shift = k_offset_d1 + k_offset_d2;
                        
                        for (int k = 0; k < N_CHANNELS_8; k += 8) {
                            
                            // K1
                            ymm0 = _mm256_load_ps(&kernel[kernel_shift + k_shift + k]);
                            
                            // M1-1
                            ymm1 = _mm256_load_ps(&matrix[m_shift + k]);
                            
                            // M1-1 * K1
                            ymm15 = _mm256_fmadd_ps(ymm0, ymm1, ymm15);
                            
                            // M1-2
                            ymm2 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_8 + k]);
                            
                            // M1-2 * K1
                            ymm14 = _mm256_fmadd_ps(ymm0, ymm2, ymm14);
                            
                            // M1-3
                            ymm3 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_8 * 2 + k]);
                            
                            // M1-3 * K1
                            ymm13 = _mm256_fmadd_ps(ymm0, ymm3, ymm13);
                            
                            // M1-4
                            ymm4 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_8 * 3 + k]);
                            
                            // M1-4 * K1
                            ymm12 = _mm256_fmadd_ps(ymm0, ymm4, ymm12);
                            
                            // M1-5
                            ymm5 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_8 * 4 + k]);
                            
                            // M1-5 * K1
                            ymm11 = _mm256_fmadd_ps(ymm0, ymm5, ymm11);
                            
                            // M1-6
                            ymm6 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_8 * 5 + k]);
                            
                            // M1-6 * K1
                            ymm10 = _mm256_fmadd_ps(ymm0, ymm6, ymm10);
                                                        
                        }
                        
                        k_offset_d2 += N_CHANNELS_8;
                    
                        m_offset_d2 += N_CHANNELS_8;
                    }
                    
                    m_offset_d1 += m_d2 * N_CHANNELS_8;
                    
                    k_offset_d1 += K_D2_6 * N_CHANNELS_8;
                }
                                
                float sum_1_1 = p_biases[kernel_id];
                float sum_1_2 = p_biases[kernel_id];
                float sum_1_3 = p_biases[kernel_id];
                float sum_1_4 = p_biases[kernel_id];
                float sum_1_5 = p_biases[kernel_id];
                float sum_1_6 = p_biases[kernel_id];
                                    
                ymm0 = _mm256_permute2f128_ps(ymm15 , ymm15 , 1);
                ymm15 = _mm256_add_ps(ymm15, ymm0);
                ymm15 = _mm256_hadd_ps(ymm15, ymm15);
                ymm15 = _mm256_hadd_ps(ymm15, ymm15);
                
                ymm1 = _mm256_permute2f128_ps(ymm14 , ymm14 , 1);
                ymm14 = _mm256_add_ps(ymm14, ymm1);
                ymm14 = _mm256_hadd_ps(ymm14, ymm14);
                ymm14 = _mm256_hadd_ps(ymm14, ymm14);
                
                ymm2 = _mm256_permute2f128_ps(ymm13 , ymm13 , 1);
                ymm13 = _mm256_add_ps(ymm13, ymm2);
                ymm13 = _mm256_hadd_ps(ymm13, ymm13);
                ymm13 = _mm256_hadd_ps(ymm13, ymm13);
                
                ymm3 = _mm256_permute2f128_ps(ymm12 , ymm12 , 1);
                ymm12 = _mm256_add_ps(ymm12, ymm3);
                ymm12 = _mm256_hadd_ps(ymm12, ymm12);
                ymm12 = _mm256_hadd_ps(ymm12, ymm12);
                
                ymm0 = _mm256_permute2f128_ps(ymm11 , ymm11 , 1);
                ymm11 = _mm256_add_ps(ymm11, ymm0);
                ymm11 = _mm256_hadd_ps(ymm11, ymm11);
                ymm11 = _mm256_hadd_ps(ymm11, ymm11);
                
                ymm1 = _mm256_permute2f128_ps(ymm10 , ymm10 , 1);
                ymm10 = _mm256_add_ps(ymm10, ymm1);
                ymm10 = _mm256_hadd_ps(ymm10, ymm10);
                ymm10 = _mm256_hadd_ps(ymm10, ymm10);
                
                _mm256_store_ps(sum_arr_1_1, ymm15);
                sum_1_1 += sum_arr_1_1[0];
                
                if (is_first_result) {
                    cur_result_1 = sum_1_1; 
                } else if (sum_1_1 > cur_result_1) {
                    cur_result_1 = sum_1_1; 
                }
                
                _mm256_store_ps(sum_arr_1_2, ymm14);
                sum_1_2 += sum_arr_1_2[0];
                
                if (is_first_result) {
                    cur_result_2 = sum_1_2; 
                } else if (sum_1_2 > cur_result_2) {
                    cur_result_2 = sum_1_2; 
                }
                
                _mm256_store_ps(sum_arr_1_3, ymm13);
                sum_1_3 += sum_arr_1_3[0];
                
                if (is_first_result) {
                    cur_result_3 = sum_1_3; 
                } else if (sum_1_3 > cur_result_3) {
                    cur_result_3 = sum_1_3; 
                }
                
                _mm256_store_ps(sum_arr_1_4, ymm12);
                sum_1_4 += sum_arr_1_4[0];
                
                if (is_first_result) {
                    cur_result_4 = sum_1_4; 
                } else if (sum_1_4 > cur_result_4) {
                    cur_result_4 = sum_1_4; 
                }
                
                _mm256_store_ps(sum_arr_1_5, ymm11);
                sum_1_5 += sum_arr_1_5[0];
                
                if (is_first_result) {
                    cur_result_5 = sum_1_5; 
                } else if (sum_1_5 > cur_result_5) {
                    cur_result_5 = sum_1_5; 
                }
                
                _mm256_store_ps(sum_arr_1_6, ymm10);
                sum_1_6 += sum_arr_1_6[0];
                
                if (is_first_result) {
                    cur_result_6 = sum_1_6; 
                } else if (sum_1_6 > cur_result_6) {
                    cur_result_6 = sum_1_6; 
                }
                
                is_first_result = 0;
            }
                        
            res_matrix[out_offset_d1 + out_offset_d2 + 0] = cur_result_1;
            res_matrix[out_offset_d1 + out_offset_d2 + 1] = cur_result_2;
            res_matrix[out_offset_d1 + out_offset_d2 + 2] = cur_result_3;
            res_matrix[out_offset_d1 + out_offset_d2 + 3] = cur_result_4;
            res_matrix[out_offset_d1 + out_offset_d2 + 4] = cur_result_5;
            res_matrix[out_offset_d1 + out_offset_d2 + 5] = cur_result_6;
            
        }
        
    }
    
    DNN_TRACE_4("finish %s\n","");
}

void conv3D_AVX_8ch_exact_2k_2X_NEW(
    const float * __restrict__ p_in_matrix, 
    int m_d1, 
    int m_d2,
    int m_d1_start,
    int m_d2_start,
    int m_d1_finish,
    int m_d2_finish,
    const float * __restrict__ p_kernel,
    const float * __restrict__ p_biases,
    int n_kernels,
    int n_maxout_kernels,
    int k_d1, 
    int k_d2,
    int n_channels,
    int n_stride,
    float * __restrict__ p_out_matrix)
{ 
    float * __restrict__ matrix = (float * __restrict__)__builtin_assume_aligned(p_in_matrix, 32);
    float * __restrict__ kernel = (float * __restrict__)__builtin_assume_aligned(p_kernel, 32);
    float * __restrict__ res_matrix = (float * __restrict__)__builtin_assume_aligned(p_out_matrix, 32);
    
	//float sum;
    float sum_arr_1_1[8];
    float sum_arr_1_2[8];
    float sum_arr_1_3[8];
    float sum_arr_1_4[8];
    float sum_arr_1_5[8];
    float sum_arr_1_6[8];
    
    float sum_arr_2_1[8];
    float sum_arr_2_2[8];
    float sum_arr_2_3[8];
    float sum_arr_2_4[8];
    float sum_arr_2_5[8];
    float sum_arr_2_6[8];
	
    DNN_TRACE_4("start %s\n","");
    DNN_TRACE_4(" p_in_matrix: %p\n", p_in_matrix);
    DNN_TRACE_4(" m_d1,m_d2: [%d,%d]\n", m_d1, m_d2);
    DNN_TRACE_4(" s_d1,s_d2: [%d,%d]\n", m_d1_start, m_d2_start);
    DNN_TRACE_4(" e_d1,e_d2: [%d,%d]\n", m_d1_finish, m_d2_finish);    
    DNN_TRACE_4(" p_kernel: %p\n", p_kernel);
    DNN_TRACE_4(" p_biases: %p\n", p_biases);
    DNN_TRACE_4(" n_kernels: %d\n", n_kernels);
    DNN_TRACE_4(" k_d1,k_d2: [%d,%d]\n", k_d1, k_d2);
    DNN_TRACE_4(" n_channels,n_stride: [%d,%d]\n", n_channels, n_stride);
    DNN_TRACE_4(" p_out_matrix: %p\n", p_out_matrix);
    
    DNN_ASSERT(n_channels == N_CHANNELS_8);
    DNN_ASSERT(k_d1 == K_D1_4);
    DNN_ASSERT(k_d2 == K_D2_4);
    DNN_ASSERT(n_kernels == 2);
    DNN_ASSERT(n_stride == 1);
    DNN_ASSERT(m_d2 >= 4);
    
    int out_d2 = m_d2 - K_D2_4 + 1;
    int out_d3 = 1;
    //int res_idx = 0;
    
    for (int cur_m_d1 = m_d1_start; (cur_m_d1 + K_D1_4) <= m_d1_finish; cur_m_d1 += 1) {
        
        int fix_offset_1 = ((cur_m_d1) + K_D1_4) - m_d1;
        if (fix_offset_1 > 0) {
            cur_m_d1 -= fix_offset_1;
            DNN_ASSERT(cur_m_d1 >= 0);
        }
        
        int out_offset_d1 = D1_TO_OFFSET(cur_m_d1, out_d2, out_d3);
        
        for (int cur_m_d2 = m_d2_start; (cur_m_d2 + K_D2_4) <= m_d2_finish; cur_m_d2 += 6) {
            
            int fix_offset_2 = ((cur_m_d2 + 5) + K_D2_4) - m_d2;
            if (fix_offset_2 > 0) {
                cur_m_d2 -= fix_offset_2;
                DNN_ASSERT(cur_m_d2 >= 0);
            }
            
            int out_offset_d2 = D2_TO_OFFSET(cur_m_d2, out_d3);
            
            //int is_first_result = 1;
            float cur_result_1 = 0;
            float cur_result_2 = 0;
            float cur_result_3 = 0;
            float cur_result_4 = 0;
            float cur_result_5 = 0;
            float cur_result_6 = 0;
            
            //for (int kernel_id = 0; kernel_id < n_kernels; kernel_id++) {
                __m256 ymm0;
                __m256 ymm1;
                __m256 ymm2;
                __m256 ymm3;
                __m256 ymm4 = _mm256_xor_ps(ymm4, ymm4); // init to 0s
                __m256 ymm5 = _mm256_xor_ps(ymm5, ymm5); // init to 0s
                __m256 ymm6 = _mm256_xor_ps(ymm6, ymm6); // init to 0s
                __m256 ymm7 = _mm256_xor_ps(ymm7, ymm7); // init to 0s
                __m256 ymm8 = _mm256_xor_ps(ymm8, ymm8); // init to 0s
                __m256 ymm9 = _mm256_xor_ps(ymm9, ymm9); // init to 0s
                __m256 ymm10 = _mm256_xor_ps(ymm10, ymm10); // init to 0s
                __m256 ymm11 = _mm256_xor_ps(ymm11, ymm11); // init to 0s
                __m256 ymm15 = _mm256_xor_ps(ymm15, ymm15); // init to 0s
                __m256 ymm14 = _mm256_xor_ps(ymm14, ymm14); // init to 0s
                __m256 ymm13 = _mm256_xor_ps(ymm13, ymm13); // init to 0s
                __m256 ymm12 = _mm256_xor_ps(ymm12, ymm12); // init to 0s

                int kernel_shift = DIMS_TO_SIZE(K_D1_4, K_D2_4, N_CHANNELS_8);
                
                int m_offset_d1 = D1_TO_OFFSET(cur_m_d1, m_d2, N_CHANNELS_8);
                
                int k_offset_d1 = 0;
                
                for (int i = 0; i < K_D1_4; i++) {

                    int m_offset_d2 = D2_TO_OFFSET(cur_m_d2, N_CHANNELS_8);

                    int k_offset_d2 = 0;

                    for (int j = 0; j < K_D2_4; j++) {
                        
                        int m_shift = m_offset_d1 + m_offset_d2;
                        int k_shift = k_offset_d1 + k_offset_d2;
                        
                        int k = 0;
                        //for (int k = 0; k < N_CHANNELS_8; k += 8) {
                            
                            // M1-1
                            ymm2 = _mm256_load_ps(&matrix[m_shift + k]);
                            
                            // K1
                            ymm0 = _mm256_load_ps(&kernel[k_shift + k]);
                            
                            // M1-1 * K1
                            ymm15 = _mm256_fmadd_ps(ymm0, ymm2, ymm15);
                                    
                            // K2
                            ymm1 = _mm256_load_ps(&kernel[kernel_shift + k_shift + k]);
                            
                            // M1-1 * K2
                            ymm14 = _mm256_fmadd_ps(ymm1, ymm2, ymm14);
                            
                            /////
                            // M1-3
                            ymm3 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_8 * 2 + k]);
                            
                            // M1-3 * K1
                            ymm7 = _mm256_fmadd_ps(ymm0, ymm3, ymm7);
                            
                            // M1-3 * K2
                            ymm6 = _mm256_fmadd_ps(ymm1, ymm3, ymm6);
                            
                            /////
                            // M1-2
                            ymm2 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_8 + k]);
                            
                            // M1-2 * K1
                            ymm13 = _mm256_fmadd_ps(ymm0, ymm2, ymm13);
                            
                            // M1-2 * K2
                            ymm12 = _mm256_fmadd_ps(ymm1, ymm2, ymm12);
                            
                            /////
                            // M1-4
                            ymm3 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_8 * 3 + k]);
                            
                            // M1-4 * K1
                            ymm5 = _mm256_fmadd_ps(ymm0, ymm3, ymm5);
                            
                            // M1-4 * K2
                            ymm4 = _mm256_fmadd_ps(ymm1, ymm3, ymm4);
                            
                            // M1-5
                            ymm2 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_8 * 4 + k]);
                            
                            // M1-5 * K1
                            ymm11 = _mm256_fmadd_ps(ymm0, ymm2, ymm11);
                            
                            // M1-5 * K2
                            ymm10 = _mm256_fmadd_ps(ymm1, ymm2, ymm10);
                            
                            // M1-6
                            ymm2 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_8 * 5 + k]);
                            
                            // M1-6 * K1
                            ymm9 = _mm256_fmadd_ps(ymm0, ymm2, ymm9);
                            
                            // M1-6 * K2
                            ymm8 = _mm256_fmadd_ps(ymm1, ymm2, ymm8);
                            
                        //}
                        
                        k_offset_d2 += N_CHANNELS_8;
                    
                        m_offset_d2 += N_CHANNELS_8;
                    }
                    
                    m_offset_d1 += m_d2 * N_CHANNELS_8;
                    
                    k_offset_d1 += K_D2_4 * N_CHANNELS_8;
                }
                
                //ymm15 = _mm256_add_ps(ymm15, ymm14);
                //ymm13 = _mm256_add_ps(ymm13, ymm12);
                //ymm11 = _mm256_add_ps(ymm11, ymm10);
                //ymm9 = _mm256_add_ps(ymm9, ymm8);
                
                float sum_1_1 = p_biases[0];
                float sum_1_2 = p_biases[0];
                float sum_1_3 = p_biases[0];
                float sum_1_4 = p_biases[0];
                float sum_1_5 = p_biases[0];
                float sum_1_6 = p_biases[0];
                
                float sum_2_1 = p_biases[1];
                float sum_2_2 = p_biases[1];
                float sum_2_3 = p_biases[1];
                float sum_2_4 = p_biases[1];
                float sum_2_5 = p_biases[1];
                float sum_2_6 = p_biases[1];
                                    
                ymm0 = _mm256_permute2f128_ps(ymm15 , ymm15 , 1);
                ymm15 = _mm256_add_ps(ymm15, ymm0);
                ymm15 = _mm256_hadd_ps(ymm15, ymm15);
                ymm15 = _mm256_hadd_ps(ymm15, ymm15);
                
                ymm1 = _mm256_permute2f128_ps(ymm14 , ymm14 , 1);
                ymm14 = _mm256_add_ps(ymm14, ymm1);
                ymm14 = _mm256_hadd_ps(ymm14, ymm14);
                ymm14 = _mm256_hadd_ps(ymm14, ymm14);
                
                ymm2 = _mm256_permute2f128_ps(ymm13 , ymm13 , 1);
                ymm13 = _mm256_add_ps(ymm13, ymm2);
                ymm13 = _mm256_hadd_ps(ymm13, ymm13);
                ymm13 = _mm256_hadd_ps(ymm13, ymm13);
                
                ymm3 = _mm256_permute2f128_ps(ymm12 , ymm12 , 1);
                ymm12 = _mm256_add_ps(ymm12, ymm3);
                ymm12 = _mm256_hadd_ps(ymm12, ymm12);
                ymm12 = _mm256_hadd_ps(ymm12, ymm12);
                
                ymm0 = _mm256_permute2f128_ps(ymm11 , ymm11 , 1);
                ymm11 = _mm256_add_ps(ymm11, ymm0);
                ymm11 = _mm256_hadd_ps(ymm11, ymm11);
                ymm11 = _mm256_hadd_ps(ymm11, ymm11);
                
                ymm1 = _mm256_permute2f128_ps(ymm10 , ymm10 , 1);
                ymm10 = _mm256_add_ps(ymm10, ymm1);
                ymm10 = _mm256_hadd_ps(ymm10, ymm10);
                ymm10 = _mm256_hadd_ps(ymm10, ymm10);
                
                ymm2 = _mm256_permute2f128_ps(ymm9 , ymm9 , 1);
                ymm9 = _mm256_add_ps(ymm9, ymm2);
                ymm9 = _mm256_hadd_ps(ymm9, ymm9);
                ymm9 = _mm256_hadd_ps(ymm9, ymm9);
                
                ymm3 = _mm256_permute2f128_ps(ymm8 , ymm8 , 1);
                ymm8 = _mm256_add_ps(ymm8, ymm3);
                ymm8 = _mm256_hadd_ps(ymm8, ymm8);
                ymm8 = _mm256_hadd_ps(ymm8, ymm8);
                
                ymm0 = _mm256_permute2f128_ps(ymm7 , ymm7 , 1);
                ymm7 = _mm256_add_ps(ymm7, ymm0);
                ymm7 = _mm256_hadd_ps(ymm7, ymm7);
                ymm7 = _mm256_hadd_ps(ymm7, ymm7);
                
                ymm1 = _mm256_permute2f128_ps(ymm6 , ymm6 , 1);
                ymm6 = _mm256_add_ps(ymm6, ymm1);
                ymm6 = _mm256_hadd_ps(ymm6, ymm6);
                ymm6 = _mm256_hadd_ps(ymm6, ymm6);
                
                ymm2 = _mm256_permute2f128_ps(ymm5 , ymm5 , 1);
                ymm5 = _mm256_add_ps(ymm5, ymm2);
                ymm5 = _mm256_hadd_ps(ymm5, ymm5);
                ymm5 = _mm256_hadd_ps(ymm5, ymm5);
                
                ymm3 = _mm256_permute2f128_ps(ymm4 , ymm4 , 1);
                ymm4 = _mm256_add_ps(ymm4, ymm3);
                ymm4 = _mm256_hadd_ps(ymm4, ymm4);
                ymm4 = _mm256_hadd_ps(ymm4, ymm4);
                
                _mm256_store_ps(sum_arr_1_1, ymm15);
                sum_1_1 += sum_arr_1_1[0];
                
                _mm256_store_ps(sum_arr_2_1, ymm14);
                sum_2_1 += sum_arr_2_1[0];
                
                if (sum_1_1 > sum_2_1) {
                    cur_result_1 = sum_1_1;
                } else {
                    cur_result_1 = sum_2_1;
                }
                
                _mm256_store_ps(sum_arr_1_2, ymm13);
                sum_1_2 += sum_arr_1_2[0];
                
                _mm256_store_ps(sum_arr_2_2, ymm12);
                sum_2_2 += sum_arr_2_2[0];
                
                if (sum_1_2 > sum_2_2) {
                    cur_result_2 = sum_1_2;
                } else {
                    cur_result_2 = sum_2_2;
                }
                
                _mm256_store_ps(sum_arr_1_3, ymm11);
                sum_1_3 += sum_arr_1_3[0];
                
                _mm256_store_ps(sum_arr_2_3, ymm10);
                sum_2_3 += sum_arr_2_3[0];
                
                if (sum_1_3 > sum_2_3) {
                    cur_result_3 = sum_1_3;
                } else {
                    cur_result_3 = sum_2_3;
                }
                
                _mm256_store_ps(sum_arr_1_4, ymm9);
                sum_1_4 += sum_arr_1_4[0];
                
                _mm256_store_ps(sum_arr_2_4, ymm8);
                sum_2_4 += sum_arr_2_4[0];
                
                if (sum_1_4 > sum_2_4) {
                    cur_result_4 = sum_1_4;
                } else {
                    cur_result_4 = sum_2_4;
                }
                
                _mm256_store_ps(sum_arr_1_5, ymm7);
                sum_1_5 += sum_arr_1_5[0];
                
                _mm256_store_ps(sum_arr_2_5, ymm6);
                sum_2_5 += sum_arr_2_5[0];
                
                if (sum_1_5 > sum_2_5) {
                    cur_result_5 = sum_1_5;
                } else {
                    cur_result_5 = sum_2_5;
                }
                
                _mm256_store_ps(sum_arr_1_6, ymm5);
                sum_1_6 += sum_arr_1_6[0];
                
                _mm256_store_ps(sum_arr_2_6, ymm4);
                sum_2_6 += sum_arr_2_6[0];
                
                if (sum_1_6 > sum_2_6) {
                    cur_result_6 = sum_1_6;
                } else {
                    cur_result_6 = sum_2_6;
                }
                
                
            //}
                        
            res_matrix[out_offset_d1 + out_offset_d2 + 0] = cur_result_1;
            res_matrix[out_offset_d1 + out_offset_d2 + 1] = cur_result_2;
            res_matrix[out_offset_d1 + out_offset_d2 + 2] = cur_result_5;
            res_matrix[out_offset_d1 + out_offset_d2 + 3] = cur_result_6;
            res_matrix[out_offset_d1 + out_offset_d2 + 4] = cur_result_3;
            res_matrix[out_offset_d1 + out_offset_d2 + 5] = cur_result_4;
            
        }
        
    }
    
    DNN_TRACE_4("finish %s\n","");
}

void conv3D_AVX_8ch_exact_2k_2X_NEW_k2_k2(
    const float * __restrict__ p_in_matrix, 
    int m_d1, 
    int m_d2,
    int m_d1_start,
    int m_d2_start,
    int m_d1_finish,
    int m_d2_finish,
    const float * __restrict__ p_kernel,
    const float * __restrict__ p_biases,
    int n_kernels,
    int n_maxout_kernels,
    int k_d1, 
    int k_d2,
    int n_channels,
    int n_stride,
    float * __restrict__ p_out_matrix)
{ 
    float * __restrict__ matrix = (float * __restrict__)__builtin_assume_aligned(p_in_matrix, 32);
    float * __restrict__ kernel = (float * __restrict__)__builtin_assume_aligned(p_kernel, 32);
    float * __restrict__ res_matrix = (float * __restrict__)__builtin_assume_aligned(p_out_matrix, 32);
    
	//float sum;
    float sum_arr_1_1[8];
    float sum_arr_1_2[8];
    float sum_arr_1_3[8];
    float sum_arr_1_4[8];
    float sum_arr_1_5[8];
    float sum_arr_1_6[8];
    
    float sum_arr_2_1[8];
    float sum_arr_2_2[8];
    float sum_arr_2_3[8];
    float sum_arr_2_4[8];
    float sum_arr_2_5[8];
    float sum_arr_2_6[8];
	
    DNN_TRACE_4("start %s\n","");
    DNN_TRACE_4(" p_in_matrix: %p\n", p_in_matrix);
    DNN_TRACE_4(" m_d1,m_d2: [%d,%d]\n", m_d1, m_d2);
    DNN_TRACE_4(" s_d1,s_d2: [%d,%d]\n", m_d1_start, m_d2_start);
    DNN_TRACE_4(" e_d1,e_d2: [%d,%d]\n", m_d1_finish, m_d2_finish);    
    DNN_TRACE_4(" p_kernel: %p\n", p_kernel);
    DNN_TRACE_4(" p_biases: %p\n", p_biases);
    DNN_TRACE_4(" n_kernels: %d\n", n_kernels);
    DNN_TRACE_4(" k_d1,k_d2: [%d,%d]\n", k_d1, k_d2);
    DNN_TRACE_4(" n_channels,n_stride: [%d,%d]\n", n_channels, n_stride);
    DNN_TRACE_4(" p_out_matrix: %p\n", p_out_matrix);
    
    DNN_ASSERT(n_channels == N_CHANNELS_8);
    DNN_ASSERT(k_d1 == K_D1_2);
    DNN_ASSERT(k_d2 == K_D2_2);
    DNN_ASSERT(n_kernels == 2);
    DNN_ASSERT(n_stride == 1);
    DNN_ASSERT(m_d2 >= 4);
    
    int out_d2 = m_d2 - K_D2_2 + 1;
    int out_d3 = 1;
    //int res_idx = 0;
    
    for (int cur_m_d1 = m_d1_start; (cur_m_d1 + K_D1_2) <= m_d1_finish; cur_m_d1 += 1) {
        
        int fix_offset_1 = ((cur_m_d1) + K_D1_2) - m_d1;
        if (fix_offset_1 > 0) {
            cur_m_d1 -= fix_offset_1;
            DNN_ASSERT(cur_m_d1 >= 0);
        }
        
        int out_offset_d1 = D1_TO_OFFSET(cur_m_d1, out_d2, out_d3);
        
        for (int cur_m_d2 = m_d2_start; (cur_m_d2 + K_D2_2) <= m_d2_finish; cur_m_d2 += 6) {
            
            int fix_offset_2 = ((cur_m_d2 + 5) + K_D2_2) - m_d2;
            if (fix_offset_2 > 0) {
                cur_m_d2 -= fix_offset_2;
                DNN_ASSERT(cur_m_d2 >= 0);
            }
            
            int out_offset_d2 = D2_TO_OFFSET(cur_m_d2, out_d3);
            
            //int is_first_result = 1;
            float cur_result_1 = 0;
            float cur_result_2 = 0;
            float cur_result_3 = 0;
            float cur_result_4 = 0;
            float cur_result_5 = 0;
            float cur_result_6 = 0;
            
            //for (int kernel_id = 0; kernel_id < n_kernels; kernel_id++) {
                __m256 ymm0;
                __m256 ymm1;
                __m256 ymm2;
                __m256 ymm3;
                __m256 ymm4 = _mm256_xor_ps(ymm4, ymm4); // init to 0s
                __m256 ymm5 = _mm256_xor_ps(ymm5, ymm5); // init to 0s
                __m256 ymm6 = _mm256_xor_ps(ymm6, ymm6); // init to 0s
                __m256 ymm7 = _mm256_xor_ps(ymm7, ymm7); // init to 0s
                __m256 ymm8 = _mm256_xor_ps(ymm8, ymm8); // init to 0s
                __m256 ymm9 = _mm256_xor_ps(ymm9, ymm9); // init to 0s
                __m256 ymm10 = _mm256_xor_ps(ymm10, ymm10); // init to 0s
                __m256 ymm11 = _mm256_xor_ps(ymm11, ymm11); // init to 0s
                __m256 ymm15 = _mm256_xor_ps(ymm15, ymm15); // init to 0s
                __m256 ymm14 = _mm256_xor_ps(ymm14, ymm14); // init to 0s
                __m256 ymm13 = _mm256_xor_ps(ymm13, ymm13); // init to 0s
                __m256 ymm12 = _mm256_xor_ps(ymm12, ymm12); // init to 0s

                int kernel_shift = DIMS_TO_SIZE(K_D1_2, K_D2_2, N_CHANNELS_8);
                
                int m_offset_d1 = D1_TO_OFFSET(cur_m_d1, m_d2, N_CHANNELS_8);
                
                int k_offset_d1 = 0;
                
                for (int i = 0; i < K_D1_2; i++) {

                    int m_offset_d2 = D2_TO_OFFSET(cur_m_d2, N_CHANNELS_8);

                    int k_offset_d2 = 0;

                    for (int j = 0; j < K_D2_2; j++) {
                        
                        int m_shift = m_offset_d1 + m_offset_d2;
                        int k_shift = k_offset_d1 + k_offset_d2;
                        
                        int k = 0;
                        //for (int k = 0; k < N_CHANNELS_8; k += 8) {
                            
                            // M1-1
                            ymm2 = _mm256_load_ps(&matrix[m_shift + k]);
                            
                            // K1
                            ymm0 = _mm256_load_ps(&kernel[k_shift + k]);
                            
                            // M1-1 * K1
                            ymm15 = _mm256_fmadd_ps(ymm0, ymm2, ymm15);
                                    
                            // K2
                            ymm1 = _mm256_load_ps(&kernel[kernel_shift + k_shift + k]);
                            
                            // M1-1 * K2
                            ymm14 = _mm256_fmadd_ps(ymm1, ymm2, ymm14);
                            
                            /////
                            // M1-3
                            ymm3 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_8 * 2 + k]);
                            
                            // M1-3 * K1
                            ymm7 = _mm256_fmadd_ps(ymm0, ymm3, ymm7);
                            
                            // M1-3 * K2
                            ymm6 = _mm256_fmadd_ps(ymm1, ymm3, ymm6);
                            
                            /////
                            // M1-2
                            ymm2 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_8 + k]);
                            
                            // M1-2 * K1
                            ymm13 = _mm256_fmadd_ps(ymm0, ymm2, ymm13);
                            
                            // M1-2 * K2
                            ymm12 = _mm256_fmadd_ps(ymm1, ymm2, ymm12);
                            
                            /////
                            // M1-4
                            ymm3 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_8 * 3 + k]);
                            
                            // M1-4 * K1
                            ymm5 = _mm256_fmadd_ps(ymm0, ymm3, ymm5);
                            
                            // M1-4 * K2
                            ymm4 = _mm256_fmadd_ps(ymm1, ymm3, ymm4);
                            
                            // M1-5
                            ymm2 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_8 * 4 + k]);
                            
                            // M1-5 * K1
                            ymm11 = _mm256_fmadd_ps(ymm0, ymm2, ymm11);
                            
                            // M1-5 * K2
                            ymm10 = _mm256_fmadd_ps(ymm1, ymm2, ymm10);
                            
                            // M1-6
                            ymm2 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_8 * 5 + k]);
                            
                            // M1-6 * K1
                            ymm9 = _mm256_fmadd_ps(ymm0, ymm2, ymm9);
                            
                            // M1-6 * K2
                            ymm8 = _mm256_fmadd_ps(ymm1, ymm2, ymm8);
                            
                        //}
                        
                        k_offset_d2 += N_CHANNELS_8;
                    
                        m_offset_d2 += N_CHANNELS_8;
                    }
                    
                    m_offset_d1 += m_d2 * N_CHANNELS_8;
                    
                    k_offset_d1 += K_D2_2 * N_CHANNELS_8;
                }
                
                //ymm15 = _mm256_add_ps(ymm15, ymm14);
                //ymm13 = _mm256_add_ps(ymm13, ymm12);
                //ymm11 = _mm256_add_ps(ymm11, ymm10);
                //ymm9 = _mm256_add_ps(ymm9, ymm8);
                
                float sum_1_1 = p_biases[0];
                float sum_1_2 = p_biases[0];
                float sum_1_3 = p_biases[0];
                float sum_1_4 = p_biases[0];
                float sum_1_5 = p_biases[0];
                float sum_1_6 = p_biases[0];
                
                float sum_2_1 = p_biases[1];
                float sum_2_2 = p_biases[1];
                float sum_2_3 = p_biases[1];
                float sum_2_4 = p_biases[1];
                float sum_2_5 = p_biases[1];
                float sum_2_6 = p_biases[1];
                                    
                ymm0 = _mm256_permute2f128_ps(ymm15 , ymm15 , 1);
                ymm15 = _mm256_add_ps(ymm15, ymm0);
                ymm15 = _mm256_hadd_ps(ymm15, ymm15);
                ymm15 = _mm256_hadd_ps(ymm15, ymm15);
                
                ymm1 = _mm256_permute2f128_ps(ymm14 , ymm14 , 1);
                ymm14 = _mm256_add_ps(ymm14, ymm1);
                ymm14 = _mm256_hadd_ps(ymm14, ymm14);
                ymm14 = _mm256_hadd_ps(ymm14, ymm14);
                
                ymm2 = _mm256_permute2f128_ps(ymm13 , ymm13 , 1);
                ymm13 = _mm256_add_ps(ymm13, ymm2);
                ymm13 = _mm256_hadd_ps(ymm13, ymm13);
                ymm13 = _mm256_hadd_ps(ymm13, ymm13);
                
                ymm3 = _mm256_permute2f128_ps(ymm12 , ymm12 , 1);
                ymm12 = _mm256_add_ps(ymm12, ymm3);
                ymm12 = _mm256_hadd_ps(ymm12, ymm12);
                ymm12 = _mm256_hadd_ps(ymm12, ymm12);
                
                ymm0 = _mm256_permute2f128_ps(ymm11 , ymm11 , 1);
                ymm11 = _mm256_add_ps(ymm11, ymm0);
                ymm11 = _mm256_hadd_ps(ymm11, ymm11);
                ymm11 = _mm256_hadd_ps(ymm11, ymm11);
                
                ymm1 = _mm256_permute2f128_ps(ymm10 , ymm10 , 1);
                ymm10 = _mm256_add_ps(ymm10, ymm1);
                ymm10 = _mm256_hadd_ps(ymm10, ymm10);
                ymm10 = _mm256_hadd_ps(ymm10, ymm10);
                
                ymm2 = _mm256_permute2f128_ps(ymm9 , ymm9 , 1);
                ymm9 = _mm256_add_ps(ymm9, ymm2);
                ymm9 = _mm256_hadd_ps(ymm9, ymm9);
                ymm9 = _mm256_hadd_ps(ymm9, ymm9);
                
                ymm3 = _mm256_permute2f128_ps(ymm8 , ymm8 , 1);
                ymm8 = _mm256_add_ps(ymm8, ymm3);
                ymm8 = _mm256_hadd_ps(ymm8, ymm8);
                ymm8 = _mm256_hadd_ps(ymm8, ymm8);
                
                ymm0 = _mm256_permute2f128_ps(ymm7 , ymm7 , 1);
                ymm7 = _mm256_add_ps(ymm7, ymm0);
                ymm7 = _mm256_hadd_ps(ymm7, ymm7);
                ymm7 = _mm256_hadd_ps(ymm7, ymm7);
                
                ymm1 = _mm256_permute2f128_ps(ymm6 , ymm6 , 1);
                ymm6 = _mm256_add_ps(ymm6, ymm1);
                ymm6 = _mm256_hadd_ps(ymm6, ymm6);
                ymm6 = _mm256_hadd_ps(ymm6, ymm6);
                
                ymm2 = _mm256_permute2f128_ps(ymm5 , ymm5 , 1);
                ymm5 = _mm256_add_ps(ymm5, ymm2);
                ymm5 = _mm256_hadd_ps(ymm5, ymm5);
                ymm5 = _mm256_hadd_ps(ymm5, ymm5);
                
                ymm3 = _mm256_permute2f128_ps(ymm4 , ymm4 , 1);
                ymm4 = _mm256_add_ps(ymm4, ymm3);
                ymm4 = _mm256_hadd_ps(ymm4, ymm4);
                ymm4 = _mm256_hadd_ps(ymm4, ymm4);
                
                _mm256_store_ps(sum_arr_1_1, ymm15);
                sum_1_1 += sum_arr_1_1[0];
                
                _mm256_store_ps(sum_arr_2_1, ymm14);
                sum_2_1 += sum_arr_2_1[0];
                
                if (sum_1_1 > sum_2_1) {
                    cur_result_1 = sum_1_1;
                } else {
                    cur_result_1 = sum_2_1;
                }
                
                _mm256_store_ps(sum_arr_1_2, ymm13);
                sum_1_2 += sum_arr_1_2[0];
                
                _mm256_store_ps(sum_arr_2_2, ymm12);
                sum_2_2 += sum_arr_2_2[0];
                
                if (sum_1_2 > sum_2_2) {
                    cur_result_2 = sum_1_2;
                } else {
                    cur_result_2 = sum_2_2;
                }
                
                _mm256_store_ps(sum_arr_1_3, ymm11);
                sum_1_3 += sum_arr_1_3[0];
                
                _mm256_store_ps(sum_arr_2_3, ymm10);
                sum_2_3 += sum_arr_2_3[0];
                
                if (sum_1_3 > sum_2_3) {
                    cur_result_3 = sum_1_3;
                } else {
                    cur_result_3 = sum_2_3;
                }
                
                _mm256_store_ps(sum_arr_1_4, ymm9);
                sum_1_4 += sum_arr_1_4[0];
                
                _mm256_store_ps(sum_arr_2_4, ymm8);
                sum_2_4 += sum_arr_2_4[0];
                
                if (sum_1_4 > sum_2_4) {
                    cur_result_4 = sum_1_4;
                } else {
                    cur_result_4 = sum_2_4;
                }
                
                _mm256_store_ps(sum_arr_1_5, ymm7);
                sum_1_5 += sum_arr_1_5[0];
                
                _mm256_store_ps(sum_arr_2_5, ymm6);
                sum_2_5 += sum_arr_2_5[0];
                
                if (sum_1_5 > sum_2_5) {
                    cur_result_5 = sum_1_5;
                } else {
                    cur_result_5 = sum_2_5;
                }
                
                _mm256_store_ps(sum_arr_1_6, ymm5);
                sum_1_6 += sum_arr_1_6[0];
                
                _mm256_store_ps(sum_arr_2_6, ymm4);
                sum_2_6 += sum_arr_2_6[0];
                
                if (sum_1_6 > sum_2_6) {
                    cur_result_6 = sum_1_6;
                } else {
                    cur_result_6 = sum_2_6;
                }
                
                
            //}
                        
            res_matrix[out_offset_d1 + out_offset_d2 + 0] = cur_result_1;
            res_matrix[out_offset_d1 + out_offset_d2 + 1] = cur_result_2;
            res_matrix[out_offset_d1 + out_offset_d2 + 2] = cur_result_5;
            res_matrix[out_offset_d1 + out_offset_d2 + 3] = cur_result_6;
            res_matrix[out_offset_d1 + out_offset_d2 + 4] = cur_result_3;
            res_matrix[out_offset_d1 + out_offset_d2 + 5] = cur_result_4;
            
        }
        
    }
    
    DNN_TRACE_4("finish %s\n","");
}

void conv3D_AVX_16ch_exact_2k_2X_NEW(
    const float * __restrict__ p_in_matrix, 
    int m_d1, 
    int m_d2,
    int m_d1_start,
    int m_d2_start,
    int m_d1_finish,
    int m_d2_finish,
    const float * __restrict__ p_kernel,
    const float * __restrict__ p_biases,
    int n_kernels,
    int n_maxout_kernels,
    int k_d1, 
    int k_d2,
    int n_channels,
    int n_stride,
    float * __restrict__ p_out_matrix)
{ 
    float * __restrict__ matrix = (float * __restrict__)__builtin_assume_aligned(p_in_matrix, 32);
    float * __restrict__ kernel = (float * __restrict__)__builtin_assume_aligned(p_kernel, 32);
    float * __restrict__ res_matrix = (float * __restrict__)__builtin_assume_aligned(p_out_matrix, 32);
    
	//float sum;
    float sum_arr_1_1[8];
    float sum_arr_1_2[8];
    float sum_arr_1_3[8];
    float sum_arr_1_4[8];
    float sum_arr_1_5[8];
    float sum_arr_1_6[8];
    
    float sum_arr_2_1[8];
    float sum_arr_2_2[8];
    float sum_arr_2_3[8];
    float sum_arr_2_4[8];
    float sum_arr_2_5[8];
    float sum_arr_2_6[8];
	
    DNN_TRACE_4("start %s\n","");
    DNN_TRACE_4(" p_in_matrix: %p\n", p_in_matrix);
    DNN_TRACE_4(" m_d1,m_d2: [%d,%d]\n", m_d1, m_d2);
    DNN_TRACE_4(" s_d1,s_d2: [%d,%d]\n", m_d1_start, m_d2_start);
    DNN_TRACE_4(" e_d1,e_d2: [%d,%d]\n", m_d1_finish, m_d2_finish);    
    DNN_TRACE_4(" p_kernel: %p\n", p_kernel);
    DNN_TRACE_4(" p_biases: %p\n", p_biases);
    DNN_TRACE_4(" n_kernels: %d\n", n_kernels);
    DNN_TRACE_4(" k_d1,k_d2: [%d,%d]\n", k_d1, k_d2);
    DNN_TRACE_4(" n_channels,n_stride: [%d,%d]\n", n_channels, n_stride);
    DNN_TRACE_4(" p_out_matrix: %p\n", p_out_matrix);
    
    DNN_ASSERT(n_channels == N_CHANNELS_16);
    DNN_ASSERT(k_d1 == K_D1_4);
    DNN_ASSERT(k_d2 == K_D2_4);
    DNN_ASSERT(n_kernels == 2);
    DNN_ASSERT(n_stride == 1);
    DNN_ASSERT(m_d2 >= 4);
    
    int out_d2 = m_d2 - K_D2_4 + 1;
    int out_d3 = 1;
    //int res_idx = 0;
    
    for (int cur_m_d1 = m_d1_start; (cur_m_d1 + K_D1_4) <= m_d1_finish; cur_m_d1 += 1) {
        
        int fix_offset_1 = ((cur_m_d1) + K_D1_4) - m_d1;
        if (fix_offset_1 > 0) {
            cur_m_d1 -= fix_offset_1;
            DNN_ASSERT(cur_m_d1 >= 0);
        }
        
        int out_offset_d1 = D1_TO_OFFSET(cur_m_d1, out_d2, out_d3);
        
        for (int cur_m_d2 = m_d2_start; (cur_m_d2 + K_D2_4) <= m_d2_finish; cur_m_d2 += 6) {
            
            int fix_offset_2 = ((cur_m_d2 + 5) + K_D2_4) - m_d2;
            if (fix_offset_2 > 0) {
                cur_m_d2 -= fix_offset_2;
                DNN_ASSERT(cur_m_d2 >= 0);
            }
            
            int out_offset_d2 = D2_TO_OFFSET(cur_m_d2, out_d3);
            
            //int is_first_result = 1;
            float cur_result_1 = 0;
            float cur_result_2 = 0;
            float cur_result_3 = 0;
            float cur_result_4 = 0;
            float cur_result_5 = 0;
            float cur_result_6 = 0;
            
            //for (int kernel_id = 0; kernel_id < n_kernels; kernel_id++) {
                __m256 ymm0;
                __m256 ymm1;
                __m256 ymm2;
                __m256 ymm3;
                __m256 ymm4 = _mm256_xor_ps(ymm4, ymm4); // init to 0s
                __m256 ymm5 = _mm256_xor_ps(ymm5, ymm5); // init to 0s
                __m256 ymm6 = _mm256_xor_ps(ymm6, ymm6); // init to 0s
                __m256 ymm7 = _mm256_xor_ps(ymm7, ymm7); // init to 0s
                __m256 ymm8 = _mm256_xor_ps(ymm8, ymm8); // init to 0s
                __m256 ymm9 = _mm256_xor_ps(ymm9, ymm9); // init to 0s
                __m256 ymm10 = _mm256_xor_ps(ymm10, ymm10); // init to 0s
                __m256 ymm11 = _mm256_xor_ps(ymm11, ymm11); // init to 0s
                __m256 ymm15 = _mm256_xor_ps(ymm15, ymm15); // init to 0s
                __m256 ymm14 = _mm256_xor_ps(ymm14, ymm14); // init to 0s
                __m256 ymm13 = _mm256_xor_ps(ymm13, ymm13); // init to 0s
                __m256 ymm12 = _mm256_xor_ps(ymm12, ymm12); // init to 0s

                int kernel_shift = DIMS_TO_SIZE(K_D1_4, K_D2_4, N_CHANNELS_16);
                
                int m_offset_d1 = D1_TO_OFFSET(cur_m_d1, m_d2, N_CHANNELS_16);
                
                int k_offset_d1 = 0;
                
                for (int i = 0; i < K_D1_4; i++) {

                    int m_offset_d2 = D2_TO_OFFSET(cur_m_d2, N_CHANNELS_16);

                    int k_offset_d2 = 0;

                    for (int j = 0; j < K_D2_4; j++) {
                        
                        int m_shift = m_offset_d1 + m_offset_d2;
                        int k_shift = k_offset_d1 + k_offset_d2;
                        
                        for (int k = 0; k < N_CHANNELS_16; k += 8) {
                            
                            // M1-1
                            ymm2 = _mm256_load_ps(&matrix[m_shift + k]);
                            
                            // K1
                            ymm0 = _mm256_load_ps(&kernel[k_shift + k]);
                            
                            // M1-1 * K1
                            ymm15 = _mm256_fmadd_ps(ymm0, ymm2, ymm15);
                                    
                            // K2
                            ymm1 = _mm256_load_ps(&kernel[kernel_shift + k_shift + k]);
                            
                            // M1-1 * K2
                            ymm14 = _mm256_fmadd_ps(ymm1, ymm2, ymm14);
                            
                            /////
                            // M1-3
                            ymm3 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_16 * 2 + k]);
                            
                            // M1-3 * K1
                            ymm7 = _mm256_fmadd_ps(ymm0, ymm3, ymm7);
                            
                            // M1-3 * K2
                            ymm6 = _mm256_fmadd_ps(ymm1, ymm3, ymm6);
                            
                            /////
                            // M1-2
                            ymm2 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_16 + k]);
                            
                            // M1-2 * K1
                            ymm13 = _mm256_fmadd_ps(ymm0, ymm2, ymm13);
                            
                            // M1-2 * K2
                            ymm12 = _mm256_fmadd_ps(ymm1, ymm2, ymm12);
                            
                            /////
                            // M1-4
                            ymm3 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_16 * 3 + k]);
                            
                            // M1-4 * K1
                            ymm5 = _mm256_fmadd_ps(ymm0, ymm3, ymm5);
                            
                            // M1-4 * K2
                            ymm4 = _mm256_fmadd_ps(ymm1, ymm3, ymm4);
                            
                            // M1-5
                            ymm2 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_16 * 4 + k]);
                            
                            // M1-5 * K1
                            ymm11 = _mm256_fmadd_ps(ymm0, ymm2, ymm11);
                            
                            // M1-5 * K2
                            ymm10 = _mm256_fmadd_ps(ymm1, ymm2, ymm10);
                            
                            // M1-6
                            ymm2 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_16 * 5 + k]);
                            
                            // M1-6 * K1
                            ymm9 = _mm256_fmadd_ps(ymm0, ymm2, ymm9);
                            
                            // M1-6 * K2
                            ymm8 = _mm256_fmadd_ps(ymm1, ymm2, ymm8);
                            
                        }
                        
                        k_offset_d2 += N_CHANNELS_16;
                    
                        m_offset_d2 += N_CHANNELS_16;
                    }
                    
                    m_offset_d1 += m_d2 * N_CHANNELS_16;
                    
                    k_offset_d1 += K_D2_4 * N_CHANNELS_16;
                }
                
                //ymm15 = _mm256_add_ps(ymm15, ymm14);
                //ymm13 = _mm256_add_ps(ymm13, ymm12);
                //ymm11 = _mm256_add_ps(ymm11, ymm10);
                //ymm9 = _mm256_add_ps(ymm9, ymm8);
                
                float sum_1_1 = p_biases[0];
                float sum_1_2 = p_biases[0];
                float sum_1_3 = p_biases[0];
                float sum_1_4 = p_biases[0];
                float sum_1_5 = p_biases[0];
                float sum_1_6 = p_biases[0];
                
                float sum_2_1 = p_biases[1];
                float sum_2_2 = p_biases[1];
                float sum_2_3 = p_biases[1];
                float sum_2_4 = p_biases[1];
                float sum_2_5 = p_biases[1];
                float sum_2_6 = p_biases[1];
                                    
                ymm0 = _mm256_permute2f128_ps(ymm15 , ymm15 , 1);
                ymm15 = _mm256_add_ps(ymm15, ymm0);
                ymm15 = _mm256_hadd_ps(ymm15, ymm15);
                ymm15 = _mm256_hadd_ps(ymm15, ymm15);
                
                ymm1 = _mm256_permute2f128_ps(ymm14 , ymm14 , 1);
                ymm14 = _mm256_add_ps(ymm14, ymm1);
                ymm14 = _mm256_hadd_ps(ymm14, ymm14);
                ymm14 = _mm256_hadd_ps(ymm14, ymm14);
                
                ymm2 = _mm256_permute2f128_ps(ymm13 , ymm13 , 1);
                ymm13 = _mm256_add_ps(ymm13, ymm2);
                ymm13 = _mm256_hadd_ps(ymm13, ymm13);
                ymm13 = _mm256_hadd_ps(ymm13, ymm13);
                
                ymm3 = _mm256_permute2f128_ps(ymm12 , ymm12 , 1);
                ymm12 = _mm256_add_ps(ymm12, ymm3);
                ymm12 = _mm256_hadd_ps(ymm12, ymm12);
                ymm12 = _mm256_hadd_ps(ymm12, ymm12);
                
                ymm0 = _mm256_permute2f128_ps(ymm11 , ymm11 , 1);
                ymm11 = _mm256_add_ps(ymm11, ymm0);
                ymm11 = _mm256_hadd_ps(ymm11, ymm11);
                ymm11 = _mm256_hadd_ps(ymm11, ymm11);
                
                ymm1 = _mm256_permute2f128_ps(ymm10 , ymm10 , 1);
                ymm10 = _mm256_add_ps(ymm10, ymm1);
                ymm10 = _mm256_hadd_ps(ymm10, ymm10);
                ymm10 = _mm256_hadd_ps(ymm10, ymm10);
                
                ymm2 = _mm256_permute2f128_ps(ymm9 , ymm9 , 1);
                ymm9 = _mm256_add_ps(ymm9, ymm2);
                ymm9 = _mm256_hadd_ps(ymm9, ymm9);
                ymm9 = _mm256_hadd_ps(ymm9, ymm9);
                
                ymm3 = _mm256_permute2f128_ps(ymm8 , ymm8 , 1);
                ymm8 = _mm256_add_ps(ymm8, ymm3);
                ymm8 = _mm256_hadd_ps(ymm8, ymm8);
                ymm8 = _mm256_hadd_ps(ymm8, ymm8);
                
                ymm0 = _mm256_permute2f128_ps(ymm7 , ymm7 , 1);
                ymm7 = _mm256_add_ps(ymm7, ymm0);
                ymm7 = _mm256_hadd_ps(ymm7, ymm7);
                ymm7 = _mm256_hadd_ps(ymm7, ymm7);
                
                ymm1 = _mm256_permute2f128_ps(ymm6 , ymm6 , 1);
                ymm6 = _mm256_add_ps(ymm6, ymm1);
                ymm6 = _mm256_hadd_ps(ymm6, ymm6);
                ymm6 = _mm256_hadd_ps(ymm6, ymm6);
                
                ymm2 = _mm256_permute2f128_ps(ymm5 , ymm5 , 1);
                ymm5 = _mm256_add_ps(ymm5, ymm2);
                ymm5 = _mm256_hadd_ps(ymm5, ymm5);
                ymm5 = _mm256_hadd_ps(ymm5, ymm5);
                
                ymm3 = _mm256_permute2f128_ps(ymm4 , ymm4 , 1);
                ymm4 = _mm256_add_ps(ymm4, ymm3);
                ymm4 = _mm256_hadd_ps(ymm4, ymm4);
                ymm4 = _mm256_hadd_ps(ymm4, ymm4);
                
                _mm256_store_ps(sum_arr_1_1, ymm15);
                sum_1_1 += sum_arr_1_1[0];
                
                _mm256_store_ps(sum_arr_2_1, ymm14);
                sum_2_1 += sum_arr_2_1[0];
                
                if (sum_1_1 > sum_2_1) {
                    cur_result_1 = sum_1_1;
                } else {
                    cur_result_1 = sum_2_1;
                }
                
                _mm256_store_ps(sum_arr_1_2, ymm13);
                sum_1_2 += sum_arr_1_2[0];
                
                _mm256_store_ps(sum_arr_2_2, ymm12);
                sum_2_2 += sum_arr_2_2[0];
                
                if (sum_1_2 > sum_2_2) {
                    cur_result_2 = sum_1_2;
                } else {
                    cur_result_2 = sum_2_2;
                }
                
                _mm256_store_ps(sum_arr_1_3, ymm11);
                sum_1_3 += sum_arr_1_3[0];
                
                _mm256_store_ps(sum_arr_2_3, ymm10);
                sum_2_3 += sum_arr_2_3[0];
                
                if (sum_1_3 > sum_2_3) {
                    cur_result_3 = sum_1_3;
                } else {
                    cur_result_3 = sum_2_3;
                }
                
                _mm256_store_ps(sum_arr_1_4, ymm9);
                sum_1_4 += sum_arr_1_4[0];
                
                _mm256_store_ps(sum_arr_2_4, ymm8);
                sum_2_4 += sum_arr_2_4[0];
                
                if (sum_1_4 > sum_2_4) {
                    cur_result_4 = sum_1_4;
                } else {
                    cur_result_4 = sum_2_4;
                }
                
                _mm256_store_ps(sum_arr_1_5, ymm7);
                sum_1_5 += sum_arr_1_5[0];
                
                _mm256_store_ps(sum_arr_2_5, ymm6);
                sum_2_5 += sum_arr_2_5[0];
                
                if (sum_1_5 > sum_2_5) {
                    cur_result_5 = sum_1_5;
                } else {
                    cur_result_5 = sum_2_5;
                }
                
                _mm256_store_ps(sum_arr_1_6, ymm5);
                sum_1_6 += sum_arr_1_6[0];
                
                _mm256_store_ps(sum_arr_2_6, ymm4);
                sum_2_6 += sum_arr_2_6[0];
                
                if (sum_1_6 > sum_2_6) {
                    cur_result_6 = sum_1_6;
                } else {
                    cur_result_6 = sum_2_6;
                }
                
                
            //}
                        
            res_matrix[out_offset_d1 + out_offset_d2 + 0] = cur_result_1;
            res_matrix[out_offset_d1 + out_offset_d2 + 1] = cur_result_2;
            res_matrix[out_offset_d1 + out_offset_d2 + 2] = cur_result_5;
            res_matrix[out_offset_d1 + out_offset_d2 + 3] = cur_result_6;
            res_matrix[out_offset_d1 + out_offset_d2 + 4] = cur_result_3;
            res_matrix[out_offset_d1 + out_offset_d2 + 5] = cur_result_4;
            
        }
        
    }
    
    DNN_TRACE_4("finish %s\n","");
}

void conv3D_AVX_16ch_exact_1k_2X_NEW(
    const float * __restrict__ p_in_matrix, 
    int m_d1, 
    int m_d2,
    int m_d1_start,
    int m_d2_start,
    int m_d1_finish,
    int m_d2_finish,
    const float * __restrict__ p_kernel,
    const float * __restrict__ p_biases,
    int n_kernels,
    int n_maxout_kernels,
    int k_d1, 
    int k_d2,
    int n_channels,
    int n_stride,
    float * __restrict__ p_out_matrix)
{ 
    float * __restrict__ matrix = (float * __restrict__)__builtin_assume_aligned(p_in_matrix, 32);
    float * __restrict__ kernel = (float * __restrict__)__builtin_assume_aligned(p_kernel, 32);
    float * __restrict__ res_matrix = (float * __restrict__)__builtin_assume_aligned(p_out_matrix, 32);
    
	//float sum;
    float sum_arr_1_1[8];
    float sum_arr_1_2[8];
    float sum_arr_1_3[8];
    float sum_arr_1_4[8];
    float sum_arr_1_5[8];
    float sum_arr_1_6[8];
    
    float sum_arr_2_1[8];
    float sum_arr_2_2[8];
    float sum_arr_2_3[8];
    float sum_arr_2_4[8];
    float sum_arr_2_5[8];
    float sum_arr_2_6[8];
	
    DNN_TRACE_4("start %s\n","");
    DNN_TRACE_4(" p_in_matrix: %p\n", p_in_matrix);
    DNN_TRACE_4(" m_d1,m_d2: [%d,%d]\n", m_d1, m_d2);
    DNN_TRACE_4(" s_d1,s_d2: [%d,%d]\n", m_d1_start, m_d2_start);
    DNN_TRACE_4(" e_d1,e_d2: [%d,%d]\n", m_d1_finish, m_d2_finish);    
    DNN_TRACE_4(" p_kernel: %p\n", p_kernel);
    DNN_TRACE_4(" p_biases: %p\n", p_biases);
    DNN_TRACE_4(" n_kernels: %d\n", n_kernels);
    DNN_TRACE_4(" k_d1,k_d2: [%d,%d]\n", k_d1, k_d2);
    DNN_TRACE_4(" n_channels,n_stride: [%d,%d]\n", n_channels, n_stride);
    DNN_TRACE_4(" p_out_matrix: %p\n", p_out_matrix);
    
    DNN_ASSERT(n_channels == N_CHANNELS_16);
    DNN_ASSERT(k_d1 == K_D1_1);
    DNN_ASSERT(k_d2 == K_D2_1);
    DNN_ASSERT(n_kernels == 2);
    DNN_ASSERT(n_stride == 1);
    DNN_ASSERT(m_d2 >= 4);
    
    int out_d2 = m_d2 - K_D2_1 + 1;
    int out_d3 = 1;
    //int res_idx = 0;
    
    for (int cur_m_d1 = m_d1_start; (cur_m_d1 + K_D1_1) <= m_d1_finish; cur_m_d1 += 1) {
        
        int fix_offset_1 = ((cur_m_d1) + K_D1_1) - m_d1;
        if (fix_offset_1 > 0) {
            cur_m_d1 -= fix_offset_1;
            DNN_ASSERT(cur_m_d1 >= 0);
        }
        
        int out_offset_d1 = D1_TO_OFFSET(cur_m_d1, out_d2, out_d3);
        
        for (int cur_m_d2 = m_d2_start; (cur_m_d2 + K_D2_1) <= m_d2_finish; cur_m_d2 += 6) {
            
            int fix_offset_2 = ((cur_m_d2 + 5) + K_D2_1) - m_d2;
            if (fix_offset_2 > 0) {
                cur_m_d2 -= fix_offset_2;
                DNN_ASSERT(cur_m_d2 >= 0);
            }
            
            int out_offset_d2 = D2_TO_OFFSET(cur_m_d2, out_d3);
            
            //int is_first_result = 1;
            float cur_result_1 = 0;
            float cur_result_2 = 0;
            float cur_result_3 = 0;
            float cur_result_4 = 0;
            float cur_result_5 = 0;
            float cur_result_6 = 0;
            
            //for (int kernel_id = 0; kernel_id < n_kernels; kernel_id++) {
                __m256 ymm0;
                __m256 ymm1;
                __m256 ymm2;
                __m256 ymm3;
                __m256 ymm4 = _mm256_xor_ps(ymm4, ymm4); // init to 0s
                __m256 ymm5 = _mm256_xor_ps(ymm5, ymm5); // init to 0s
                __m256 ymm6 = _mm256_xor_ps(ymm6, ymm6); // init to 0s
                __m256 ymm7 = _mm256_xor_ps(ymm7, ymm7); // init to 0s
                __m256 ymm8 = _mm256_xor_ps(ymm8, ymm8); // init to 0s
                __m256 ymm9 = _mm256_xor_ps(ymm9, ymm9); // init to 0s
                __m256 ymm10 = _mm256_xor_ps(ymm10, ymm10); // init to 0s
                __m256 ymm11 = _mm256_xor_ps(ymm11, ymm11); // init to 0s
                __m256 ymm15 = _mm256_xor_ps(ymm15, ymm15); // init to 0s
                __m256 ymm14 = _mm256_xor_ps(ymm14, ymm14); // init to 0s
                __m256 ymm13 = _mm256_xor_ps(ymm13, ymm13); // init to 0s
                __m256 ymm12 = _mm256_xor_ps(ymm12, ymm12); // init to 0s

                int kernel_shift = DIMS_TO_SIZE(K_D1_1, K_D2_1, N_CHANNELS_16);
                
                int m_offset_d1 = D1_TO_OFFSET(cur_m_d1, m_d2, N_CHANNELS_16);
                
                int k_offset_d1 = 0;
                
                for (int i = 0; i < K_D1_1; i++) {

                    int m_offset_d2 = D2_TO_OFFSET(cur_m_d2, N_CHANNELS_16);

                    int k_offset_d2 = 0;

                    for (int j = 0; j < K_D2_1; j++) {
                        
                        int m_shift = m_offset_d1 + m_offset_d2;
                        int k_shift = k_offset_d1 + k_offset_d2;
                        
                        for (int k = 0; k < N_CHANNELS_16; k += 8) {
                            
                            // M1-1
                            ymm2 = _mm256_load_ps(&matrix[m_shift + k]);
                            
                            // K1
                            ymm0 = _mm256_load_ps(&kernel[k_shift + k]);
                            
                            // M1-1 * K1
                            ymm15 = _mm256_fmadd_ps(ymm0, ymm2, ymm15);
                                    
                            // K2
                            ymm1 = _mm256_load_ps(&kernel[kernel_shift + k_shift + k]);
                            
                            // M1-1 * K2
                            ymm14 = _mm256_fmadd_ps(ymm1, ymm2, ymm14);
                            
                            /////
                            // M1-3
                            ymm3 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_16 * 2 + k]);
                            
                            // M1-3 * K1
                            ymm7 = _mm256_fmadd_ps(ymm0, ymm3, ymm7);
                            
                            // M1-3 * K2
                            ymm6 = _mm256_fmadd_ps(ymm1, ymm3, ymm6);
                            
                            /////
                            // M1-2
                            ymm2 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_16 + k]);
                            
                            // M1-2 * K1
                            ymm13 = _mm256_fmadd_ps(ymm0, ymm2, ymm13);
                            
                            // M1-2 * K2
                            ymm12 = _mm256_fmadd_ps(ymm1, ymm2, ymm12);
                            
                            /////
                            // M1-4
                            ymm3 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_16 * 3 + k]);
                            
                            // M1-4 * K1
                            ymm5 = _mm256_fmadd_ps(ymm0, ymm3, ymm5);
                            
                            // M1-4 * K2
                            ymm4 = _mm256_fmadd_ps(ymm1, ymm3, ymm4);
                            
                            // M1-5
                            ymm2 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_16 * 4 + k]);
                            
                            // M1-5 * K1
                            ymm11 = _mm256_fmadd_ps(ymm0, ymm2, ymm11);
                            
                            // M1-5 * K2
                            ymm10 = _mm256_fmadd_ps(ymm1, ymm2, ymm10);
                            
                            // M1-6
                            ymm2 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_16 * 5 + k]);
                            
                            // M1-6 * K1
                            ymm9 = _mm256_fmadd_ps(ymm0, ymm2, ymm9);
                            
                            // M1-6 * K2
                            ymm8 = _mm256_fmadd_ps(ymm1, ymm2, ymm8);
                            
                        }
                        
                        k_offset_d2 += N_CHANNELS_16;
                    
                        m_offset_d2 += N_CHANNELS_16;
                    }
                    
                    m_offset_d1 += m_d2 * N_CHANNELS_16;
                    
                    k_offset_d1 += K_D2_1 * N_CHANNELS_16;
                }
                
                //ymm15 = _mm256_add_ps(ymm15, ymm14);
                //ymm13 = _mm256_add_ps(ymm13, ymm12);
                //ymm11 = _mm256_add_ps(ymm11, ymm10);
                //ymm9 = _mm256_add_ps(ymm9, ymm8);
                
                float sum_1_1 = p_biases[0];
                float sum_1_2 = p_biases[0];
                float sum_1_3 = p_biases[0];
                float sum_1_4 = p_biases[0];
                float sum_1_5 = p_biases[0];
                float sum_1_6 = p_biases[0];
                
                float sum_2_1 = p_biases[1];
                float sum_2_2 = p_biases[1];
                float sum_2_3 = p_biases[1];
                float sum_2_4 = p_biases[1];
                float sum_2_5 = p_biases[1];
                float sum_2_6 = p_biases[1];
                                    
                ymm0 = _mm256_permute2f128_ps(ymm15 , ymm15 , 1);
                ymm15 = _mm256_add_ps(ymm15, ymm0);
                ymm15 = _mm256_hadd_ps(ymm15, ymm15);
                ymm15 = _mm256_hadd_ps(ymm15, ymm15);
                
                ymm1 = _mm256_permute2f128_ps(ymm14 , ymm14 , 1);
                ymm14 = _mm256_add_ps(ymm14, ymm1);
                ymm14 = _mm256_hadd_ps(ymm14, ymm14);
                ymm14 = _mm256_hadd_ps(ymm14, ymm14);
                
                ymm2 = _mm256_permute2f128_ps(ymm13 , ymm13 , 1);
                ymm13 = _mm256_add_ps(ymm13, ymm2);
                ymm13 = _mm256_hadd_ps(ymm13, ymm13);
                ymm13 = _mm256_hadd_ps(ymm13, ymm13);
                
                ymm3 = _mm256_permute2f128_ps(ymm12 , ymm12 , 1);
                ymm12 = _mm256_add_ps(ymm12, ymm3);
                ymm12 = _mm256_hadd_ps(ymm12, ymm12);
                ymm12 = _mm256_hadd_ps(ymm12, ymm12);
                
                ymm0 = _mm256_permute2f128_ps(ymm11 , ymm11 , 1);
                ymm11 = _mm256_add_ps(ymm11, ymm0);
                ymm11 = _mm256_hadd_ps(ymm11, ymm11);
                ymm11 = _mm256_hadd_ps(ymm11, ymm11);
                
                ymm1 = _mm256_permute2f128_ps(ymm10 , ymm10 , 1);
                ymm10 = _mm256_add_ps(ymm10, ymm1);
                ymm10 = _mm256_hadd_ps(ymm10, ymm10);
                ymm10 = _mm256_hadd_ps(ymm10, ymm10);
                
                ymm2 = _mm256_permute2f128_ps(ymm9 , ymm9 , 1);
                ymm9 = _mm256_add_ps(ymm9, ymm2);
                ymm9 = _mm256_hadd_ps(ymm9, ymm9);
                ymm9 = _mm256_hadd_ps(ymm9, ymm9);
                
                ymm3 = _mm256_permute2f128_ps(ymm8 , ymm8 , 1);
                ymm8 = _mm256_add_ps(ymm8, ymm3);
                ymm8 = _mm256_hadd_ps(ymm8, ymm8);
                ymm8 = _mm256_hadd_ps(ymm8, ymm8);
                
                ymm0 = _mm256_permute2f128_ps(ymm7 , ymm7 , 1);
                ymm7 = _mm256_add_ps(ymm7, ymm0);
                ymm7 = _mm256_hadd_ps(ymm7, ymm7);
                ymm7 = _mm256_hadd_ps(ymm7, ymm7);
                
                ymm1 = _mm256_permute2f128_ps(ymm6 , ymm6 , 1);
                ymm6 = _mm256_add_ps(ymm6, ymm1);
                ymm6 = _mm256_hadd_ps(ymm6, ymm6);
                ymm6 = _mm256_hadd_ps(ymm6, ymm6);
                
                ymm2 = _mm256_permute2f128_ps(ymm5 , ymm5 , 1);
                ymm5 = _mm256_add_ps(ymm5, ymm2);
                ymm5 = _mm256_hadd_ps(ymm5, ymm5);
                ymm5 = _mm256_hadd_ps(ymm5, ymm5);
                
                ymm3 = _mm256_permute2f128_ps(ymm4 , ymm4 , 1);
                ymm4 = _mm256_add_ps(ymm4, ymm3);
                ymm4 = _mm256_hadd_ps(ymm4, ymm4);
                ymm4 = _mm256_hadd_ps(ymm4, ymm4);
                
                _mm256_store_ps(sum_arr_1_1, ymm15);
                sum_1_1 += sum_arr_1_1[0];
                
                _mm256_store_ps(sum_arr_2_1, ymm14);
                sum_2_1 += sum_arr_2_1[0];
                
                if (sum_1_1 > sum_2_1) {
                    cur_result_1 = sum_1_1;
                } else {
                    cur_result_1 = sum_2_1;
                }
                
                _mm256_store_ps(sum_arr_1_2, ymm13);
                sum_1_2 += sum_arr_1_2[0];
                
                _mm256_store_ps(sum_arr_2_2, ymm12);
                sum_2_2 += sum_arr_2_2[0];
                
                if (sum_1_2 > sum_2_2) {
                    cur_result_2 = sum_1_2;
                } else {
                    cur_result_2 = sum_2_2;
                }
                
                _mm256_store_ps(sum_arr_1_3, ymm11);
                sum_1_3 += sum_arr_1_3[0];
                
                _mm256_store_ps(sum_arr_2_3, ymm10);
                sum_2_3 += sum_arr_2_3[0];
                
                if (sum_1_3 > sum_2_3) {
                    cur_result_3 = sum_1_3;
                } else {
                    cur_result_3 = sum_2_3;
                }
                
                _mm256_store_ps(sum_arr_1_4, ymm9);
                sum_1_4 += sum_arr_1_4[0];
                
                _mm256_store_ps(sum_arr_2_4, ymm8);
                sum_2_4 += sum_arr_2_4[0];
                
                if (sum_1_4 > sum_2_4) {
                    cur_result_4 = sum_1_4;
                } else {
                    cur_result_4 = sum_2_4;
                }
                
                _mm256_store_ps(sum_arr_1_5, ymm7);
                sum_1_5 += sum_arr_1_5[0];
                
                _mm256_store_ps(sum_arr_2_5, ymm6);
                sum_2_5 += sum_arr_2_5[0];
                
                if (sum_1_5 > sum_2_5) {
                    cur_result_5 = sum_1_5;
                } else {
                    cur_result_5 = sum_2_5;
                }
                
                _mm256_store_ps(sum_arr_1_6, ymm5);
                sum_1_6 += sum_arr_1_6[0];
                
                _mm256_store_ps(sum_arr_2_6, ymm4);
                sum_2_6 += sum_arr_2_6[0];
                
                if (sum_1_6 > sum_2_6) {
                    cur_result_6 = sum_1_6;
                } else {
                    cur_result_6 = sum_2_6;
                }
                
                
            //}
                        
            res_matrix[out_offset_d1 + out_offset_d2 + 0] = cur_result_1;
            res_matrix[out_offset_d1 + out_offset_d2 + 1] = cur_result_2;
            res_matrix[out_offset_d1 + out_offset_d2 + 2] = cur_result_5;
            res_matrix[out_offset_d1 + out_offset_d2 + 3] = cur_result_6;
            res_matrix[out_offset_d1 + out_offset_d2 + 4] = cur_result_3;
            res_matrix[out_offset_d1 + out_offset_d2 + 5] = cur_result_4;
            
        }
        
    }
    
    DNN_TRACE_4("finish %s\n","");
}

void conv3D_AVX_48ch_exact_1k_2X_NEW(
    const float * __restrict__ p_in_matrix, 
    int m_d1, 
    int m_d2,
    int m_d1_start,
    int m_d2_start,
    int m_d1_finish,
    int m_d2_finish,
    const float * __restrict__ p_kernel,
    const float * __restrict__ p_biases,
    int n_kernels,
    int n_maxout_kernels,
    int k_d1, 
    int k_d2,
    int n_channels,
    int n_stride,
    float * __restrict__ p_out_matrix)
{ 
    float * __restrict__ matrix = (float * __restrict__)__builtin_assume_aligned(p_in_matrix, 32);
    float * __restrict__ kernel = (float * __restrict__)__builtin_assume_aligned(p_kernel, 32);
    float * __restrict__ res_matrix = (float * __restrict__)__builtin_assume_aligned(p_out_matrix, 32);
    
	//float sum;
    float sum_arr_1_1[8];
    float sum_arr_1_2[8];
    float sum_arr_1_3[8];
    float sum_arr_1_4[8];
    float sum_arr_1_5[8];
    float sum_arr_1_6[8];
    
    float sum_arr_2_1[8];
    float sum_arr_2_2[8];
    float sum_arr_2_3[8];
    float sum_arr_2_4[8];
    float sum_arr_2_5[8];
    float sum_arr_2_6[8];
	
    DNN_TRACE_4("start %s\n","");
    DNN_TRACE_4(" p_in_matrix: %p\n", p_in_matrix);
    DNN_TRACE_4(" m_d1,m_d2: [%d,%d]\n", m_d1, m_d2);
    DNN_TRACE_4(" s_d1,s_d2: [%d,%d]\n", m_d1_start, m_d2_start);
    DNN_TRACE_4(" e_d1,e_d2: [%d,%d]\n", m_d1_finish, m_d2_finish);    
    DNN_TRACE_4(" p_kernel: %p\n", p_kernel);
    DNN_TRACE_4(" p_biases: %p\n", p_biases);
    DNN_TRACE_4(" n_kernels: %d\n", n_kernels);
    DNN_TRACE_4(" k_d1,k_d2: [%d,%d]\n", k_d1, k_d2);
    DNN_TRACE_4(" n_channels,n_stride: [%d,%d]\n", n_channels, n_stride);
    DNN_TRACE_4(" p_out_matrix: %p\n", p_out_matrix);
    
    DNN_ASSERT(n_channels == N_CHANNELS_48);
    DNN_ASSERT(k_d1 == K_D1_1);
    DNN_ASSERT(k_d2 == K_D2_1);
    DNN_ASSERT(n_kernels == 2);
    DNN_ASSERT(n_stride == 1);
    DNN_ASSERT(m_d2 >= 4);
    
    int out_d2 = m_d2 - K_D2_1 + 1;
    int out_d3 = 1;
    //int res_idx = 0;
    
    for (int cur_m_d1 = m_d1_start; (cur_m_d1 + K_D1_1) <= m_d1_finish; cur_m_d1 += 1) {
        
        int fix_offset_1 = ((cur_m_d1) + K_D1_1) - m_d1;
        if (fix_offset_1 > 0) {
            cur_m_d1 -= fix_offset_1;
            DNN_ASSERT(cur_m_d1 >= 0);
        }
        
        int out_offset_d1 = D1_TO_OFFSET(cur_m_d1, out_d2, out_d3);
        
        for (int cur_m_d2 = m_d2_start; (cur_m_d2 + K_D2_1) <= m_d2_finish; cur_m_d2 += 6) {
            
            int fix_offset_2 = ((cur_m_d2 + 5) + K_D2_1) - m_d2;
            if (fix_offset_2 > 0) {
                cur_m_d2 -= fix_offset_2;
                DNN_ASSERT(cur_m_d2 >= 0);
            }
            
            int out_offset_d2 = D2_TO_OFFSET(cur_m_d2, out_d3);
            
            //int is_first_result = 1;
            float cur_result_1 = 0;
            float cur_result_2 = 0;
            float cur_result_3 = 0;
            float cur_result_4 = 0;
            float cur_result_5 = 0;
            float cur_result_6 = 0;
            
            //for (int kernel_id = 0; kernel_id < n_kernels; kernel_id++) {
                __m256 ymm0;
                __m256 ymm1;
                __m256 ymm2;
                __m256 ymm3;
                __m256 ymm4 = _mm256_xor_ps(ymm4, ymm4); // init to 0s
                __m256 ymm5 = _mm256_xor_ps(ymm5, ymm5); // init to 0s
                __m256 ymm6 = _mm256_xor_ps(ymm6, ymm6); // init to 0s
                __m256 ymm7 = _mm256_xor_ps(ymm7, ymm7); // init to 0s
                __m256 ymm8 = _mm256_xor_ps(ymm8, ymm8); // init to 0s
                __m256 ymm9 = _mm256_xor_ps(ymm9, ymm9); // init to 0s
                __m256 ymm10 = _mm256_xor_ps(ymm10, ymm10); // init to 0s
                __m256 ymm11 = _mm256_xor_ps(ymm11, ymm11); // init to 0s
                __m256 ymm15 = _mm256_xor_ps(ymm15, ymm15); // init to 0s
                __m256 ymm14 = _mm256_xor_ps(ymm14, ymm14); // init to 0s
                __m256 ymm13 = _mm256_xor_ps(ymm13, ymm13); // init to 0s
                __m256 ymm12 = _mm256_xor_ps(ymm12, ymm12); // init to 0s

                int kernel_shift = DIMS_TO_SIZE(K_D1_1, K_D2_1, N_CHANNELS_48);
                
                int m_offset_d1 = D1_TO_OFFSET(cur_m_d1, m_d2, N_CHANNELS_48);
                
                int k_offset_d1 = 0;
                
                for (int i = 0; i < K_D1_1; i++) {

                    int m_offset_d2 = D2_TO_OFFSET(cur_m_d2, N_CHANNELS_48);

                    int k_offset_d2 = 0;

                    for (int j = 0; j < K_D2_1; j++) {
                        
                        int m_shift = m_offset_d1 + m_offset_d2;
                        int k_shift = k_offset_d1 + k_offset_d2;
                        
                        for (int k = 0; k < N_CHANNELS_48; k += 8) {
                            
                            // M1-1
                            ymm2 = _mm256_load_ps(&matrix[m_shift + k]);
                            
                            // K1
                            ymm0 = _mm256_load_ps(&kernel[k_shift + k]);
                            
                            // M1-1 * K1
                            ymm15 = _mm256_fmadd_ps(ymm0, ymm2, ymm15);
                                    
                            // K2
                            ymm1 = _mm256_load_ps(&kernel[kernel_shift + k_shift + k]);
                            
                            // M1-1 * K2
                            ymm14 = _mm256_fmadd_ps(ymm1, ymm2, ymm14);
                            
                            /////
                            // M1-3
                            ymm3 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_48 * 2 + k]);
                            
                            // M1-3 * K1
                            ymm7 = _mm256_fmadd_ps(ymm0, ymm3, ymm7);
                            
                            // M1-3 * K2
                            ymm6 = _mm256_fmadd_ps(ymm1, ymm3, ymm6);
                            
                            /////
                            // M1-2
                            ymm2 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_48 + k]);
                            
                            // M1-2 * K1
                            ymm13 = _mm256_fmadd_ps(ymm0, ymm2, ymm13);
                            
                            // M1-2 * K2
                            ymm12 = _mm256_fmadd_ps(ymm1, ymm2, ymm12);
                            
                            /////
                            // M1-4
                            ymm3 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_48 * 3 + k]);
                            
                            // M1-4 * K1
                            ymm5 = _mm256_fmadd_ps(ymm0, ymm3, ymm5);
                            
                            // M1-4 * K2
                            ymm4 = _mm256_fmadd_ps(ymm1, ymm3, ymm4);
                            
                            // M1-5
                            ymm2 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_48 * 4 + k]);
                            
                            // M1-5 * K1
                            ymm11 = _mm256_fmadd_ps(ymm0, ymm2, ymm11);
                            
                            // M1-5 * K2
                            ymm10 = _mm256_fmadd_ps(ymm1, ymm2, ymm10);
                            
                            // M1-6
                            ymm2 = _mm256_load_ps(&matrix[m_shift + N_CHANNELS_48 * 5 + k]);
                            
                            // M1-6 * K1
                            ymm9 = _mm256_fmadd_ps(ymm0, ymm2, ymm9);
                            
                            // M1-6 * K2
                            ymm8 = _mm256_fmadd_ps(ymm1, ymm2, ymm8);
                            
                        }
                        
                        k_offset_d2 += N_CHANNELS_48;
                    
                        m_offset_d2 += N_CHANNELS_48;
                    }
                    
                    m_offset_d1 += m_d2 * N_CHANNELS_48;
                    
                    k_offset_d1 += K_D2_1 * N_CHANNELS_48;
                }
                
                //ymm15 = _mm256_add_ps(ymm15, ymm14);
                //ymm13 = _mm256_add_ps(ymm13, ymm12);
                //ymm11 = _mm256_add_ps(ymm11, ymm10);
                //ymm9 = _mm256_add_ps(ymm9, ymm8);
                
                float sum_1_1 = p_biases[0];
                float sum_1_2 = p_biases[0];
                float sum_1_3 = p_biases[0];
                float sum_1_4 = p_biases[0];
                float sum_1_5 = p_biases[0];
                float sum_1_6 = p_biases[0];
                
                float sum_2_1 = p_biases[1];
                float sum_2_2 = p_biases[1];
                float sum_2_3 = p_biases[1];
                float sum_2_4 = p_biases[1];
                float sum_2_5 = p_biases[1];
                float sum_2_6 = p_biases[1];
                                    
                ymm0 = _mm256_permute2f128_ps(ymm15 , ymm15 , 1);
                ymm15 = _mm256_add_ps(ymm15, ymm0);
                ymm15 = _mm256_hadd_ps(ymm15, ymm15);
                ymm15 = _mm256_hadd_ps(ymm15, ymm15);
                
                ymm1 = _mm256_permute2f128_ps(ymm14 , ymm14 , 1);
                ymm14 = _mm256_add_ps(ymm14, ymm1);
                ymm14 = _mm256_hadd_ps(ymm14, ymm14);
                ymm14 = _mm256_hadd_ps(ymm14, ymm14);
                
                ymm2 = _mm256_permute2f128_ps(ymm13 , ymm13 , 1);
                ymm13 = _mm256_add_ps(ymm13, ymm2);
                ymm13 = _mm256_hadd_ps(ymm13, ymm13);
                ymm13 = _mm256_hadd_ps(ymm13, ymm13);
                
                ymm3 = _mm256_permute2f128_ps(ymm12 , ymm12 , 1);
                ymm12 = _mm256_add_ps(ymm12, ymm3);
                ymm12 = _mm256_hadd_ps(ymm12, ymm12);
                ymm12 = _mm256_hadd_ps(ymm12, ymm12);
                
                ymm0 = _mm256_permute2f128_ps(ymm11 , ymm11 , 1);
                ymm11 = _mm256_add_ps(ymm11, ymm0);
                ymm11 = _mm256_hadd_ps(ymm11, ymm11);
                ymm11 = _mm256_hadd_ps(ymm11, ymm11);
                
                ymm1 = _mm256_permute2f128_ps(ymm10 , ymm10 , 1);
                ymm10 = _mm256_add_ps(ymm10, ymm1);
                ymm10 = _mm256_hadd_ps(ymm10, ymm10);
                ymm10 = _mm256_hadd_ps(ymm10, ymm10);
                
                ymm2 = _mm256_permute2f128_ps(ymm9 , ymm9 , 1);
                ymm9 = _mm256_add_ps(ymm9, ymm2);
                ymm9 = _mm256_hadd_ps(ymm9, ymm9);
                ymm9 = _mm256_hadd_ps(ymm9, ymm9);
                
                ymm3 = _mm256_permute2f128_ps(ymm8 , ymm8 , 1);
                ymm8 = _mm256_add_ps(ymm8, ymm3);
                ymm8 = _mm256_hadd_ps(ymm8, ymm8);
                ymm8 = _mm256_hadd_ps(ymm8, ymm8);
                
                ymm0 = _mm256_permute2f128_ps(ymm7 , ymm7 , 1);
                ymm7 = _mm256_add_ps(ymm7, ymm0);
                ymm7 = _mm256_hadd_ps(ymm7, ymm7);
                ymm7 = _mm256_hadd_ps(ymm7, ymm7);
                
                ymm1 = _mm256_permute2f128_ps(ymm6 , ymm6 , 1);
                ymm6 = _mm256_add_ps(ymm6, ymm1);
                ymm6 = _mm256_hadd_ps(ymm6, ymm6);
                ymm6 = _mm256_hadd_ps(ymm6, ymm6);
                
                ymm2 = _mm256_permute2f128_ps(ymm5 , ymm5 , 1);
                ymm5 = _mm256_add_ps(ymm5, ymm2);
                ymm5 = _mm256_hadd_ps(ymm5, ymm5);
                ymm5 = _mm256_hadd_ps(ymm5, ymm5);
                
                ymm3 = _mm256_permute2f128_ps(ymm4 , ymm4 , 1);
                ymm4 = _mm256_add_ps(ymm4, ymm3);
                ymm4 = _mm256_hadd_ps(ymm4, ymm4);
                ymm4 = _mm256_hadd_ps(ymm4, ymm4);
                
                _mm256_store_ps(sum_arr_1_1, ymm15);
                sum_1_1 += sum_arr_1_1[0];
                
                _mm256_store_ps(sum_arr_2_1, ymm14);
                sum_2_1 += sum_arr_2_1[0];
                
                if (sum_1_1 > sum_2_1) {
                    cur_result_1 = sum_1_1;
                } else {
                    cur_result_1 = sum_2_1;
                }
                
                _mm256_store_ps(sum_arr_1_2, ymm13);
                sum_1_2 += sum_arr_1_2[0];
                
                _mm256_store_ps(sum_arr_2_2, ymm12);
                sum_2_2 += sum_arr_2_2[0];
                
                if (sum_1_2 > sum_2_2) {
                    cur_result_2 = sum_1_2;
                } else {
                    cur_result_2 = sum_2_2;
                }
                
                _mm256_store_ps(sum_arr_1_3, ymm11);
                sum_1_3 += sum_arr_1_3[0];
                
                _mm256_store_ps(sum_arr_2_3, ymm10);
                sum_2_3 += sum_arr_2_3[0];
                
                if (sum_1_3 > sum_2_3) {
                    cur_result_3 = sum_1_3;
                } else {
                    cur_result_3 = sum_2_3;
                }
                
                _mm256_store_ps(sum_arr_1_4, ymm9);
                sum_1_4 += sum_arr_1_4[0];
                
                _mm256_store_ps(sum_arr_2_4, ymm8);
                sum_2_4 += sum_arr_2_4[0];
                
                if (sum_1_4 > sum_2_4) {
                    cur_result_4 = sum_1_4;
                } else {
                    cur_result_4 = sum_2_4;
                }
                
                _mm256_store_ps(sum_arr_1_5, ymm7);
                sum_1_5 += sum_arr_1_5[0];
                
                _mm256_store_ps(sum_arr_2_5, ymm6);
                sum_2_5 += sum_arr_2_5[0];
                
                if (sum_1_5 > sum_2_5) {
                    cur_result_5 = sum_1_5;
                } else {
                    cur_result_5 = sum_2_5;
                }
                
                _mm256_store_ps(sum_arr_1_6, ymm5);
                sum_1_6 += sum_arr_1_6[0];
                
                _mm256_store_ps(sum_arr_2_6, ymm4);
                sum_2_6 += sum_arr_2_6[0];
                
                if (sum_1_6 > sum_2_6) {
                    cur_result_6 = sum_1_6;
                } else {
                    cur_result_6 = sum_2_6;
                }
                
                
            //}
                        
            res_matrix[out_offset_d1 + out_offset_d2 + 0] = cur_result_1;
            res_matrix[out_offset_d1 + out_offset_d2 + 1] = cur_result_2;
            res_matrix[out_offset_d1 + out_offset_d2 + 2] = cur_result_5;
            res_matrix[out_offset_d1 + out_offset_d2 + 3] = cur_result_6;
            res_matrix[out_offset_d1 + out_offset_d2 + 4] = cur_result_3;
            res_matrix[out_offset_d1 + out_offset_d2 + 5] = cur_result_4;
            
        }
        
    }
    
    DNN_TRACE_4("finish %s\n","");
}

void conv3D_AVX_8ch(
    const float * __restrict__ p_in_matrix, 
    int m_d1, 
    int m_d2,
    const float * __restrict__ p_kernel,
    const float * __restrict__ p_biases,
    int n_kernels,
    int k_d1, 
    int k_d2,
    int n_channels,
    int n_stride,
    float * __restrict__ p_out_matrix) 
{    
    float *matrix = (float *)__builtin_assume_aligned(p_in_matrix, 64);
    float *kernel = (float *)__builtin_assume_aligned(p_kernel, 64);
    float *res_matrix = (float *)__builtin_assume_aligned(p_out_matrix, 64);
    
	float sum;
    float sum_arr[8];
	
    DNN_TRACE_4("start %s\n","");
    DNN_TRACE_4(" p_in_matrix: %p\n", p_in_matrix);
    DNN_TRACE_4(" m_d1,m_d2: [%d,%d]\n", m_d1, m_d2);
    DNN_TRACE_4(" p_kernel: %p\n", p_kernel);
    DNN_TRACE_4(" p_biases: %p\n", p_biases);
    DNN_TRACE_4(" n_kernels: %d\n", n_kernels);
    DNN_TRACE_4(" k_d1,k_d2: [%d,%d]\n", k_d1, k_d2);
    DNN_TRACE_4(" n_channels,n_stride: [%d,%d]\n", n_channels, n_stride);
    DNN_TRACE_4(" p_out_matrix: %p\n", p_out_matrix);
    
    int res_idx = 0;
    for (int cur_m_d1 = 0; (cur_m_d1 + k_d1) <= m_d1; cur_m_d1 += n_stride) {

        for (int cur_m_d2 = 0; (cur_m_d2 + k_d2) <= m_d2; cur_m_d2 += n_stride) {
            
            int is_first_result = 1;
            float cur_result = 0;
            
            for (int kernel_id = 0; kernel_id < n_kernels; kernel_id++) {
                
                __m256 ymm0;
                __m256 ymm1;
                __m256 ymm15 = _mm256_xor_ps(ymm15, ymm15); // init to 0s
                
                int kernel_shift = DIMS_TO_SIZE(k_d1, k_d2, n_channels) * kernel_id;
                
                for (int i = 0; i < k_d1; i++) {

                    int m_offset_d1 = D1_TO_OFFSET(cur_m_d1 + i, m_d2, n_channels);

                    int k_offset_d1 = D1_TO_OFFSET(i, k_d2, n_channels);

                    for (int j = 0; j < k_d2; j++) {

                        int m_offset_d2 = D2_TO_OFFSET(cur_m_d2 + j, n_channels);
                        
                        int k_offset_d2 = D2_TO_OFFSET(j, n_channels);

                        int m_shift = m_offset_d1 + m_offset_d2;
                        int k_shift = k_offset_d1 + k_offset_d2;

                        for (int k = 0 ; k < n_channels; k += 8) {       
                            ymm0 = _mm256_load_ps(&matrix[m_shift + k]);
                            ymm1 = _mm256_load_ps(&kernel[kernel_shift + k_shift + k]);

                            ymm15 = _mm256_fmadd_ps(ymm0, ymm1, ymm15);
                        }
                    }
                }

                sum = 0;
                _mm256_store_ps(sum_arr, ymm15);
                for (int m = 0; m < 8; m++) {
                    sum += sum_arr[m];
                }
                
                sum += p_biases[kernel_id];
                
                if (is_first_result) {
                    cur_result = sum;
                    is_first_result = 0;
                } else {
                    if (sum > cur_result) { // MAX
                        cur_result = sum;
                    }
                }
                
            }
            res_matrix[res_idx] = cur_result;
            res_idx++;
            
        }
        
    }
    
    DNN_TRACE_4("finish %s\n","");
}

void conv3D_AVX_8ch_NEW(
    const float * __restrict__ p_in_matrix, 
    const int m_d1, 
    const int m_d2,
    const int m_d1_start,
    const int m_d2_start,
    const int m_d1_finish,
    const int m_d2_finish,
    const float * __restrict__ p_kernel,
    const float * __restrict__ p_biases,
    const int n_maxout_kernels,
    const int k_d1, 
    const int k_d2,
    const int n_channels,
    const int n_stride,
    float * __restrict__ p_res_matrix) 
{    
    float * __restrict__ matrix = (float * __restrict__)__builtin_assume_aligned(p_in_matrix, 32);
    float * __restrict__ kernel = (float * __restrict__)__builtin_assume_aligned(p_kernel, 32);
    float * __restrict__ res_matrix = (float * __restrict__)__builtin_assume_aligned(p_res_matrix, 32);
    
    //float *matrix = p_in_matrix; 
    //float *kernel = p_kernel;
    //float *res_matrix = p_out_matrix;
    
	float sum;
    float sum_arr[8];
	
    DNN_TRACE_4("start %s\n","");
    DNN_TRACE_4(" p_in_matrix: %p\n", p_in_matrix);
    DNN_TRACE_4(" m_d1,m_d2: [%d,%d]\n", m_d1, m_d2);
    DNN_TRACE_4(" m_d1_start/finish,m_d2_start/finish: [%d,%d] <=> [%d,%d]\n", m_d1_start, m_d2_start, m_d1_finish, m_d2_finish);
    DNN_TRACE_4(" p_kernel: %p\n", p_kernel);
    DNN_TRACE_4(" p_biases: %p\n", p_biases);
    DNN_TRACE_4(" n_maxout_kernels: %d\n", n_maxout_kernels);
    DNN_TRACE_4(" k_d1,k_d2: [%d,%d]\n", k_d1, k_d2);
    DNN_TRACE_4(" n_channels,n_stride: [%d,%d]\n", n_channels, n_stride);
    DNN_TRACE_4(" p_res_matrix: %p\n", p_res_matrix);
    
    DNN_ASSERT(k_d2 == K_D2_4);
    
    int out_d2 = m_d2 - k_d2 + 1;
    int out_d3 = 1;
    
    for (int cur_m_d1 = m_d1_start; (cur_m_d1 + k_d1) <= m_d1_finish; cur_m_d1 += n_stride) {
        
        int out_offset_d1 = D1_TO_OFFSET(cur_m_d1, out_d2, out_d3);
        
        for (int cur_m_d2 = m_d2_start; (cur_m_d2 + k_d2) <= m_d2_finish; cur_m_d2 += n_stride) {
            
            int out_offset_d2 = D2_TO_OFFSET(cur_m_d2, out_d3);
            int res_idx = 0;
            
            int is_first_result = 1;
            float cur_result = 0;
            
            for (int kernel_id = 0; kernel_id < n_maxout_kernels; kernel_id++) {
                
                __m256 ymm0;
                __m256 ymm1;
                __m256 ymm2;
                __m256 ymm3;
                __m256 ymm4;
                __m256 ymm5;
                __m256 ymm6;
                __m256 ymm7;
                __m256 ymm12 = _mm256_xor_ps(ymm12, ymm12); // init to 0s
                __m256 ymm13 = _mm256_xor_ps(ymm13, ymm13); // init to 0s
                __m256 ymm14 = _mm256_xor_ps(ymm14, ymm14); // init to 0s
                __m256 ymm15 = _mm256_xor_ps(ymm15, ymm15); // init to 0s
                
                int kernel_shift = DIMS_TO_SIZE(k_d1, k_d2, n_channels) * kernel_id;
                
                for (int i = 0; i < k_d1; i++) {

                    int m_offset_d1 = D1_TO_OFFSET(cur_m_d1 + i, m_d2, n_channels);

                    int k_offset_d1 = D1_TO_OFFSET(i, k_d2, n_channels);

                    for (int j = 0; j < k_d2; j += 4) {

                        int m_offset_d2 = D2_TO_OFFSET(cur_m_d2 + j, n_channels);
                        
                        int k_offset_d2 = D2_TO_OFFSET(j, n_channels);

                        int m_shift = m_offset_d1 + m_offset_d2;
                        int k_shift = k_offset_d1 + k_offset_d2;

                        for (int k = 0 ; k < n_channels; k += 8) {       
                            int m_total_shift = m_shift + k;
                            int k_total_shift = kernel_shift + k_shift + k;
                            
                            ymm0 = _mm256_load_ps(&matrix[m_total_shift]);
                            ymm1 = _mm256_load_ps(&kernel[k_total_shift]);
                            
                            ymm15 = _mm256_fmadd_ps(ymm0, ymm1, ymm15);
                            
                            ymm2 = _mm256_load_ps(&matrix[m_total_shift + n_channels]);
                            ymm3 = _mm256_load_ps(&kernel[k_total_shift + n_channels]);
                            
                            ymm14 = _mm256_fmadd_ps(ymm2, ymm3, ymm14);
                            
                            ymm4 = _mm256_load_ps(&matrix[m_total_shift + n_channels * 2]);
                            ymm5 = _mm256_load_ps(&kernel[k_total_shift + n_channels * 2]);
                            
                            ymm13 = _mm256_fmadd_ps(ymm4, ymm5, ymm13);
                            
                            ymm6 = _mm256_load_ps(&matrix[m_total_shift + n_channels * 3]);
                            ymm7 = _mm256_load_ps(&kernel[k_total_shift + n_channels * 3]);
                            
                            ymm12 = _mm256_fmadd_ps(ymm6, ymm7, ymm12);
                            
                        }
                    }
                }
                
                ymm13 = _mm256_add_ps(ymm12, ymm13);
                ymm15 = _mm256_add_ps(ymm14, ymm15);
                ymm15 = _mm256_add_ps(ymm13, ymm15);
                sum = 0;
                _mm256_store_ps(sum_arr, ymm15);
                for (int m = 0; m < 8; m++) {
                    sum += sum_arr[m];
                }
                
                sum += p_biases[kernel_id];
                
                if (is_first_result) {
                    cur_result = sum;
                    is_first_result = 0;
                } else {
                    if (sum > cur_result) { // MAX
                        cur_result = sum;
                    }
                }
                
            }
            res_matrix[out_offset_d1 + out_offset_d2 + res_idx] = cur_result;
            res_idx++;
            
        }
        
    }
    
    DNN_TRACE_4("finish %s\n","");
}

void conv3D_1ch(
    const float * __restrict__ p_in_matrix, 
    int m_d1, 
    int m_d2,
    const float * __restrict__ p_kernel,
    const float * __restrict__ p_biases,
    int n_kernels,
    int k_d1, 
    int k_d2,
    int n_channels,
    int n_stride,
    float * __restrict__ p_out_matrix) 
{    
    float *matrix = (float *)__builtin_assume_aligned(p_in_matrix, 64);
    float *kernel = (float *)__builtin_assume_aligned(p_kernel, 64);
    float *res_matrix = (float *)__builtin_assume_aligned(p_out_matrix, 64);
    
    DNN_TRACE_4("start %s\n","");
    DNN_TRACE_4(" p_in_matrix: %p\n", p_in_matrix);
    DNN_TRACE_4(" m_d1,m_d2: [%d,%d]\n", m_d1, m_d2);
    DNN_TRACE_4(" p_kernel: %p\n", p_kernel);
    DNN_TRACE_4(" p_biases: %p\n", p_biases);
    DNN_TRACE_4(" n_kernels: %d\n", n_kernels);
    DNN_TRACE_4(" k_d1,k_d2: [%d,%d]\n", k_d1, k_d2);
    DNN_TRACE_4(" n_channels,n_stride: [%d,%d]\n", n_channels, n_stride);
    DNN_TRACE_4(" p_out_matrix: %p\n", p_out_matrix);
    
    int res_idx = 0;
    for (int cur_m_d1 = 0; (cur_m_d1 + k_d1) <= m_d1; cur_m_d1 += n_stride) {

        for (int cur_m_d2 = 0; (cur_m_d2 + k_d2) <= m_d2; cur_m_d2 += n_stride) {
            
            int is_first_result = 1;
            float cur_result = 0;
            
            for (int kernel_id = 0; kernel_id < n_kernels; kernel_id++) {
                
                float sum = 0;
                
                int kernel_shift = DIMS_TO_SIZE(k_d1, k_d2, n_channels) * kernel_id;
                
                for (int i = 0; i < k_d1; i++) {

                    int m_offset_d1 = D1_TO_OFFSET(cur_m_d1 + i, m_d2, n_channels);

                    int k_offset_d1 = D1_TO_OFFSET(i, k_d2, n_channels);

                    for (int j = 0; j < k_d2; j++) {

                        int m_offset_d2 = D2_TO_OFFSET(cur_m_d2 + j, n_channels);

                        int k_offset_d2 = D2_TO_OFFSET(j, n_channels);

                        int m_shift = m_offset_d1 + m_offset_d2;
                        int k_shift = k_offset_d1 + k_offset_d2;

                        for (int k = 0 ; k < n_channels; k += 1) {       
                            sum += matrix[m_shift + k] * kernel[kernel_shift + k_shift + k];
                        }
                    }
                }
                
                sum += p_biases[kernel_id];
                
                if (is_first_result) {
                    cur_result = sum;
                    is_first_result = 0;
                } else {
                    if (sum > cur_result) { // MAX
                        cur_result = sum;
                    }
                }
                
            }
            res_matrix[res_idx] = cur_result;
            res_idx++;
            
        }
        
    }
    
    DNN_TRACE_4("finish %s\n","");
}

void conv3D_1ch_NEW(
    const float * __restrict__ p_in_matrix, 
    int m_d1, 
    int m_d2,
    const float * __restrict__ p_kernel,
    const float * __restrict__ p_biases,
    int n_kernels,
    int k_d1, 
    int k_d2,
    int n_channels,
    int n_stride,
    float * __restrict__ p_res_matrix) 
{    
    float * __restrict__ matrix = (float * __restrict__)__builtin_assume_aligned(p_in_matrix, 32);
    float * __restrict__ kernel = (float * __restrict__)__builtin_assume_aligned(p_kernel, 32);
    float * __restrict__ res_matrix = (float * __restrict__)__builtin_assume_aligned(p_res_matrix, 32);
    
    float sum;
    float sum_arr[8];
    
    float m_f_buffer[256];
    float k_f_buffer[256];
    
    DNN_TRACE_4("start %s\n","");
    DNN_TRACE_4(" p_in_matrix: %p\n", p_in_matrix);
    DNN_TRACE_4(" m_d1,m_d2: [%d,%d]\n", m_d1, m_d2);
    DNN_TRACE_4(" p_kernel: %p\n", p_kernel);
    DNN_TRACE_4(" p_biases: %p\n", p_biases);
    DNN_TRACE_4(" n_kernels: %d\n", n_kernels);
    DNN_TRACE_4(" k_d1,k_d2: [%d,%d]\n", k_d1, k_d2);
    DNN_TRACE_4(" n_channels,n_stride: [%d,%d]\n", n_channels, n_stride);
    DNN_TRACE_4(" p_res_matrix: %p\n", p_res_matrix);
    
    int res_idx = 0;
    int buf_size = 0;
    for (int cur_m_d1 = 0; (cur_m_d1 + k_d1) <= m_d1; cur_m_d1 += n_stride) {

        for (int cur_m_d2 = 0; (cur_m_d2 + k_d2) <= m_d2; cur_m_d2 += n_stride) {
            
            int is_first_result = 1;
            float cur_result = 0;
            
            for (int kernel_id = 0; kernel_id < n_kernels; kernel_id++) {
                
                int kernel_shift = DIMS_TO_SIZE(k_d1, k_d2, n_channels) * kernel_id;
                
                __m256 ymm0;
                __m256 ymm1;
                __m256 ymm14 = _mm256_xor_ps(ymm14, ymm14); // init to 0s
                __m256 ymm15 = _mm256_xor_ps(ymm15, ymm15); // init to 0s
                
                int m_offset_d1 = D1_TO_OFFSET(cur_m_d1, m_d2, n_channels) - (m_d2 * n_channels);
                
                int k_offset_d1 = -(k_d2 * n_channels);
                for (int i = 0; i < k_d1; i++) {

                    m_offset_d1 += (m_d2 * n_channels);
                    
                    k_offset_d1 += (k_d2 * n_channels);

                    int m_offset_d2 = D2_TO_OFFSET(cur_m_d2, n_channels) - n_channels;
                    
                    int k_offset_d2 = -n_channels;
                    
                    for (int j = 0; j < k_d2; j++) {

                        m_offset_d2 += n_channels;

                        k_offset_d2 += n_channels;

                        for (int k = 0 ; k < n_channels; k += 1) {       
                            m_f_buffer[buf_size] = matrix[m_offset_d1 + m_offset_d2 + k];
                            k_f_buffer[buf_size] = kernel[kernel_shift + k_offset_d1 + k_offset_d2 + k];
                            buf_size++;
                            
                            if (buf_size % 8 == 0) {
                                ymm0 = _mm256_load_ps(&m_f_buffer[buf_size]);
                                ymm1 = _mm256_load_ps(&k_f_buffer[buf_size]);

                                ymm15 = _mm256_fmadd_ps(ymm0, ymm1, ymm15);
                            }
                            
                            //matrix[m_shift + k] * kernel[kernel_shift + k_shift + k];
                        }
                    }
                }
                
                if (buf_size >= 256) {
                    buf_size = 0;
                }
                
                //ymm15 = _mm256_add_ps(ymm14, ymm15);
                
                sum = 0;
                _mm256_store_ps(sum_arr, ymm15);
                for (int m = 0; m < 8; m++) {
                    sum += sum_arr[m];
                }
                
                sum += p_biases[kernel_id];
                
                if (is_first_result) {
                    cur_result = sum;
                    is_first_result = 0;
                } else {
                    if (sum > cur_result) { // MAX
                        cur_result = sum;
                    }
                }
                
            }
            res_matrix[res_idx] = cur_result;
            res_idx++;
            
        }
        
    }
    
    DNN_TRACE_4("finish %s\n","");
}

void matrix_to_conv_depth_4x4x1k(
    const float * __restrict__ p_in_matrix,
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
    DNN_ASSERT(k_d1 == K_D1_4);
    DNN_ASSERT(k_d2 == K_D2_4);
    DNN_ASSERT(n_channels == 1);
    
    int out_d2 = OUT_DIM(m_d2, K_D2_4, n_stride);
    int out_d3 = K_D1_4 * K_D2_4;
    
    for (int cur_m_d1 = m_d1_start; (cur_m_d1 + k_d1) <= m_d1_finish; cur_m_d1 += n_stride) {
        
        int out_offset_d1 = D1_TO_OFFSET(cur_m_d1, out_d2, out_d3);
        
        for (int cur_m_d2 = m_d2_start; (cur_m_d2 + k_d2) <= m_d2_finish; cur_m_d2 += n_stride) {
            
            int out_offset_d2 = D2_TO_OFFSET(cur_m_d2, out_d3);
            
            int cur_idx = 0;
            for (int i = 0; i < K_D1_4; i++) {
                
                int m_offset_d1 = D1_TO_OFFSET(cur_m_d1 + i, m_d2, n_channels);
                
                for (int j = 0; j < K_D2_4; j++) {
                    
                    int m_offset_d2 = D2_TO_OFFSET(cur_m_d2 + j, n_channels);
                    
                    p_out_matrix[out_offset_d1 + out_offset_d2 + cur_idx] = p_in_matrix[m_offset_d1 + m_offset_d2];
                    cur_idx++;
                    
                }
            }
        }
    }
     
}

void matrix_to_conv_depth_general(
    const float * __restrict__ p_in_matrix,
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
    int out_d2 = OUT_DIM(m_d2, k_d2, n_stride);
    int out_d3 = k_d1 * k_d2 * n_channels;
    
    for (int cur_m_d1 = m_d1_start; (cur_m_d1 + k_d1) <= m_d1_finish; cur_m_d1 += n_stride) {
        
        int out_offset_d1 = D1_TO_OFFSET(cur_m_d1, out_d2, out_d3);
        
        for (int cur_m_d2 = m_d2_start; (cur_m_d2 + k_d2) <= m_d2_finish; cur_m_d2 += n_stride) {
            
            int out_offset_d2 = D2_TO_OFFSET(cur_m_d2, out_d3);
            
            int cur_idx = 0;
            for (int i = 0; i < k_d1; i++) {
                
                int m_offset_d1 = D1_TO_OFFSET(cur_m_d1 + i, m_d2, n_channels);
                
                for (int j = 0; j < k_d2; j++) {
                    
                    int m_offset_d2 = D2_TO_OFFSET(cur_m_d2 + j, n_channels);
                    
                    for (int k = 0; k < n_channels; k++) {
                        p_out_matrix[out_offset_d1 + out_offset_d2 + cur_idx] = p_in_matrix[m_offset_d1 + m_offset_d2 + k];
                        cur_idx++;
                    }
                }
            }
        }
    }
     
}

void matrix_to_conv_depth(
    const float * __restrict__ p_in_matrix,
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
    if ((k_d1 == K_D1_4) && (k_d2 == K_D2_4) && (n_channels == 1)) {
        matrix_to_conv_depth_4x4x1k(
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
        
    } else {
        matrix_to_conv_depth_general(
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
           
    }
}

void conv3D_depth(
    const float * __restrict__ p_in_matrix, 
    int m_d1, 
    int m_d2,
    const float * __restrict__ p_kernel,
    const float * __restrict__ p_biases,
    int n_kernels,
    int k_d1,
    int k_d2,
    int n_channels,
    float * __restrict__ p_out_matrix) 
{
    float * __restrict__ matrix = (float * __restrict__)__builtin_assume_aligned(p_in_matrix, 32);
    float * __restrict__ kernel = (float * __restrict__)__builtin_assume_aligned(p_kernel, 32);
    float * __restrict__ out_matrix = (float * __restrict__)__builtin_assume_aligned(p_out_matrix, 32);
    
    float sum;
    float sum_arr[8];
        
    DNN_TRACE_4("start %s\n","");
    DNN_TRACE_4(" p_in_matrix: %p\n", p_in_matrix);
    DNN_TRACE_4(" m_d1,m_d2: [%d,%d]\n", m_d1, m_d2);
    DNN_TRACE_4(" p_kernel: %p\n", p_kernel);
    DNN_TRACE_4(" p_biases: %p\n", p_biases);
    DNN_TRACE_4(" n_kernels: %d\n", n_kernels);
    DNN_TRACE_4(" k_d1,k_d2: [%d,%d]\n", k_d1, k_d2);
    DNN_TRACE_4(" n_channels: %d\n", n_channels);
    DNN_TRACE_4(" p_out_matrix: %p\n", p_out_matrix);
    
    int kernel_size = DIMS_TO_SIZE(k_d1, k_d2, n_channels);
    
    int m_d3 = kernel_size;
    
    int out_idx = 0;
    
    for (int cur_m_d1 = 0; cur_m_d1 < m_d1; cur_m_d1++) {

        for (int cur_m_d2 = 0; cur_m_d2 < m_d2; cur_m_d2++) {
            
            int is_first_result = 1;
            float cur_result = 0;
            
            int m_offset = D1_TO_OFFSET(cur_m_d1, m_d2, m_d3) + D2_TO_OFFSET(cur_m_d2, m_d3);
            
            //prefetch_read_no_locality(&matrix[m_offset + m_d3]);
            
            for (int kernel_id = 0; kernel_id < n_kernels; kernel_id++) {
                
                int kernel_shift = kernel_size * kernel_id;
                
                __m256 ymm0;
                __m256 ymm1;
                __m256 ymm2;
                __m256 ymm3;
                __m256 ymm4;
                __m256 ymm5;
                __m256 ymm6;
                __m256 ymm7;
                __m256 ymm12 = _mm256_xor_ps(ymm12, ymm12); // init to 0s
                __m256 ymm13 = _mm256_xor_ps(ymm13, ymm13); // init to 0s
                __m256 ymm14 = _mm256_xor_ps(ymm14, ymm14); // init to 0s
                __m256 ymm15 = _mm256_xor_ps(ymm15, ymm15); // init to 0s
                
                if (kernel_size <= 16) { 
                    for (int i = 0; i < kernel_size; i += 16) {
                    
                        ymm0 = _mm256_load_ps(&matrix[m_offset + i]);
                        ymm1 = _mm256_load_ps(&kernel[kernel_shift + i]);
                        
                        ymm15 = _mm256_fmadd_ps(ymm0, ymm1, ymm15);

                        ymm2 = _mm256_load_ps(&matrix[m_offset + i + 8]);
                        ymm3 = _mm256_load_ps(&kernel[kernel_shift + i + 8]);

                        ymm14 = _mm256_fmadd_ps(ymm2, ymm3, ymm14);

                    }
                } else {
                    for (int i = 0; i < kernel_size; i += 32) {
                    
                        ymm0 = _mm256_load_ps(&matrix[m_offset + i]);
                        ymm1 = _mm256_load_ps(&kernel[kernel_shift + i]);

                        ymm15 = _mm256_fmadd_ps(ymm0, ymm1, ymm15);

                        ymm2 = _mm256_load_ps(&matrix[m_offset + i + 8]);
                        ymm3 = _mm256_load_ps(&kernel[kernel_shift + i + 8]);

                        ymm14 = _mm256_fmadd_ps(ymm2, ymm3, ymm14);
                        
                        ymm4 = _mm256_load_ps(&matrix[m_offset + i + 16]);
                        ymm5 = _mm256_load_ps(&kernel[kernel_shift + i + 16]);

                        ymm13 = _mm256_fmadd_ps(ymm4, ymm5, ymm13);
                        
                        ymm6 = _mm256_load_ps(&matrix[m_offset + i + 24]);
                        ymm7 = _mm256_load_ps(&kernel[kernel_shift + i + 24]);

                        ymm12 = _mm256_fmadd_ps(ymm6, ymm7, ymm12);

                    }
                }
                
                if (kernel_size <= 16) { 
                    ymm15 = _mm256_add_ps(ymm14, ymm15);
                } else {
                    ymm15 = _mm256_add_ps(ymm14, ymm15);
                    ymm13 = _mm256_add_ps(ymm12, ymm13);
                    ymm15 = _mm256_add_ps(ymm13, ymm15);
                }
                
                sum = 0;
                _mm256_store_ps(sum_arr, ymm15);
                for (int m = 0; m < 8; m++) {
                    sum += sum_arr[m];
                }
                
                sum += p_biases[kernel_id];
                
                if (is_first_result) {
                    cur_result = sum;
                    is_first_result = 0;
                } else {
                    if (sum > cur_result) { // MAX
                        cur_result = sum;
                    }
                }
                
            }
            out_matrix[out_idx] = cur_result;
            out_idx++;
            
        }
        
    }
    
    DNN_TRACE_4("finish %s\n","");
}

void conv3D(
    const float * __restrict__ p_in_matrix, 
    const uint32_t * __restrict__ p_sp_out_matrix, 
    const uint32_t * __restrict__ p_sp_mask_matrix,
    int m_d1, 
    int m_d2,
    int m_d1_start,
    int m_d2_start,
    int m_d1_finish,
    int m_d2_finish,
    const float * __restrict__ p_kernel,
    const float * __restrict__ p_biases,
    int n_kernels,
    int n_maxout_kernels,
    int k_d1, 
    int k_d2,
    int n_channels,
    int n_stride,
    float * __restrict__ p_out_matrix)
{
    DNN_ASSERT(n_kernels == n_maxout_kernels);
    
    if ((n_channels == N_CHANNELS_48) && (k_d1 == K_D1_1) && (k_d2 == K_D2_1) && (n_kernels == 2)) {
        conv3D_AVX_48ch_exact_1k_2X_NEW(
            p_in_matrix, 
            m_d1,
            m_d2,
            m_d1_start,
            m_d2_start,
            m_d1_finish,
            m_d2_finish,
            p_kernel, 
            p_biases, 
            n_kernels,
            n_maxout_kernels, 
            k_d1,
            k_d2, 
            n_channels, 
            n_stride, 
            p_out_matrix);
        
    } else if ((n_channels == N_CHANNELS_16) && (k_d1 == K_D1_1) && (k_d2 == K_D2_1) && (n_kernels == 2)) {
        conv3D_AVX_16ch_exact_1k_2X_NEW(
            p_in_matrix, 
            m_d1,
            m_d2,
            m_d1_start,
            m_d2_start,
            m_d1_finish,
            m_d2_finish,
            p_kernel, 
            p_biases, 
            n_kernels,
            n_maxout_kernels, 
            k_d1, 
            k_d2, 
            n_channels, 
            n_stride, 
            p_out_matrix);
    
    } else if ((n_channels == N_CHANNELS_16) && (k_d1 == K_D1_4) && (k_d2 == K_D2_4) && (n_kernels == 2)) {
        conv3D_AVX_16ch_exact_2k_2X_NEW(
            p_in_matrix, 
            m_d1,
            m_d2,
            m_d1_start,
            m_d2_start,
            m_d1_finish,
            m_d2_finish,
            p_kernel, 
            p_biases, 
            n_kernels,
            n_maxout_kernels, 
            k_d1, 
            k_d2, 
            n_channels, 
            n_stride, 
            p_out_matrix);
    
   } else if ((n_channels == N_CHANNELS_8) && (k_d1 == K_D1_4) && (k_d2 == K_D2_4) && (n_kernels == 2)) {
        // FIX FIX
        conv3D_AVX_8ch_exact_2k_2X_NEW(
            p_in_matrix, 
            m_d1,
            m_d2,
            m_d1_start,
            m_d2_start,
            m_d1_finish,
            m_d2_finish,
            p_kernel, 
            p_biases, 
            n_kernels,
            n_maxout_kernels, 
            k_d1, 
            k_d2, 
            n_channels, 
            n_stride, 
            p_out_matrix);
        
        // QTD
        // convolve_by_parts(
        //                             p_in_matrix, 
        //                             p_sp_out_matrix,
        //                             p_sp_mask_matrix,
        //                             m_d1,
        //                             m_d2,
        //                             p_kernel, 
        //                             p_biases,
        //                             n_kernels,
        //                             k_d1, 
        //                             k_d2, 
        //                             n_channels,
        //                             p_out_matrix);
        
    } else if ((n_channels == N_CHANNELS_8) && (k_d1 == K_D1_2) && (k_d2 == K_D2_2) && (n_kernels == 2)) {
        conv3D_AVX_8ch_exact_2k_2X_NEW_k2_k2(
            p_in_matrix, 
            m_d1,
            m_d2,
            m_d1_start,
            m_d2_start,
            m_d1_finish,
            m_d2_finish,
            p_kernel, 
            p_biases, 
            n_kernels,
            n_maxout_kernels, 
            k_d1, 
            k_d2, 
            n_channels, 
            n_stride, 
            p_out_matrix);

    } else if ((n_channels == N_CHANNELS_32) && (k_d1 == K_D1_4) && (k_d2 == K_D2_4) && (n_kernels == 2)) {
        //conv3D_AVX_32ch_exact_2k(p_in_matrix, m_d1, m_d2, p_kernel, p_biases, n_kernels, k_d1, k_d2, n_channels, n_stride, p_out_matrix);
        
        // FIX FIX FIX
        conv3D_AVX_32ch_exact_2k_2X_NEW(
            p_in_matrix, 
            m_d1,
            m_d2,
            m_d1_start,
            m_d2_start,
            m_d1_finish,
            m_d2_finish,
            p_kernel, 
            p_biases, 
            n_kernels,
            n_maxout_kernels, 
            k_d1, 
            k_d2, 
            n_channels, 
            n_stride, 
            p_out_matrix);

        //         
        // conv3D_AVX_32ch_exact_2k_2X_NEW_QTD(
        //             p_in_matrix, 
        //             p_sp_out_matrix,
        //             m_d1,
        //             m_d2,
        //             m_d1_start,
        //             m_d2_start,
        //             m_d1_finish,
        //             m_d2_finish,
        //             p_kernel, 
        //             p_biases, 
        //             n_kernels,
        //             n_maxout_kernels, 
        //             k_d1, 
        //             k_d2, 
        //             n_channels, 
        //             n_stride, 
        //             p_out_matrix);

        // QTD
        // convolve_by_parts(
        //                             p_in_matrix, 
        //                             p_sp_out_matrix,
        //                             p_sp_mask_matrix,
        //                             m_d1,
        //                             m_d2,
        //                             p_kernel, 
        //                             p_biases,
        //                             n_kernels,
        //                             k_d1, 
        //                             k_d2, 
        //                             n_channels,
        //                             p_out_matrix);

        //conv3D_AVX_32ch_exact(p_in_matrix, m_d1, m_d2, p_kernel, p_biases, n_kernels, k_d1, k_d2, n_channels, n_stride, p_out_matrix);
        
    } else if ((n_channels == N_CHANNELS_32) && (k_d1 == K_D1_4) && (k_d2 == K_D2_4)) {
        conv3D_AVX_32ch_exact_2X_NEW(
            p_in_matrix, 
            m_d1,
            m_d2,
            m_d1_start,
            m_d2_start,
            m_d1_finish,
            m_d2_finish,
            p_kernel, 
            p_biases, 
            n_kernels,
            n_maxout_kernels, 
            k_d1, 
            k_d2, 
            n_channels, 
            n_stride, 
            p_out_matrix);
        //conv3D_AVX_32ch_exact(p_in_matrix, m_d1, m_d2, p_kernel, p_biases, n_kernels, k_d1, k_d2, n_channels, n_stride, p_out_matrix);
        
        //conv3D_AVX_32ch_depth_exact(p_in_matrix, m_d1, m_d2, p_kernel, p_biases, n_kernels, n_maxout_kernels, k_d1, k_d2, n_channels, n_stride, p_out_matrix);
    } else if ((n_channels == N_CHANNELS_32) && (k_d1 == K_D1_6) && (k_d2 == K_D2_6)) {
        conv3D_AVX_32ch_exact_2X_NEW_k6_k6(
            p_in_matrix, 
            m_d1,
            m_d2,
            m_d1_start,
            m_d2_start,
            m_d1_finish,
            m_d2_finish,
            p_kernel, 
            p_biases, 
            n_kernels,
            n_maxout_kernels, 
            k_d1, 
            k_d2, 
            n_channels, 
            n_stride, 
            p_out_matrix);
            //conv3D_AVX_32ch_exact(p_in_matrix, m_d1, m_d2, p_kernel, p_biases, n_kernels, k_d1, k_d2, n_channels, n_stride, p_out_matrix);

            //conv3D_AVX_32ch_depth_exact(p_in_matrix, m_d1, m_d2, p_kernel, p_biases, n_kernels, n_maxout_kernels, k_d1, k_d2, n_channels, n_stride, p_out_matrix);
    
    } else if ((n_channels == N_CHANNELS_8) && (k_d1 == K_D1_6) && (k_d2 == K_D2_6)) {
        conv3D_AVX_8ch_exact_2X_NEW_k6_k6(
            p_in_matrix, 
            m_d1,
            m_d2,
            m_d1_start,
            m_d2_start,
            m_d1_finish,
            m_d2_finish,
            p_kernel, 
            p_biases, 
            n_kernels,
            n_maxout_kernels, 
            k_d1, 
            k_d2, 
            n_channels, 
            n_stride, 
            p_out_matrix);

    } else if (n_channels % 32 == 0) {
        abort();
        //conv3D_AVX_32ch_depth(p_in_matrix, m_d1, m_d2, p_kernel, p_biases, n_kernels, n_maxout_kernels, k_d1, k_d2, n_channels, n_stride, p_out_matrix);
        conv3D_AVX_32ch(p_in_matrix, m_d1, m_d2, p_kernel, p_biases, n_maxout_kernels, k_d1, k_d2, n_channels, n_stride, p_out_matrix);
    } else if (n_channels % 8 == 0) {
        DNN_ASSERT(n_maxout_kernels == n_kernels);
        
        conv3D_AVX_8ch_NEW(
            p_in_matrix, 
            m_d1, 
            m_d2, 
            m_d1_start,
            m_d2_start,
            m_d1_finish,
            m_d2_finish,
            p_kernel, 
            p_biases, 
            n_maxout_kernels, 
            k_d1,
            k_d2, 
            n_channels, 
            n_stride, 
            p_out_matrix);
        
    } else {
        abort();
        //conv3D_1ch_NEW_NEW(p_in_matrix, m_d1, m_d2, p_kernel, p_biases, n_kernels, k_d1, k_d2, n_channels, n_stride, p_out_matrix, (float *)&g_tmp_matrix);
    }
}