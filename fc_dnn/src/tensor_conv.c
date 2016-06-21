////////////////////////////////////////////////////////////////////////////////
// tensorConv.cpp
// David Budden
// Different methods for convolving tensors in a FCNN context.

#include "tensor_conv.h"

////////////////////////////////////////////////////////////////////////////////
// Helper functions for encoding/decoding 2D points (sizes, locations, etc.)
int32_t encode(const int x, const int y) {
	int32_t out = (x & 0xffff) << 16;
	return out | (y & 0xffff);
}
void decode(const int32_t in, int *p_x, int *p_y ) {
	*p_x = (in >> 16) & 0xffff;
	*p_y = in & 0xffff;
}

////////////////////////////////////////////////////////////////////////////////
// convolve_by_parts
// Convolution by parts (1x1 kernels), lossless (interpolate SP borders).
void convolve_by_parts(
    const float * __restrict__ p_in_matrix, 
    const uint32_t * __restrict__ p_s_matrix,
    const uint32_t * __restrict__ p_mask_matrix,
    const int in_d1,
    const int in_d2,
    const float * __restrict__ p_kernels,
    const float * __restrict__ p_biases,
    int n_maxout_kernels,
    int k_d1, 
    int k_d2,
    int n_channels,
    float * __restrict__ p_out_matrix) {
     
    DNN_TRACE_4("start %s\n","");
    DNN_TRACE_4(" p_in_matrix: %p\n", p_in_matrix);
    DNN_TRACE_4(" in_d1,in_d2: [%d,%d]\n", in_d1, in_d2);
    DNN_TRACE_4(" p_kernels: %p\n", p_kernels);
    DNN_TRACE_4(" p_biases: %p\n", p_biases);
    DNN_TRACE_4(" n_maxout_kernels: %d\n", n_maxout_kernels);
    DNN_TRACE_4(" k_d1,k_d2: [%d,%d]\n", k_d1, k_d2);
    DNN_TRACE_4(" n_channels: %d\n", n_channels);
    DNN_TRACE_4(" p_out_matrix: %p\n", p_out_matrix);
    
    DNN_ASSERT(k_d1 == 4);
    DNN_ASSERT(k_d2 == 4);
    DNN_ASSERT(n_maxout_kernels == 2);
    DNN_ASSERT((n_channels == 32) || (n_channels == 8));
    
	// Dimensions etc.
    const int N1 = in_d1;
    const int N2 = in_d2;
    const int D = n_channels;
    const int K1 = k_d1;
    const int K2 = k_d2;
	const int M1 = OUT_DIM(in_d1, k_d1, 1);
	const int M2 = OUT_DIM(in_d2, k_d2, 1);
        
	// Stack for storing next superpixel location.
	int32_t* stk = (int32_t *)malloc(sizeof(int32_t) * M1 * M2);
    
    char *p_visited = (char *)malloc(sizeof(char) * N1 * N2);
    
    float *p_local_out_matrix = (float *)malloc(sizeof(float) * M1 * M2);
    
    int is_first_inprod = 1;
    
    int n_qtd = 0;
    
	// Convolve 1x1 kernels only.
	for (int k_id = 0; k_id < n_maxout_kernels; k_id++) {
	    
        for (int i = 0; i < (M1 * M2); i++) {
            p_local_out_matrix[i] = 0.0;
        }
        
        for (int krow = 0; krow < K1; krow++) {
            for (int kcol = 0; kcol < K2; kcol++) {
                // NOTE: Can optimize out this 'visited' matrix if using
                // separate output layers for each 1x1 kernel; just check the 
                // output for a non-default value.
                //Matrix32 visited = Matrix32::Zero(M1, M2);
                memset(p_visited, 0, sizeof(char) * N1 * N2);

                // Reset stack (this step is different in lossy version).
                int stkptr = 0;
                stk[stkptr++] = encode(krow, kcol);

                // Iterate across all superpixels.
                while (stkptr) {
                    // Extract superpixel size and location.
                    int srow, scol, size_x, size_y;

                    decode(stk[--stkptr], &srow, &scol);

                    int s_d1_offset = D1_TO_OFFSET(srow, N2, 1);
                    int s_d2_offset = D2_TO_OFFSET(scol, 1);

                    uint32_t s_val = p_s_matrix[s_d1_offset + s_d2_offset];
                    decode(s_val, &size_x, &size_y);
                    
                    // Mark this position visited (see above)
                    p_visited[s_d1_offset + s_d2_offset] = 1;

                    // Only need to process depth once for this SP.
                    // Determine inner product down z-axis (AVX!)

                    int in_d1_offset = D1_TO_OFFSET(srow, N2, D);
                    int in_d2_offset = D2_TO_OFFSET(scol, D);

                    int cur_k_id_offset = k_id * DIMS_TO_SIZE(K1, K2, D);
                    int cur_k_d1_offset = D1_TO_OFFSET(krow, K2, D);
                    int cur_k_d2_offset = D2_TO_OFFSET(kcol, D);
                    int cur_k_full_offset = cur_k_id_offset + cur_k_d1_offset + cur_k_d2_offset;

                    float inprod = 0.0;    

                    for (int d = 0; d < D; d++) {

                        float k_val = p_kernels[cur_k_full_offset + d];

                        float in_val = p_in_matrix[in_d1_offset + in_d2_offset + d];

                        inprod += (k_val * in_val);
                    }
                    
                    // Process superpixel to populate output image.
                    const int offset_x = srow + size_x;
                    const int offset_y = scol + size_y;

                    // This step is different in lossy version.
                    
                    int x_s_limit = M1 + krow;
                    if (offset_x < (M1 + krow)) {
                        x_s_limit = offset_x;
                    }
                    
                    int y_s_limit = M2 + kcol;
                    if (offset_y < (M2 + kcol)) {
                        y_s_limit = offset_y;
                    }
                    
                    for (int row = srow; row < x_s_limit; row++) {
                        for (int col = scol; col < y_s_limit; col++) {
                            if (!((row == srow) && (col == scol))) {
                                n_qtd++;
                            }
                            
                            // This just adds to a single output matrix, but you 
                            // could write to separate mats and reduce later.

                            int cur_out_d1 = row - krow;
                            int cur_out_d2 = col - kcol;

                            int out_d1_offset = D1_TO_OFFSET(cur_out_d1, M2, 1);
                            int out_d2_offset = D2_TO_OFFSET(cur_out_d2, 1);

                            p_local_out_matrix[out_d1_offset + out_d2_offset] += inprod;
                            //p_out_matrix[out_d1_offset + out_d2_offset] += inprod_max;
                        }
                    }

                    // Add next superpixels to stack if unvisited.
                    // This step is different in lossy version.
                    int x_s_d1_offset = D1_TO_OFFSET(offset_x, N2, 1);
                    int x_s_d2_offset = D2_TO_OFFSET(scol, 1);

                    int y_s_d1_offset = D1_TO_OFFSET(srow, N2, 1);
                    int y_s_d2_offset = D2_TO_OFFSET(offset_y, 1);

                    if (((offset_x < N1) && (p_visited[x_s_d1_offset + x_s_d2_offset] == 0)) &&
                         (scol == kcol || (p_mask_matrix[x_s_d1_offset + x_s_d2_offset] != 0))) { // Check for SP.
                        stk[stkptr++] = encode(offset_x, scol);
                    }

                    if (((offset_y < N2) && (p_visited[y_s_d1_offset + y_s_d2_offset] == 0)) &&
                         (srow == krow || (p_mask_matrix[y_s_d1_offset + y_s_d2_offset] != 0))) { // Check for SP.
                        stk[stkptr++] = encode(srow, offset_y);
                    }
                }
            }
        }
        
        for (int out_d1 = 0; out_d1 < M1; out_d1++) {
            
            int out_d1_offset = D1_TO_OFFSET(out_d1, M2, 1);
            
            for (int out_d2 = 0; out_d2 < M2; out_d2++) {
                int out_d2_offset = D2_TO_OFFSET(out_d2, 1);
                
                p_local_out_matrix[out_d1_offset + out_d2_offset] += p_biases[k_id];
                
                float cur_res = p_local_out_matrix[out_d1_offset + out_d2_offset];
                
                if (is_first_inprod) {
                    p_out_matrix[out_d1_offset + out_d2_offset] = cur_res;
                } else {
                    if (cur_res > p_out_matrix[out_d1_offset + out_d2_offset]) {
                        p_out_matrix[out_d1_offset + out_d2_offset] = cur_res;
                    }
                }
            }
        }
        
        if (is_first_inprod) {
            is_first_inprod = 0;
        }
        
    }   
	
    free(stk);
    free(p_visited);
    free(p_local_out_matrix);
    
	DNN_TRACE_4("finish [qtd = %d (%.2f)]\n", 
	    n_qtd, 
	    float(n_qtd) / ((N1 * N2 * K1 * K2) * n_maxout_kernels));
}

// void convolve_by_parts(
//     const float * __restrict__ p_in_matrix, 
//     const uint32_t * __restrict__ p_s_matrix,
//     const uint32_t * __restrict__ p_mask_matrix,
//     const int in_d1,
//     const int in_d2,
//     const float * __restrict__ p_kernels,
//     const float * __restrict__ p_biases,
//     int n_maxout_kernels,
//     int k_d1, 
//     int k_d2,
//     int n_channels,
//     float * __restrict__ p_out_matrix) {
//      
//     DNN_TRACE_4("start %s\n","");
//     DNN_TRACE_4(" p_in_matrix: %p\n", p_in_matrix);
//     DNN_TRACE_4(" in_d1,in_d2: [%d,%d]\n", in_d1, in_d2);
//     DNN_TRACE_4(" p_kernels: %p\n", p_kernels);
//     DNN_TRACE_4(" p_biases: %p\n", p_biases);
//     DNN_TRACE_4(" n_maxout_kernels: %d\n", n_maxout_kernels);
//     DNN_TRACE_4(" k_d1,k_d2: [%d,%d]\n", k_d1, k_d2);
//     DNN_TRACE_4(" n_channels: %d\n", n_channels);
//     DNN_TRACE_4(" p_out_matrix: %p\n", p_out_matrix);
//     
//     DNN_ASSERT(k_d1 == 4);
//     DNN_ASSERT(k_d2 == 4);
//     DNN_ASSERT(n_maxout_kernels == 2);
//     DNN_ASSERT(n_channels == 32);
//     
//  // Dimensions etc.
//     const int N1 = in_d1;
//     const int N2 = in_d2;
//     const int D = n_channels;
//     const int K1 = k_d1;
//     const int K2 = k_d2;
//  const int M1 = OUT_DIM(in_d1, K1, 1);
//  const int M2 = OUT_DIM(in_d2, K2, 1);
//     
//  // Stack for storing next superpixel location.
//  int32_t* stk = (int32_t *)malloc(sizeof(int32_t) * M1);
//     
//     char *p_visited = (char *)malloc(sizeof(char) * N1 * N2);
//     
//  // Convolve 1x1 kernels only.
//  for (int krow = 0; krow < 4; krow++) {
//      for (int kcol = 0; kcol < 4; kcol++) {
//          // NOTE: Can optimize out this 'visited' matrix if using
//          // separate output layers for each 1x1 kernel; just check the 
//          // output for a non-default value.
//          //Matrix32 visited = Matrix32::Zero(M1, M2);
//             memset(p_visited, 0, sizeof(char) * N1 * N2);
//             
//          // Reset stack (this step is different in lossy version).
//          int stkptr = 0;
//          stk[stkptr++] = encode(krow, kcol);
// 
//          // Iterate across all superpixels.
//          while (stkptr) {
//              // Extract superpixel size and location.
//              int srow, scol, size_x, size_y;
//              
//              decode(stk[--stkptr], &srow, &scol);
//              
//              int s_d1_offset = D1_TO_OFFSET(srow, N2, 1);
//              int s_d2_offset = D2_TO_OFFSET(scol, 1);
// 
//                 uint32_t s_val = p_s_matrix[s_d1_offset + s_d2_offset];
//              decode(s_val, &size_x, &size_y);
// 
//              // Mark this position visited (see above)
//                 p_visited[s_d1_offset + s_d2_offset] = 1;
// 
//              // Only need to process depth once for this SP.
//              // Determine inner product down z-axis (AVX!)
//              
//                 int in_d1_offset = D1_TO_OFFSET(srow, N2, D);
//              int in_d2_offset = D2_TO_OFFSET(scol, D);
//              
//                 int is_first_inprod = 1;
//              float inprod_max = 0.0;
//                 for (int k_id = 0; k_id < 1; k_id++) {
//                     
//                     int cur_k_id_offset = k_id * DIMS_TO_SIZE(K1, K2, D);
//                     int cur_k_d1_offset = D1_TO_OFFSET(krow, K2, D);
//                     int cur_k_d2_offset = D2_TO_OFFSET(kcol, D);
//                     int cur_k_full_offset = cur_k_id_offset + cur_k_d1_offset + cur_k_d2_offset;
//                     
//                     float inprod = 0.0;    
//                     
//                     for (int d = 0; d < 32; d++) {
//                         
//                         float k_val = p_kernels[cur_k_full_offset + d];
//                         
//                         float in_val = p_in_matrix[in_d1_offset + in_d2_offset + d];
//                         
//                         inprod += k_val * in_val;
//                     }
//                     
//                     if ((krow == 0) && (kcol == 0)) {
//                         inprod += p_biases[k_id];
//                     }
//                     
//                     if (is_first_inprod) {
//                         is_first_inprod = 0;
//                         inprod_max = inprod;
//                         
//                     } else if (inprod > inprod_max) {
//                         inprod_max = inprod;
//                     }
//                     
//                     
//                 }
//                 
//              // Process superpixel to populate output image.
//              const int offset_x = srow + size_x;
//              const int offset_y = scol + size_y;
// 
//              // This step is different in lossy version.
//              
//                 int x_s_limit = M1 + krow;
//                 if (offset_x < (M1 + krow)) {
//                     x_s_limit = offset_x;
//                 }
//                 
//                 int y_s_limit = M2 + kcol;
//                 if (offset_y < (M2 + kcol)) {
//                     y_s_limit = offset_y;
//                 }
//                 
//              for (int row = srow; row < x_s_limit; row++) {
//                  for (int col = scol; col < y_s_limit; col++) {
//                      // This just adds to a single output matrix, but you 
//                      // could write to separate mats and reduce later.
//                      
//                         int cur_out_d1 = row - krow;
//                         int cur_out_d2 = col - kcol;
//                         
//                      int out_d1_offset = D1_TO_OFFSET(cur_out_d1, M2, 1);
//                      int out_d2_offset = D2_TO_OFFSET(cur_out_d2, 1);
//                         
//                         p_out_matrix[out_d1_offset + out_d2_offset] += inprod_max;
//                  }
//              }
// 
//              // Add next superpixels to stack if unvisited.
//              // This step is different in lossy version.
//              int x_s_d1_offset = D1_TO_OFFSET(offset_x, N2, 1);
//              int x_s_d2_offset = D2_TO_OFFSET(scol, 1);
//              
//              int y_s_d1_offset = D1_TO_OFFSET(srow, N2, 1);
//              int y_s_d2_offset = D2_TO_OFFSET(offset_y, 1);
//                 
//              if (((offset_x < N1) && (p_visited[x_s_d1_offset + x_s_d2_offset] == 0)) &&
//                  (scol == kcol || (p_mask_matrix[x_s_d1_offset + x_s_d2_offset] != 0))) { // Check for SP.
//                  stk[stkptr++] = encode(offset_x, scol);
//              }
//              
//              if (((offset_y < N2) && (p_visited[y_s_d1_offset + y_s_d2_offset] == 0)) &&
//                  (srow == krow || (p_mask_matrix[y_s_d1_offset + y_s_d2_offset] != 0))) { // Check for SP.
//                  stk[stkptr++] = encode(srow, offset_y);
//              }
//                 
//          }
//      }
//  }
//  
//  DNN_TRACE_4("finish %s\n","");
// }
// 
