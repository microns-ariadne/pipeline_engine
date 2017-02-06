////////////////////////////////////////////////////////////////////////////////
// tensorConv.h
// David Budden
// Different methods for convolving tensors in a FCNN context.

#ifndef TENSOR_CONV_H_
#define TENSOR_CONV_H_

#include "common.h"

////////////////////////////////////////////////////////////////////////////////
// convolve_by_parts
// Convolution by parts (1x1 kernels), lossless.
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
    float * __restrict__ p_out_matrix);

#endif /* TENSOR_CONV_H_ */
