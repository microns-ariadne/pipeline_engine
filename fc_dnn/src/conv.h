/////////////////////////////////////////////////////////////////////////////////////////
//
//
//
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef CONV_H
#define CONV_H 1

/////////////////////////////////////////////////////////////////////////////////////////
// INCLUDES
/////////////////////////////////////////////////////////////////////////////////////////
#include "tensor_decomp.h"
#include "tensor_conv.h"

/////////////////////////////////////////////////////////////////////////////////////////
// DEFINES 
/////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////
// TYPES
/////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////
// EXTERNAL FUNCTIONS
/////////////////////////////////////////////////////////////////////////////////////////
void conv3D(
    const float * __restrict__ p_matrix, 
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
    float * __restrict__ p_res_matrix);

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
    float * __restrict__ p_out_matrix);

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
    float * __restrict__ p_out_matrix);

/////////////////////////////////////////////////////////////////////////////////////////
// EXTERNAL MACROS
/////////////////////////////////////////////////////////////////////////////////////////

#endif // CONV_H
