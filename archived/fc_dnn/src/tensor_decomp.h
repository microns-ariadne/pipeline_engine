////////////////////////////////////////////////////////////////////////////////
// tensorDecomp.h
// David Budden
// Decomposition and compression for tensor-represented image stacks.

#ifndef TENSOR_DECOMP_H_
#define TENSOR_DECOMP_H_

#include "common.h"

////////////////////////////////////////////////////////////////////////////////
// qtdDecompose
// Optimized, general quad tree decomposition.
void qtd_decompose(
    const float * __restrict__ p_in_matrix,
    const int in_d1,
    const int in_d2,
    const int in_d3,
    const float n_threshold, 
    uint32_t * __restrict__ p_out_matrix,
    uint32_t * __restrict__ p_mask_matrix);
    
#endif /* TENSOR_DECOMP_H_ */