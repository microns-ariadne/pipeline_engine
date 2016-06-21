////////////////////////////////////////////////////////////////////////////////
// tensorDecomp.cpp
// David Budden
// Decomposition and compression for tensor-represented image stacks.

#include "tensor_decomp.h"

//#include <cmath>
//#include <iostream>
//#include <limits>

////////////////////////////////////////////////////////////////////////////////
// Helper functions for encoding/decoding 2D points (sizes, locations, etc.)
////////////////////////////////////////////////////////////////////////////////
int64_t encode64(const int x1, const int y1, const int x2, const int y2) {
	int64_t out = ((int64_t)x1 & 0xffff) << 48;
	out |= ((int64_t)y1 & 0xffff) << 32;
	out |= ((int64_t)x2 & 0xffff) << 16;
	return out | ((int64_t)y2 & 0xffff);
}
void decode64(const int64_t in, int *p_x1, int *p_y1, int *p_x2, int *p_y2) {
	*p_x1 = (in >> 48) & 0xffff;
	*p_y1 = (in >> 32) & 0xffff;
	*p_x2 = (in >> 16) & 0xffff;
	*p_y2 = in & 0xffff;
}
int32_t encode32(const int x, const int y) {
	int32_t out = (x & 0xffff) << 16;
	return out | (y & 0xffff);
}
void decode32(const int32_t in, int *p_x, int *p_y ) {
	*p_x = (in >> 16) & 0xffff;
	*p_y = in & 0xffff;
}

////////////////////////////////////////////////////////////////////////////////
// split
// Determine if a 3D tile/superpixel requires splitting.
////////////////////////////////////////////////////////////////////////////////
int split(
    const float * __restrict__ p_in_matrix,
    const int in_d1,
    const int in_d2,
    const int in_d3,
    const int srow, 
    const int scol,
	const int M1, 
	const int M2, 
	const float n_threshold) {
	    
	if (M1 <= 1 || M2 <= 1) {
		return 0;	// Cannot split atomic pixels.
	}
    
    const int N2 = in_d2;
    const int D = in_d3;
    
	// Split if any layer in stack requires splitting.
	// (Necessary for AVX alignment).
    
    int cur_in_d1_offset = D1_TO_OFFSET(srow, N2, D);
    int cur_in_d2_offset = D2_TO_OFFSET(scol, D);
    
    float min = p_in_matrix[cur_in_d1_offset + cur_in_d2_offset];
    float max = p_in_matrix[cur_in_d1_offset + cur_in_d2_offset];
    
    for (int row = srow; row < srow + M1; row++) {
		for (int col = scol; col < scol + M2; col++) {
            
            int cur_in_d1_offset = D1_TO_OFFSET(row, N2, D);
            int cur_in_d2_offset = D2_TO_OFFSET(col, D);
            
            int n_depth = D;
            // FIX FIX
            if (n_threshold == QTD_THREASHOLD_INPUT) {
                n_depth = 3;
            }
            
	        for (int ch_id = 0; ch_id < n_depth; ch_id++) {
                
                float val = p_in_matrix[cur_in_d1_offset + cur_in_d2_offset + ch_id];

				min = val < min ? val : min;
				max = val > max ? val : max;
                
				if ((max - min) > n_threshold) {
					//printf("min/max: %f %f\n", min, max);
					return 1;
				}
            }
	    }
    }
	
	return 0;
	
}

////////////////////////////////////////////////////////////////////////////////
// qtdDecompose
// Optimized, general quad tree decomposition.
////////////////////////////////////////////////////////////////////////////////
void qtd_decompose(
    const float * __restrict__ p_in_matrix,
    const int in_d1,
    const int in_d2,
    const int in_d3,
    const float n_threshold, 
    uint32_t * __restrict__ p_out_matrix,
    uint32_t * __restrict__ p_mask_matrix) {
	    
	// }
	// 
	// void qtdDecompose(const Eigen::Tensor<float, 3>& in, const int THRESH, 
	//     Matrix32& out, Matrix32& mask) { 
    
	const int N1 = in_d1;
	const int N2 = in_d2;
    
	// Upper bound on stack size is 3 * log2(max{N1, N2}) + 1 )
	// (Don't ask me to prove it)
    int64_t* stk = (int64_t *)malloc(sizeof(int64_t) * N1 * N2);
	int stkptr = 0;
	stk[stkptr++] = encode64(0, 0, N1, N2);

	// Perform decomposition.
	while (stkptr) {
		// Pop and decode superpixel.
		// Superpixel encoding convention: [ (pos_x, pos_y), (size_x, size_y) ]
		int srow, scol, M1, M2;
		decode64(stk[--stkptr], &srow, &scol, &M1, &M2);

		// If split required, add descendants to stack.
		if (split(p_in_matrix, in_d1, in_d2, in_d3, srow, scol, M1, M2, n_threshold)) {
			// Where are the next superpixels?
			const int x1 = M1 / 2;
			const int y1 = M2 / 2;
			const int x2 = x1 + M1 % 2;
			const int y2 = y1 + M2 % 2;

			stk[stkptr++] = encode64(srow, scol, x2, y2);
			stk[stkptr++] = encode64(srow + x2, scol, x1, y2);
			stk[stkptr++] = encode64(srow, scol + y2, x2, y1);
			stk[stkptr++] = encode64(srow + x2, scol + y2, x1, y1);
		} else { 
			// Write this superpixel to output.
			int cur_d1_offset = D1_TO_OFFSET(srow, N2, 1);
    	    int cur_d2_offset = D2_TO_OFFSET(scol, 1);
            
            p_mask_matrix[cur_d1_offset + cur_d2_offset] = 1;

            for (int row = srow; row < srow + M1; row++) {
                for (int col = scol; col < scol + M2; col++) {

                    int cur_d1_offset = D1_TO_OFFSET(row, N2, 1);
                    int cur_d2_offset = D2_TO_OFFSET(col, 1);

                    p_out_matrix[cur_d1_offset + cur_d2_offset] = encode32(M1 + srow - row, 
                        M2 + scol - col);
                }
            }
            
            // MASK
            
			// cur_d1_offset = D1_TO_OFFSET(srow, N2, 1);
			//           cur_d2_offset = D2_TO_OFFSET(scol, 1);
			//           
			//             p_out_matrix[cur_d1_offset + cur_d2_offset] = 0;
			//             
			//             
			//             for (int col = scol + 1; col < scol + M2; col++) {
			//                 cur_d1_offset = D1_TO_OFFSET(srow, N2, 1);
			//                 cur_d2_offset = D2_TO_OFFSET(col, 1);
			//                 p_out_matrix[cur_d1_offset + cur_d2_offset] = 1;
			//             }
			//             
			//             for (int row = srow + 1; row < srow + M1; row++) {
			//                 cur_d1_offset = D1_TO_OFFSET(row, N2, 1);
			//                 cur_d2_offset = D2_TO_OFFSET(scol, 1);
			//                 p_out_matrix[cur_d1_offset + cur_d2_offset] = 2;
			//             }
			// 
			//           
			//           for (int row = srow + 1; row < srow + M1; row++) {
			//               for (int col = scol + 1; col < scol + M2; col++) {
			//                   cur_d1_offset = D1_TO_OFFSET(row, N2, 1);
			//                   cur_d2_offset = D2_TO_OFFSET(col, 1);
			//                     p_out_matrix[cur_d1_offset + cur_d2_offset] = 3;
			//               }
			//           }
			
			// MASK
			
		}
	}
}
