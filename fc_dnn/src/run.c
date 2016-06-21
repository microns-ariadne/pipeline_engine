/////////////////////////////////////////////////////////////////////////////////////////
//
//
//
/////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////
// INCLUDES
/////////////////////////////////////////////////////////////////////////////////////////
#include "dnn.h"

/////////////////////////////////////////////////////////////////////////////////////////
// DEFINES
/////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////
// TYPES
/////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////
// GLOBALS
/////////////////////////////////////////////////////////////////////////////////////////
/*layer_t net[] = {
// l_id,  op_type,       n_channels, n_stride, n_maxout, { d1, d2, d3, n_matrices, n_size, p_data}, init_filename 
 { 1,     OP_CONV,       32,          1,        2,        { 4,  4,  1,  64,          -1,     NULL  }, (char *)"./test_49_w2_maxout_full/conv1A_conv1B_kernels.txt" },
 { 2,     OP_POOL,       32,          2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 3,     OP_CONV,       32,         1,        2,        { 4,  4,  32,  64,         -1,     NULL  }, (char *)"./test_49_w2_maxout_full/conv2A_conv2B_kernels.txt" },
 { 4,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 5,     OP_CONV,       32,         1,        2,        { 4,  4,  32, 64,         -1,     NULL  }, (char *)"./test_49_w2_maxout_full/conv3A_conv3B_kernels.txt" },
 { 6,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 7,     OP_CONV,       2,          1,        1,        { 4,  4,  32, 2,          -1,     NULL  }, (char *)"./test_49_w2_maxout_full/ip_conv_kernels.txt" },
 { 8,     OP_SOFTMAX,    2,          1,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
 { 8,     OP_INTERWEAVE, 64,         8,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
};*/

/*layer_t net[] = {
// l_id,  op_type,       n_channels, n_stride, n_maxout, { d1, d2, d3, n_matrices, n_size, p_data}, init_filename 
 { 1,     OP_CONV,       32,          1,        2,        { 8,  8,  3,  64,          -1,     NULL  }, (char *)"./test_49_w2_maxout_full_3D_2/conv1A_conv1B_kernels.txt" },
 { 2,     OP_POOL,       32,          2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 3,     OP_CONV,       64,         1,        2,        { 4,  4,  32,  128,         -1,     NULL  }, (char *)"./test_49_w2_maxout_full_3D_2/conv2A_conv2B_kernels.txt" },
 { 4,     OP_POOL,       64,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 5,     OP_CONV,       128,         1,        2,        { 4,  4,  64, 256,         -1,     NULL  }, (char *)"./test_49_w2_maxout_full_3D_2/conv3A_conv3B_kernels.txt" },
 { 6,     OP_POOL,       128,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 7,     OP_CONV,       2,          1,        1,        { 3,  3,  128, 2,          -1,     NULL  }, (char *)"./test_49_w2_maxout_full_3D_2/ip_conv_kernels.txt" },
 { 8,     OP_SOFTMAX,    2,          1,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
 { 8,     OP_INTERWEAVE, 64,         8,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
};*/

/*layer_t net[] = {
// l_id,  op_type,       n_channels, n_stride, n_maxout, { d1, d2, d3, n_matrices, n_size, p_data}, init_filename 
 { 1,     OP_CONV,       32,          1,        2,        { 4,  4,  1,  64,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/test_49_w2_4M/conv1A_1B_kernels.txt" },
 { 2,     OP_POOL,       32,          2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 3,     OP_CONV,       32,         1,        2,        { 4,  4,  32,  64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/test_49_w2_4M/conv2A_2B_kernels.txt" },
 { 4,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 5,     OP_CONV,       32,         1,        2,        { 4,  4,  32, 64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/test_49_w2_4M/conv3A_3B_kernels.txt" },
 { 6,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 7,     OP_CONV,       2,          1,        1,        { 4,  4,  32,  2,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/test_49_w2_4M/ip_conv_kernels.txt" },
 { 8,     OP_SOFTMAX,    2,          1,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
 { 8,     OP_INTERWEAVE, 64,         8,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
};*/

// 8F
/*layer_t net[] = {
// l_id,  op_type,       n_channels, n_stride, n_maxout, { d1, d2, d3, n_matrices, n_size, p_data}, init_filename 
 { 1,     OP_CONV,       8,          1,        2,        { 4,  4,  1,  16,          -1,     NULL  }, (char *)"./test_49_w2_4M_8f/conv1A_conv1B_kernels.txt" },
 { 2,     OP_POOL,       8,          2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 3,     OP_CONV,       8,         1,        2,        { 4,  4,  8,  16,         -1,     NULL  }, (char *)"./test_49_w2_4M_8f/conv2A_conv2B_kernels.txt" },
 { 4,     OP_POOL,       8,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 5,     OP_CONV,       8,         1,        2,        { 4,  4,  8, 16,         -1,     NULL  }, (char *)"./test_49_w2_4M_8f/conv3A_conv3B_kernels.txt" },
 { 6,     OP_POOL,       8,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 7,     OP_CONV,       2,          1,        1,        { 4,  4,  8,  2,          -1,     NULL  }, (char *)"./test_49_w2_4M_8f/ip_conv_kernels.txt" },
 { 8,     OP_SOFTMAX,    2,          1,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
 { 8,     OP_INTERWEAVE, 64,         8,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
};*/

// OSDI 8f
// 8F - P3
/*layer_t net[] = {
// l_id,  op_type,       n_channels, n_stride, n_maxout, { d1, d2, d3, n_matrices, n_size, p_data}, init_filename 
 { 1,     OP_CONV,       8,          1,        2,        { 4,  4,  3,  16,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/P3_49_w2_4M_8f_3D/conv1A_1B_kernels.txt" },
 { 2,     OP_POOL,       8,          2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 3,     OP_CONV,       8,         1,        2,        { 4,  4,  8,  16,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/P3_49_w2_4M_8f_3D/conv2A_2B_kernels.txt" },
 { 4,     OP_POOL,       8,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 5,     OP_CONV,       8,         1,        2,        { 4,  4,  8, 16,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/P3_49_w2_4M_8f_3D/conv3A_3B_kernels.txt" },
 { 6,     OP_POOL,       8,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 7,     OP_CONV,       2,          1,        1,        { 4,  4,  8,  2,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/P3_49_w2_4M_8f_3D/ip_conv_kernels.txt" },
 { 8,     OP_SOFTMAX,    2,          1,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
 { 9,     OP_INTERWEAVE, 64,         8,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
};*/

// OSDI 32F
// 32F - P3 - errors
/*layer_t net[] = {
// l_id,  op_type,       n_channels, n_stride, n_maxout, { d1, d2, d3, n_matrices, n_size, p_data}, init_filename 
 { 1,     OP_CONV,       32,          1,        2,        { 4,  4,  3,  64,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/P3_49_w2_4M_8f_3D_errors/conv1A_1B_kernels.txt" },
 { 2,     OP_POOL,       32,          2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 3,     OP_CONV,       32,         1,        2,        { 4,  4,  32,  64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/P3_49_w2_4M_8f_3D_errors/conv2A_2B_kernels.txt" },
 { 4,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 5,     OP_CONV,       32,         1,        2,        { 4,  4,  32, 64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/P3_49_w2_4M_8f_3D_errors/conv3A_3B_kernels.txt" },
 { 6,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 7,     OP_CONV,       2,          1,        1,        { 4,  4,  32, 2,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/P3_49_w2_4M_8f_3D_errors/ip_conv_kernels.txt" },
 { 8,     OP_SOFTMAX,    2,          1,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
 { 8,     OP_INTERWEAVE, 64,         8,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
};*/

// 32F - P3 - errors NEW 2
/*layer_t net[] = {
// l_id,  op_type,       n_channels, n_stride, n_maxout, { d1, d2, d3, n_matrices, n_size, p_data}, init_filename 
 { 1,     OP_CONV,       32,          1,        2,        { 4,  4,  3,  64,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/P3_49_w2_4M_8f_3D_errors_NEW_2/conv1A_1B_kernels.txt" },
 { 2,     OP_POOL,       32,          2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 3,     OP_CONV,       32,         1,        2,        { 4,  4,  32,  64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/P3_49_w2_4M_8f_3D_errors_NEW_2/conv2A_2B_kernels.txt" },
 { 4,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 5,     OP_CONV,       32,         1,        2,        { 4,  4,  32, 64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/P3_49_w2_4M_8f_3D_errors_NEW_2/conv3A_3B_kernels.txt" },
 { 6,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 7,     OP_CONV,       2,          1,        1,        { 4,  4,  32, 2,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/P3_49_w2_4M_8f_3D_errors_NEW_2/ip_conv_kernels.txt" },
 { 8,     OP_SOFTMAX,    2,          1,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
 { 8,     OP_INTERWEAVE, 64,         8,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
};*/

// 32F - P3 - errors blacked
/*layer_t net[] = {
// l_id,  op_type,       n_channels, n_stride, n_maxout, { d1, d2, d3, n_matrices, n_size, p_data}, init_filename 
 { 1,     OP_CONV,       32,          1,        2,        { 4,  4,  3,  64,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/P3_49_w2_4M_8f_3D_errors_blacked/conv1A_1B_kernels.txt" },
 { 2,     OP_POOL,       32,          2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 3,     OP_CONV,       32,         1,        2,        { 4,  4,  32,  64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/P3_49_w2_4M_8f_3D_errors_blacked/conv2A_2B_kernels.txt" },
 { 4,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 5,     OP_CONV,       32,         1,        2,        { 4,  4,  32, 64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/P3_49_w2_4M_8f_3D_errors_blacked/conv3A_3B_kernels.txt" },
 { 6,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 7,     OP_CONV,       2,          1,        1,        { 4,  4,  32, 2,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/P3_49_w2_4M_8f_3D_errors_blacked/ip_conv_kernels.txt" },
 { 8,     OP_SOFTMAX,    2,          1,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
 { 8,     OP_INTERWEAVE, 64,         8,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
};*/

// 32F - P3 - 2D errors FM
/*layer_t net[] = {
// l_id,  op_type,       n_channels, n_stride, n_maxout, { d1, d2, d3, n_matrices, n_size, p_data}, init_filename 
 { 1,     OP_CONV,       32,          1,        2,        { 4,  4,  1,  64,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/P3_49_w2_4M_8f_3D_2D_errors_FM/conv1A_1B_kernels.txt" },
 { 2,     OP_POOL,       32,          2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 3,     OP_CONV,       32,         1,        2,        { 4,  4,  32,  64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/P3_49_w2_4M_8f_3D_2D_errors_FM/conv2A_2B_kernels.txt" },
 { 4,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 5,     OP_CONV,       32,         1,        2,        { 4,  4,  32, 64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/P3_49_w2_4M_8f_3D_2D_errors_FM/conv3A_3B_kernels.txt" },
 { 6,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 7,     OP_CONV,       2,          1,        1,        { 4,  4,  32, 2,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/P3_49_w2_4M_8f_3D_2D_errors_FM/ip_conv_kernels.txt" },
 { 8,     OP_SOFTMAX,    2,          1,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
 { 8,     OP_INTERWEAVE, 64,         8,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
};*/

// 32F - P3 - 3D errors FM
/*layer_t net[] = {
// l_id,  op_type,       n_channels, n_stride, n_maxout, { d1, d2, d3, n_matrices, n_size, p_data}, init_filename 
 { 1,     OP_CONV,       32,          1,        2,        { 4,  4,  3,  64,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/P3_49_w2_4M_3D_errors_FM/conv1A_1B_kernels.txt" },
 { 2,     OP_POOL,       32,          2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 3,     OP_CONV,       32,         1,        2,        { 4,  4,  32,  64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/P3_49_w2_4M_3D_errors_FM/conv2A_2B_kernels.txt" },
 { 4,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 5,     OP_CONV,       32,         1,        2,        { 4,  4,  32, 64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/P3_49_w2_4M_3D_errors_FM/conv3A_3B_kernels.txt" },
 { 6,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 7,     OP_CONV,       2,          1,        1,        { 4,  4,  32, 2,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/P3_49_w2_4M_3D_errors_FM/ip_conv_kernels.txt" },
 { 8,     OP_SOFTMAX,    2,          1,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
 { 8,     OP_INTERWEAVE, 64,         8,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
};*/

// 32F - P3 - 3D blur and shift errors FM (bc)
/*layer_t net[] = {
// l_id,  op_type,       n_channels, n_stride, n_maxout, { d1, d2, d3, n_matrices, n_size, p_data}, init_filename 
 { 1,     OP_CONV,       32,          1,        2,        { 4,  4,  3,  64,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/P3_3D_blur_shift_errors_FM/conv1A_1B_kernels.txt" },
 { 2,     OP_POOL,       32,          2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 3,     OP_CONV,       32,         1,        2,        { 4,  4,  32,  64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/P3_3D_blur_shift_errors_FM/conv2A_2B_kernels.txt" },
 { 4,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 5,     OP_CONV,       32,         1,        2,        { 4,  4,  32, 64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/P3_3D_blur_shift_errors_FM/conv3A_3B_kernels.txt" },
 { 6,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 7,     OP_CONV,       2,          1,        1,        { 4,  4,  32, 2,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/P3_3D_blur_shift_errors_FM/ip_conv_kernels.txt" },
 { 8,     OP_SOFTMAX,    2,          1,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
 { 8,     OP_INTERWEAVE, 64,         8,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
};*/

// 32F - P3 - 2D FM
/*layer_t net[] = {
// l_id,  op_type,       n_channels, n_stride, n_maxout, { d1, d2, d3, n_matrices, n_size, p_data}, init_filename 
 { 1,     OP_CONV,       32,          1,        2,        { 4,  4,  1,  64,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/P3_3D_2D_FM/conv1A_1B_kernels.txt" },
 { 2,     OP_POOL,       32,          2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 3,     OP_CONV,       32,         1,        2,        { 4,  4,  32,  64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/P3_3D_2D_FM/conv2A_2B_kernels.txt" },
 { 4,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 5,     OP_CONV,       32,         1,        2,        { 4,  4,  32, 64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/P3_3D_2D_FM/conv3A_3B_kernels.txt" },
 { 6,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 7,     OP_CONV,       2,          1,        1,        { 4,  4,  32, 2,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/P3_3D_2D_FM/ip_conv_kernels.txt" },
 { 8,     OP_SOFTMAX,    2,          1,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
 { 8,     OP_INTERWEAVE, 64,         8,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
};*/

// 32F - K11_AC3_from_OCP_cc_3D_PAD_FM25
/*layer_t net[] = {
// l_id,  op_type,       n_channels, n_stride, n_maxout, { d1, d2, d3, n_matrices, n_size, p_data}, init_filename 
 { 1,     OP_CONV,       32,          1,        2,        { 4,  4,  3,  64,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11_AC3_from_OCP_cc_3D_PAD_FM25/conv1A_1B_kernels.txt" },
 { 2,     OP_POOL,       32,          2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 3,     OP_CONV,       32,         1,        2,        { 4,  4,  32,  64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11_AC3_from_OCP_cc_3D_PAD_FM25/conv2A_2B_kernels.txt" },
 { 4,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 5,     OP_CONV,       32,         1,        2,        { 4,  4,  32, 64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11_AC3_from_OCP_cc_3D_PAD_FM25/conv3A_3B_kernels.txt" },
 { 6,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 7,     OP_CONV,       2,          1,        1,        { 4,  4,  32, 2,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11_AC3_from_OCP_cc_3D_PAD_FM25/ip_conv_kernels.txt" },
 { 8,     OP_SOFTMAX,    2,          1,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
 { 8,     OP_INTERWEAVE, 64,         8,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
};*/

// 32F - K11_S1_AC3_256_cc_3D_PAD_FM25
/*layer_t net[] = {
// l_id,  op_type,       n_channels, n_stride, n_maxout, { d1, d2, d3, n_matrices, n_size, p_data}, init_filename 
 { 1,     OP_CONV,       32,          1,        2,        { 4,  4,  3,  64,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11_S1_AC3_256_cc_3D_PAD_FM25/conv1A_1B_kernels.txt" },
 { 2,     OP_POOL,       32,          2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 3,     OP_CONV,       32,         1,        2,        { 4,  4,  32,  64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11_S1_AC3_256_cc_3D_PAD_FM25/conv2A_2B_kernels.txt" },
 { 4,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 5,     OP_CONV,       32,         1,        2,        { 4,  4,  32, 64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11_S1_AC3_256_cc_3D_PAD_FM25/conv3A_3B_kernels.txt" },
 { 6,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 7,     OP_CONV,       2,          1,        1,        { 4,  4,  32, 2,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11_S1_AC3_256_cc_3D_PAD_FM25/ip_conv_kernels.txt" },
 { 8,     OP_SOFTMAX,    2,          1,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
 { 8,     OP_INTERWEAVE, 64,         8,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
};*/

// 32F - K11_S1_AC3_256_cc_3D_PAD
/*layer_t net[] = {
// l_id,  op_type,       n_channels, n_stride, n_maxout, { d1, d2, d3, n_matrices, n_size, p_data}, init_filename 
 { 1,     OP_CONV,       32,          1,        2,        { 4,  4,  3,  64,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11_S1_AC3_256_cc_3D_PAD/conv1A_1B_kernels.txt" },
 { 2,     OP_POOL,       32,          2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 3,     OP_CONV,       32,         1,        2,        { 4,  4,  32,  64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11_S1_AC3_256_cc_3D_PAD/conv2A_2B_kernels.txt" },
 { 4,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 5,     OP_CONV,       32,         1,        2,        { 4,  4,  32, 64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11_S1_AC3_256_cc_3D_PAD/conv3A_3B_kernels.txt" },
 { 6,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 7,     OP_CONV,       2,          1,        1,        { 4,  4,  32, 2,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11_S1_AC3_256_cc_3D_PAD/ip_conv_kernels.txt" },
 { 8,     OP_SOFTMAX,    2,          1,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
 { 9,     OP_INTERWEAVE, 64,         8,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
};*/

// 32F - K11_S1_AC3_256_cc_3D_PAD - LeeK
layer_t net[] = {
// l_id,  op_type,       n_channels, n_stride, n_maxout, { d1, d2, d3, n_matrices, n_size, p_data}, init_filename 
 { 1,     OP_CONV,       32,          1,        2,        { 4,  4,  3,  64,          -1,     NULL  }, (char *)"./K11_S1_AC3_256_cc_3D_PAD/conv1A_1B_kernels.txt" },
 { 2,     OP_POOL,       32,          2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 3,     OP_CONV,       32,         1,        2,        { 4,  4,  32,  64,         -1,     NULL  }, (char *)"./K11_S1_AC3_256_cc_3D_PAD/conv2A_2B_kernels.txt" },
 { 4,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 5,     OP_CONV,       32,         1,        2,        { 4,  4,  32, 64,         -1,     NULL  }, (char *)"./K11_S1_AC3_256_cc_3D_PAD/conv3A_3B_kernels.txt" },
 { 6,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 7,     OP_CONV,       2,          1,        1,        { 4,  4,  32, 2,          -1,     NULL  }, (char *)"./K11_S1_AC3_256_cc_3D_PAD/ip_conv_kernels.txt" },
 { 8,     OP_SOFTMAX,    2,          1,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
 { 9,     OP_INTERWEAVE, 64,         8,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
};

// 32F - K11_S1_AC3_256_cc_3D_PAD_sub_sample_2
/*layer_t net[] = {
// l_id,  op_type,       n_channels, n_stride, n_maxout, { d1, d2, d3, n_matrices, n_size, p_data}, init_filename 
 { 1,     OP_CONV,       32,          1,        2,        { 4,  4,  3,  64,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11_S1_AC3_256_cc_3D_PAD_sub_sample_2/conv1A_1B_kernels.txt" },
 { 2,     OP_POOL,       32,          2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 3,     OP_CONV,       32,         1,        2,        { 4,  4,  32,  64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11_S1_AC3_256_cc_3D_PAD_sub_sample_2/conv2A_2B_kernels.txt" },
 { 4,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 5,     OP_CONV,       32,         1,        2,        { 4,  4,  32, 64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11_S1_AC3_256_cc_3D_PAD_sub_sample_2/conv3A_3B_kernels.txt" },
 { 6,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 7,     OP_CONV,       2,          1,        1,        { 4,  4,  32, 2,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11_S1_AC3_256_cc_3D_PAD_sub_sample_2/ip_conv_kernels.txt" },
 { 8,     OP_SOFTMAX,    2,          1,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
 { 8,     OP_INTERWEAVE, 64,         8,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
};*/

// 32F - K11_S1_AC3_256_cc_53_dist_4_3D_PAD
/*layer_t net[] = {
// // l_id,  op_type,       n_channels, n_stride, n_maxout, { d1, d2, d3, n_matrices, n_size, p_data}, init_filename 
 { 1,     OP_CONV,       32,          1,        2,        { 4,  4,  3,  64,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11_S1_AC3_256_cc_53_dist_4_3D_PAD/conv1A_1B_kernels.txt" },
 { 2,     OP_POOL,       32,          2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 3,     OP_CONV,       32,         1,        2,        { 4,  4,  32,  64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11_S1_AC3_256_cc_53_dist_4_3D_PAD/conv2A_2B_kernels.txt" },
 { 4,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 5,     OP_CONV,       32,         1,        2,        { 4,  4,  32, 64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11_S1_AC3_256_cc_53_dist_4_3D_PAD/conv3A_3B_kernels.txt" },
 { 6,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 7,     OP_CONV,       4,          1,        1,        { 4,  4,  32, 4,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11_S1_AC3_256_cc_53_dist_4_3D_PAD/ip_conv_kernels.txt" },
 { 8,     OP_SOFTMAX,    4,          1,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
 { 9,     OP_INTERWEAVE, 64,         8,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
};*/

// 32F - Harvardnet_dist_4_53x53_3D_4M_K11_3nm_AC3
/*layer_t net[] = {
// l_id,  op_type,       n_channels, n_stride, n_maxout, { d1, d2, d3, n_matrices, n_size, p_data}, init_filename 
 { 1,     OP_CONV,       32,          1,        2,        { 4,  4,  3,  64,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/extracted_kernels_harvardnet_dist_4_53x53_3D_4M_K11_3nm_AC3/conv1A_1B_kernels.txt" },
 { 2,     OP_POOL,       32,          2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 3,     OP_CONV,       32,         1,        2,        { 4,  4,  32,  64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/extracted_kernels_harvardnet_dist_4_53x53_3D_4M_K11_3nm_AC3/conv2A_2B_kernels.txt" },
 { 4,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 5,     OP_CONV,       32,         1,        2,        { 4,  4,  32, 64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/extracted_kernels_harvardnet_dist_4_53x53_3D_4M_K11_3nm_AC3/conv3A_3B_kernels.txt" },
 { 6,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 7,     OP_CONV,       4,          1,        1,        { 4,  4,  32, 4,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/extracted_kernels_harvardnet_dist_4_53x53_3D_4M_K11_3nm_AC3/ip_conv_kernels.txt" },
 { 8,     OP_SOFTMAX,    4,          1,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
 { 8,     OP_INTERWEAVE, 64,         8,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
};*/

// 32F - harvardnet_dist_4_53x53_3D_4M_K11_AC4
/*layer_t net[] = {
// l_id,  op_type,       n_channels, n_stride, n_maxout, { d1, d2, d3, n_matrices, n_size, p_data}, init_filename 
 { 1,     OP_CONV,       32,          1,        2,        { 4,  4,  3,  64,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11_6nm_harvardnet_dist_4_53x53_3D_4M_K11_AC4/conv_1A_1B_kernels.txt" },
 { 2,     OP_POOL,       32,          2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 3,     OP_CONV,       32,         1,        2,        { 4,  4,  32,  64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11_6nm_harvardnet_dist_4_53x53_3D_4M_K11_AC4/conv_2A_2B_kernels.txt" },
 { 4,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 5,     OP_CONV,       32,         1,        2,        { 4,  4,  32, 64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11_6nm_harvardnet_dist_4_53x53_3D_4M_K11_AC4/conv_3A_3B_kernels.txt" },
 { 6,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 7,     OP_CONV,       4,          1,        1,        { 4,  4,  32, 4,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11_6nm_harvardnet_dist_4_53x53_3D_4M_K11_AC4/ip_conv_kernels.txt" },
 { 8,     OP_SOFTMAX,    4,          1,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
 { 8,     OP_INTERWEAVE, 64,         8,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
};*/

// 32F - K11-S1-6nm-harvardnet_dist_4_53x53_3D_4M_K11_AC3_and_AC4_blacked
/*layer_t net[] = {
// l_id,  op_type,       n_channels, n_stride, n_maxout, { d1, d2, d3, n_matrices, n_size, p_data}, init_filename 
 { 1,     OP_CONV,       32,          1,        2,        { 4,  4,  3,  64,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-6nm-harvardnet_dist_4_53x53_3D_4M_K11_AC3_and_AC4_blacked/conv_1A_1B_kernels.txt" },
 { 2,     OP_POOL,       32,          2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 3,     OP_CONV,       32,         1,        2,        { 4,  4,  32,  64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-6nm-harvardnet_dist_4_53x53_3D_4M_K11_AC3_and_AC4_blacked/conv_2A_2B_kernels.txt" },
 { 4,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 5,     OP_CONV,       32,         1,        2,        { 4,  4,  32, 64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-6nm-harvardnet_dist_4_53x53_3D_4M_K11_AC3_and_AC4_blacked/conv_3A_3B_kernels.txt" },
 { 6,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 7,     OP_CONV,       4,          1,        1,        { 4,  4,  32, 4,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-6nm-harvardnet_dist_4_53x53_3D_4M_K11_AC3_and_AC4_blacked/ip_conv_kernels.txt" },
 { 8,     OP_SOFTMAX,    4,          1,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
 { 9,     OP_INTERWEAVE, 64,         8,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
};*/

// 32F - K11-S1-6nm-harvardnet_dist_4_53x53_3D_4M_K11_AC3_and_AC4_extra_GT
/*layer_t net[] = {
// l_id,  op_type,       n_channels, n_stride, n_maxout, { d1, d2, d3, n_matrices, n_size, p_data}, init_filename 
 { 1,     OP_CONV,       32,          1,        2,        { 4,  4,  3,  64,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-6nm-harvardnet_dist_4_53x53_3D_4M_K11_AC3_and_AC4_extra_GT/conv_1A_1B_kernels.txt" },
 { 2,     OP_POOL,       32,          2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 3,     OP_CONV,       32,         1,        2,        { 4,  4,  32,  64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-6nm-harvardnet_dist_4_53x53_3D_4M_K11_AC3_and_AC4_extra_GT/conv_2A_2B_kernels.txt" },
 { 4,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 5,     OP_CONV,       32,         1,        2,        { 4,  4,  32, 64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-6nm-harvardnet_dist_4_53x53_3D_4M_K11_AC3_and_AC4_extra_GT/conv_3A_3B_kernels.txt" },
 { 6,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 7,     OP_CONV,       4,          1,        1,        { 4,  4,  32, 4,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-6nm-harvardnet_dist_4_53x53_3D_4M_K11_AC3_and_AC4_extra_GT/ip_conv_kernels.txt" },
 { 8,     OP_SOFTMAX,    4,          1,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
 { 9,     OP_INTERWEAVE, 64,         8,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
};*/

// 32F - K11-S1-6nm-harvardnet_w2_53x53_3D_4M_K11_AC4
/*layer_t net[] = {
// l_id,  op_type,       n_channels, n_stride, n_maxout, { d1, d2, d3, n_matrices, n_size, p_data}, init_filename 
 { 1,     OP_CONV,       32,          1,        2,        { 4,  4,  3,  64,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-6nm-harvardnet_w2_53x53_3D_4M_K11_AC4/conv_1A_1B_kernels.txt" },
 { 2,     OP_POOL,       32,          2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 3,     OP_CONV,       32,         1,        2,        { 4,  4,  32,  64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-6nm-harvardnet_w2_53x53_3D_4M_K11_AC4/conv_2A_2B_kernels.txt" },
 { 4,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 5,     OP_CONV,       32,         1,        2,        { 4,  4,  32, 64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-6nm-harvardnet_w2_53x53_3D_4M_K11_AC4/conv_3A_3B_kernels.txt" },
 { 6,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 7,     OP_CONV,       2,          1,        1,        { 4,  4,  32, 2,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-6nm-harvardnet_w2_53x53_3D_4M_K11_AC4/ip_conv_kernels.txt" },
 { 8,     OP_SOFTMAX,    2,          1,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
 { 9,     OP_INTERWEAVE, 64,         8,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
};*/

// 32F - K11-S1-6nm-harvardnet_65x65_2D_Vesicle_AC4
/*layer_t net[] = {
// l_id,  op_type,       n_channels, n_stride, n_maxout, { d1, d2, d3, n_matrices, n_size, p_data}, init_filename 
 { 1,     OP_CONV,       32,          1,        2,        { 4,  4,  1,  64,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-6nm-harvardnet_65x65_2D_Vesicle_AC4/conv_1A_1B_kernels.txt" },
 { 2,     OP_POOL,       32,          2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 3,     OP_CONV,       32,         1,        2,        { 4,  4,  32,  64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-6nm-harvardnet_65x65_2D_Vesicle_AC4/conv_2A_2B_kernels.txt" },
 { 4,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 5,     OP_CONV,       32,         1,        2,        { 4,  4,  32, 64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-6nm-harvardnet_65x65_2D_Vesicle_AC4/conv_3A_3B_kernels.txt" },
 { 6,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 7,     OP_CONV,       2,          1,        1,        { 6,  6,  32, 2,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-6nm-harvardnet_65x65_2D_Vesicle_AC4/ip_conv_kernels.txt" },
 { 8,     OP_SOFTMAX,    2,          1,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
 { 9,     OP_INTERWEAVE, 64,         8,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
};*/

// 32F - K11-S1-6nm-harvardnet_65x65_3D_Synapse_AC4
/*layer_t net[] = {
// l_id,  op_type,       n_channels, n_stride, n_maxout, { d1, d2, d3, n_matrices, n_size, p_data}, init_filename 
 { 1,     OP_CONV,       32,          1,        2,        { 4,  4,  3,  64,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-6nm-harvardnet_65x65_3D_Synapse_AC4/conv_1A_1B_kernels.txt" },
 { 2,     OP_POOL,       32,          2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 3,     OP_CONV,       32,         1,        2,        { 4,  4,  32,  64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-6nm-harvardnet_65x65_3D_Synapse_AC4/conv_2A_2B_kernels.txt" },
 { 4,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 5,     OP_CONV,       32,         1,        2,        { 4,  4,  32, 64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-6nm-harvardnet_65x65_3D_Synapse_AC4/conv_3A_3B_kernels.txt" },
 { 6,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 7,     OP_CONV,       2,          1,        1,        { 6,  6,  32, 2,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-6nm-harvardnet_65x65_3D_Synapse_AC4/ip_conv_kernels.txt" },
 { 8,     OP_SOFTMAX,    2,          1,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
 { 9,     OP_INTERWEAVE, 64,         8,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
};*/

// 32F - Harvardnet_dist_4_105x105_3D_4M_K11_3nm_AC3
/*layer_t net[] = {
// l_id,  op_type,       n_channels, n_stride, n_maxout, { d1, d2, d3, n_matrices, n_size, p_data}, init_filename 
 { 1,     OP_CONV,       32,          1,        2,        { 4,  4,  3,  64,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-3nm-harvardnet_dist_4_105x105_3D_4M/conv_1A_1B_kernels.txt" },
 { 2,     OP_POOL,       32,          2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },

 { 3,     OP_CONV,       32,         1,        2,        { 4,  4,  32,  64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-3nm-harvardnet_dist_4_105x105_3D_4M/conv_2A_2B_kernels.txt" },
 { 4,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },

 { 5,     OP_CONV,       32,         1,        2,        { 4,  4,  32, 64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-3nm-harvardnet_dist_4_105x105_3D_4M/conv_3A_3B_kernels.txt" },
 { 6,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },

 { 7,     OP_CONV,       32,         1,        2,        { 4,  4,  32, 64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-3nm-harvardnet_dist_4_105x105_3D_4M/conv_4A_4B_kernels.txt" },
 { 8,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 
 { 9,     OP_CONV,       4,          1,        1,        { 4,  4,  32, 4,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-3nm-harvardnet_dist_4_105x105_3D_4M/ip_conv_kernels.txt" },
 {10,     OP_SOFTMAX,    4,          1,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
 {11,     OP_INTERWEAVE, 256,         16,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
};*/

// 32F - Harvardnet_w2_bg_32f_105x105_3D_4M_K11_3nm_AC3
/*layer_t net[] = {
// l_id,  op_type,       n_channels, n_stride, n_maxout, { d1, d2, d3, n_matrices, n_size, p_data}, init_filename 
 { 1,     OP_CONV,       32,          1,        2,        { 4,  4,  3,  64,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-3nm-harvardnet_w2_bg_32f_105x105_3D_4M/conv_1A_1B_kernels.txt" },
 { 2,     OP_POOL,       32,          2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },

 { 3,     OP_CONV,       32,         1,        2,        { 4,  4,  32,  64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-3nm-harvardnet_w2_bg_32f_105x105_3D_4M/conv_2A_2B_kernels.txt" },
 { 4,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },

 { 5,     OP_CONV,       32,         1,        2,        { 4,  4,  32, 64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-3nm-harvardnet_w2_bg_32f_105x105_3D_4M/conv_3A_3B_kernels.txt" },
 { 6,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },

 { 7,     OP_CONV,       32,         1,        2,        { 4,  4,  32, 64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-3nm-harvardnet_w2_bg_32f_105x105_3D_4M/conv_4A_4B_kernels.txt" },
 { 8,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 
 { 9,     OP_CONV,       2,          1,        1,        { 4,  4,  32, 2,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-3nm-harvardnet_w2_bg_32f_105x105_3D_4M/ip_conv_kernels.txt" },
 {10,     OP_SOFTMAX,    2,          1,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
 {11,     OP_INTERWEAVE, 256,         16,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
};*/

// 32F - ECS_harvardnet_w2_bg_only_32f_105x105_3D_4M
/*layer_t net[] = {
// l_id,  op_type,       n_channels, n_stride, n_maxout, { d1, d2, d3, n_matrices, n_size, p_data}, init_filename 
 { 1,     OP_CONV,       32,          1,        2,        { 4,  4,  3,  64,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/ECS_harvardnet_w2_bg_only_32f_105x105_3D_4M/conv_1A_1B_kernels.txt" },
 { 2,     OP_POOL,       32,          2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },

 { 3,     OP_CONV,       32,         1,        2,        { 4,  4,  32,  64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/ECS_harvardnet_w2_bg_only_32f_105x105_3D_4M/conv_2A_2B_kernels.txt" },
 { 4,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },

 { 5,     OP_CONV,       32,         1,        2,        { 4,  4,  32, 64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/ECS_harvardnet_w2_bg_only_32f_105x105_3D_4M/conv_3A_3B_kernels.txt" },
 { 6,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },

 { 7,     OP_CONV,       32,         1,        2,        { 4,  4,  32, 64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/ECS_harvardnet_w2_bg_only_32f_105x105_3D_4M/conv_4A_4B_kernels.txt" },
 { 8,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 
 { 9,     OP_CONV,       2,          1,        1,        { 4,  4,  32, 2,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/ECS_harvardnet_w2_bg_only_32f_105x105_3D_4M/ip_conv_kernels.txt" },
 {10,     OP_SOFTMAX,    2,          1,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
 {11,     OP_INTERWEAVE, 256,         16,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
};*/

// 32F - ECS_harvardnet_w2_bg_dist_4_32f_105x105_3D_4M
/*layer_t net[] = {
// l_id,  op_type,       n_channels, n_stride, n_maxout, { d1, d2, d3, n_matrices, n_size, p_data}, init_filename 
 { 1,     OP_CONV,       32,          1,        2,        { 4,  4,  3,  64,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/ECS_harvardnet_w2_bg_dist_4_32f_105x105_3D_4M/conv_1A_1B_kernels.txt" },
 { 2,     OP_POOL,       32,          2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },

 { 3,     OP_CONV,       32,         1,        2,        { 4,  4,  32,  64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/ECS_harvardnet_w2_bg_dist_4_32f_105x105_3D_4M/conv_2A_2B_kernels.txt" },
 { 4,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },

 { 5,     OP_CONV,       32,         1,        2,        { 4,  4,  32, 64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/ECS_harvardnet_w2_bg_dist_4_32f_105x105_3D_4M/conv_3A_3B_kernels.txt" },
 { 6,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },

 { 7,     OP_CONV,       32,         1,        2,        { 4,  4,  32, 64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/ECS_harvardnet_w2_bg_dist_4_32f_105x105_3D_4M/conv_4A_4B_kernels.txt" },
 { 8,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 
 { 9,     OP_CONV,       4,          1,        1,        { 4,  4,  32, 4,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/ECS_harvardnet_w2_bg_dist_4_32f_105x105_3D_4M/ip_conv_kernels.txt" },
 {10,     OP_SOFTMAX,    4,          1,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
 {11,     OP_INTERWEAVE, 256,         16,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
};*/



// 32F - Harvardnet_w2_bg_32f_105x105_3D_4M_K11_3nm_AC3 (SUB-2)
/*layer_t net[] = {
// l_id,  op_type,       n_channels, n_stride, n_maxout, { d1, d2, d3, n_matrices, n_size, p_data}, init_filename 
 { 1,     OP_CONV,       32,          1,        2,        { 4,  4,  3,  64,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-3nm-harvardnet_w2_bg_32f_105x105_3D_4M/conv_1A_1B_kernels.txt" },
 { 2,     OP_POOL,       32,          2,        -1,       { 2,  2,  -1, -1,         0,     NULL  }, NULL },

 { 3,     OP_CONV,       32,         1,        2,        { 4,  4,  32,  64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-3nm-harvardnet_w2_bg_32f_105x105_3D_4M/conv_2A_2B_kernels.txt" },
 { 4,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },

 { 5,     OP_CONV,       32,         1,        2,        { 4,  4,  32, 64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-3nm-harvardnet_w2_bg_32f_105x105_3D_4M/conv_3A_3B_kernels.txt" },
 { 6,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },

 { 7,     OP_CONV,       32,         1,        2,        { 4,  4,  32, 64,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-3nm-harvardnet_w2_bg_32f_105x105_3D_4M/conv_4A_4B_kernels.txt" },
 { 8,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 
 { 9,     OP_CONV,       2,          1,        1,        { 4,  4,  32, 2,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-3nm-harvardnet_w2_bg_32f_105x105_3D_4M/ip_conv_kernels.txt" },
 {10,     OP_SOFTMAX,    2,          1,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
 {11,     OP_INTERWEAVE, 64,         8,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
};*/

// OSDI NET
// 32F - Harvardnet_w2_bg_8f_105x105_3D_4M_K11_3nm_AC3
/*layer_t net[] = {
// l_id,  op_type,       n_channels, n_stride, n_maxout, { d1, d2, d3, n_matrices, n_size, p_data}, init_filename 
 { 1,     OP_CONV,       8,          1,        2,        { 4,  4,  3,  16,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-3nm-harvardnet_w2_bg_8f_105x105_3D_4M/conv_1A_1B_kernels.txt" },
 { 2,     OP_POOL,       8,          2,        -1,       { 2,  2,  -1, -1,         0,     NULL  }, NULL },

 { 3,     OP_CONV,       8,         1,        2,        { 4,  4,  8,  16,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-3nm-harvardnet_w2_bg_8f_105x105_3D_4M/conv_2A_2B_kernels.txt" },
 { 4,     OP_POOL,       8,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },

 { 5,     OP_CONV,       8,         1,        2,        { 4,  4,  8,  16,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-3nm-harvardnet_w2_bg_8f_105x105_3D_4M/conv_3A_3B_kernels.txt" },
 { 6,     OP_POOL,       8,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },

 { 7,     OP_CONV,       8,         1,        2,        { 4,  4,  8,  16,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-3nm-harvardnet_w2_bg_8f_105x105_3D_4M/conv_4A_4B_kernels.txt" },
 { 8,     OP_POOL,       8,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 
 { 9,     OP_CONV,       2,          1,        1,        { 4,  4,  8, 2,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-3nm-harvardnet_w2_bg_8f_105x105_3D_4M/ip_conv_kernels.txt" },
 {10,     OP_SOFTMAX,    2,          1,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
 {11,     OP_INTERWEAVE, 64,         8,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
};*/

// OSDI NET AC3 train on 10 val on 65
/*layer_t net[] = {
// l_id,  op_type,       n_channels, n_stride, n_maxout, { d1, d2, d3, n_matrices, n_size, p_data}, init_filename 
 { 1,     OP_CONV,       8,          1,        2,        { 4,  4,  3,  16,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-3nm-harvardnet_w2_bg_105x105_3D_4M_K11_3nm_AC3_train_on_10_val_on_65/conv_1A_1B_kernels.txt" },
 { 2,     OP_POOL,       8,          2,        -1,       { 2,  2,  -1, -1,         0,     NULL  }, NULL },

 { 3,     OP_CONV,       8,         1,        2,        { 4,  4,  8,  16,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-3nm-harvardnet_w2_bg_105x105_3D_4M_K11_3nm_AC3_train_on_10_val_on_65/conv_2A_2B_kernels.txt" },
 { 4,     OP_POOL,       8,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },

 { 5,     OP_CONV,       8,         1,        2,        { 4,  4,  8,  16,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-3nm-harvardnet_w2_bg_105x105_3D_4M_K11_3nm_AC3_train_on_10_val_on_65/conv_3A_3B_kernels.txt" },
 { 6,     OP_POOL,       8,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },

 { 7,     OP_CONV,       8,         1,        2,        { 4,  4,  8,  16,         -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-3nm-harvardnet_w2_bg_105x105_3D_4M_K11_3nm_AC3_train_on_10_val_on_65/conv_4A_4B_kernels.txt" },
 { 8,     OP_POOL,       8,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 
 { 9,     OP_CONV,       2,          1,        1,        { 4,  4,  8, 2,          -1,     NULL  }, (char *)"/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/K11-S1-3nm-harvardnet_w2_bg_105x105_3D_4M_K11_3nm_AC3_train_on_10_val_on_65/ip_conv_kernels.txt" },
 {10,     OP_SOFTMAX,    2,          1,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
 {11,     OP_INTERWEAVE, 64,         8,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
};*/


// 32F - K11_S1_AC3_256_cc_53_dist_4_3D_PAD
/*layer_t net[] = {
// l_id,  op_type,       n_channels, n_stride, n_maxout, { d1, d2, d3, n_matrices, n_size, p_data}, init_filename 
 { 1,     OP_CONV,       32,          1,        2,        { 4,  4,  3,  64,          -1,     NULL  }, (char *)"/home/amatveev/Pipeline/fc_dnn/fully_conv/fully_conv/K11_S1_AC3_256_cc_53_dist_4_3D_PAD/conv1A_1B_kernels.txt" },
 { 2,     OP_POOL,       32,          1,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 3,     OP_CONV,       32,         1,        2,        { 4,  4,  32,  64,         -1,     NULL  }, (char *)"/home/amatveev/Pipeline/fc_dnn/fully_conv/fully_conv/K11_S1_AC3_256_cc_53_dist_4_3D_PAD/conv2A_2B_kernels.txt" },
 { 4,     OP_POOL,       32,         1,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 5,     OP_CONV,       32,         1,        2,        { 4,  4,  32, 64,         -1,     NULL  }, (char *)"/home/amatveev/Pipeline/fc_dnn/fully_conv/fully_conv/K11_S1_AC3_256_cc_53_dist_4_3D_PAD/conv3A_3B_kernels.txt" },
 { 6,     OP_POOL,       32,         1,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 7,     OP_CONV,       4,          1,        1,        { 4,  4,  32, 4,          -1,     NULL  }, (char *)"/home/amatveev/Pipeline/fc_dnn/fully_conv/fully_conv/K11_S1_AC3_256_cc_53_dist_4_3D_PAD/ip_conv_kernels.txt" },
 { 8,     OP_SOFTMAX,    4,          1,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
 { 8,     OP_INTERWEAVE, 1,         1,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
};*/

// 8F-16F-32F
/*layer_t net[] = {
// l_id,  op_type,       n_channels, n_stride, n_maxout, { d1, d2, d3, n_matrices, n_size, p_data}, init_filename 
 { 1,     OP_CONV,       8,          1,        2,        { 4,  4,  1,  16,          -1,     NULL  }, (char *)"./test_49_w2_4M_8f_16f_32f/conv1A_conv1B_kernels.txt" },
 { 2,     OP_POOL,       8,          2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 3,     OP_CONV,       16,         1,        2,        { 4,  4,  8,  32,         -1,     NULL  }, (char *)"./test_49_w2_4M_8f_16f_32f/conv2A_conv2B_kernels.txt" },
 { 4,     OP_POOL,       16,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 5,     OP_CONV,       32,         1,        2,        { 4,  4,  16, 64,         -1,     NULL  }, (char *)"./test_49_w2_4M_8f_16f_32f/conv3A_conv3B_kernels.txt" },
 { 6,     OP_POOL,       32,         2,        -1,       { 2,  2,  -1, -1,         -1,     NULL  }, NULL },
 { 7,     OP_CONV,       2,          1,        1,        { 4,  4,  32,  2,          -1,     NULL  }, (char *)"./test_49_w2_4M_8f_16f_32f/ip_conv_kernels.txt" },
 { 8,     OP_SOFTMAX,    2,          1,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
 { 8,     OP_INTERWEAVE, 64,         8,        -1,       { -1,  -1,  -1, -1,       -1,     NULL  }, NULL },
};*/

/////////////////////////////////////////////////////////////////////////////////////////
// INTERNAL FUNCTIONS
/////////////////////////////////////////////////////////////////////////////////////////
void parse_args(dnn_t *p_dnn, int argc, char **argv) {
    
    DNN_ASSERT_MSG(argc == 11, "Usage: %s [input_type] [patch_leg] [3D_depth] [#matrices] [d1] [d2] [d3] [#out_channels] [input_file/dir] [output_file/dir]\n", argv[0]);
    
    int idx = 1;
    
    p_dnn->input_type = atoi(argv[idx]);
    idx++;
    p_dnn->input_patch_leg = atoi(argv[idx]);
    idx++;
    p_dnn->input_3D_depth = atoi(argv[idx]);
    idx++;
    p_dnn->input_n_matrices = atoi(argv[idx]);
    idx++;
    p_dnn->input_d1 = atoi(argv[idx]);
    idx++;
    p_dnn->input_d2 = atoi(argv[idx]);
    idx++;
    p_dnn->input_d3 = atoi(argv[idx]);
    idx++;
    p_dnn->n_output_channels = atoi(argv[idx]);
    idx++;
    
    DNN_ASSERT(p_dnn->input_3D_depth > 0);
    DNN_ASSERT(p_dnn->input_n_matrices > 0);
    DNN_ASSERT(p_dnn->input_d1 > 0);
    DNN_ASSERT(p_dnn->input_d2 > 0);
    DNN_ASSERT(p_dnn->input_d3 > 0);
    DNN_ASSERT(p_dnn->n_output_channels > 0);
    DNN_ASSERT(p_dnn->n_output_channels < MAX_OUTPUT_CHANNELS);
    
    p_dnn->input_filename = argv[idx];
    idx++;
    p_dnn->output_filename = argv[idx];
    idx++;
    
    p_dnn->is_pad_3D = 0;
#ifdef IS_PAD_3D
    p_dnn->is_pad_3D = 1;
#endif
    
}

/////////////////////////////////////////////////////////////////////////////////////////
// EXTERNAL FUNCTIONS
/////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    dnn_t *p_dnn;
    
    p_dnn = (dnn_t *)malloc(sizeof(dnn_t));
    
    p_dnn->p_in_matrices = NULL;
    p_dnn->p_out_matrices = NULL;
    p_dnn->n_layers = sizeof(net) / sizeof(net[0]);
    p_dnn->p_layers = net;
    
    parse_args(p_dnn, argc, argv);
    
    dnn_execute(p_dnn);
    
    
    return 0;
}