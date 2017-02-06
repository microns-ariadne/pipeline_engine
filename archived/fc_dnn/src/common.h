/////////////////////////////////////////////////////////////////////////////////////////
//
//
//
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef COMMON_H
#define COMMON_H 1

/////////////////////////////////////////////////////////////////////////////////////////
// INCLUDES
/////////////////////////////////////////////////////////////////////////////////////////
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <dirent.h> 
#include <math.h>

/////////////////////////////////////////////////////////////////////////////////////////
// DEFINES - Config 
/////////////////////////////////////////////////////////////////////////////////////////

#define RES_THREASHOLD (0.0)
#define QTD_THREASHOLD (1.5)
#define QTD_THREASHOLD_INPUT (0.8888)

#define CILK_FOR_M cilk_for

#define FIX_FC_OUT_SIZE (1)
#define FIX_FC_OUT_SIZE_DIVIDE (1) // sub-sampling

//#define IS_QTD

/////////////////////////////////////////////////////////////////////////////////////////
// DEFINES - Internal 
/////////////////////////////////////////////////////////////////////////////////////////
#define D1_TO_OFFSET(idx_d1, d2, d3) ((idx_d1) * ((d2) * (d3)))
#define D2_TO_OFFSET(idx_d2, d3) ((idx_d2) * (d3))

#define DIMS_TO_SIZE(d1, d2, d3) ((d1) * (d2) * (d3))

#define BUFFER_SIZE_FLOAT(n_matrices, d1, d2, d3) ((long)(sizeof(float) * (n_matrices) * DIMS_TO_SIZE((d1), (d2), (d3))))

#define BUFFER_SIZE_UINT32(n_matrices, d1, d2, d3) ((long)(sizeof(uint32_t) * (n_matrices) * DIMS_TO_SIZE((d1), (d2), (d3))))

#define MATRICES_DATA_ALIGNMENT (64)

#define GET_MATRIX_PTR(p_matrices, m_id) \
    ((p_matrices)->p_data + (DIMS_TO_SIZE((p_matrices)->d1, (p_matrices)->d2, (p_matrices)->d3) * (m_id)))

#define GET_AUX_DATA_PTR(p_matrices, m_id) ((p_matrices)->p_aux_data + (m_id))

#define GET_SP_OUT_MATRIX_PTR(p_matrices, m_id) \
    ((p_matrices)->p_sp_out_data + (DIMS_TO_SIZE((p_matrices)->d1, (p_matrices)->d2, 1) * (m_id)))

#define GET_SP_MASK_MATRIX_PTR(p_matrices, m_id) \
    ((p_matrices)->p_sp_mask_data + (DIMS_TO_SIZE((p_matrices)->d1, (p_matrices)->d2, 1) * (m_id)))

#define RESET_SP_OUT_MATRIX_PTR(p_matrices, m_id) { \
    uint32_t *p_sp_out_matrix = GET_SP_OUT_MATRIX_PTR(p_matrices, m_id); \
    memset(p_sp_out_matrix, 0, sizeof(uint32_t) * DIMS_TO_SIZE((p_matrices)->d1, (p_matrices)->d2, 1)); \
}
    
#define RESET_SP_MASK_MATRIX_PTR(p_matrices, m_id) { \
    uint32_t *p_sp_mask_matrix = GET_SP_MASK_MATRIX_PTR(p_matrices, m_id); \
    memset(p_sp_mask_matrix, 0, sizeof(uint32_t) * DIMS_TO_SIZE((p_matrices)->d1, (p_matrices)->d2, 1)); \
}

#define OUT_DIM(in_dim, kernel_dim, stride) ((((((in_dim) - (kernel_dim)) + 1) - 1) / (stride)) + 1)

#define MAX_OUTPUT_CHANNELS (32)

#define CONV_DEPTH_LIMIT (8)

#define IS_PAD_AVX (1)
#define MIN_AVX_DEPTH (8)

//#define WINDOW_D1 (10000)
//#define WINDOW_D2 (10000)
#define WINDOW_D1 (128)
#define WINDOW_D2 (128)

#define WINDOW_POOL_D1 (128)
#define WINDOW_POOL_D2 (128)

/////////////////////////////////////////////////////////////////////////////////////////
// DEFINES 
/////////////////////////////////////////////////////////////////////////////////////////
#define K_D1_1 (1)
#define K_D2_1 (1)

#define K_D1_2 (2)
#define K_D2_2 (2)

#define K_D1_4 (4)
#define K_D2_4 (4)

#define K_D1_6 (6)
#define K_D2_6 (6)

#define N_CHANNELS_48 (48)
#define N_CHANNELS_32 (32)
#define N_CHANNELS_16 (16)
#define N_CHANNELS_8 (8)

#define N_MAXOUT_KERNELS (2)
#define N_KERNELS (N_MAXOUT_KERNELS * 4)



//

#define likely(x) __builtin_expect ((x), 1)
#define unlikely(x) __builtin_expect ((x), 0)

#define prefetch_read(addr) __builtin_prefetch(addr, 0)

#define DNN_ENABLE_TRACE_TIMER
#define DNN_ENABLE_TRACE_TIMER_2

#define DNN_ENABLE_TRACE_1
//#define DNN_ENABLE_TRACE_2
//#define DNN_ENABLE_TRACE_3
//#define DNN_ENABLE_TRACE_4

//#define DNN_ENABLE_DUMP_MATRICES

#ifdef DNN_ENABLE_DUMP_MATRICES
#define DUMP_MATRICES(p_matrices, l_id, layer_type_str, type_str) dump_matrices(p_matrices, l_id, #layer_type_str, type_str)
#else
#define DUMP_MATRICES(p_matrices, l_id, layer_type_str, type_str)
#endif

#define DUMP_INPUT "input"
#define DUMP_KERNELS "kernels"
#define DUMP_OUTPUT "output"

#define DUMP_FILE_FORMAT "./dump/layer_%d_%s_%s.txt"

#define MAGIC_TOTAL_MATRIX (0x1212121221212121)
#define MAGIC_MATRIX_START (0x1111111111111111)
#define MAGIC_MATRIX_END (0x2222222222222222)

/////////////////////////////////////////////////////////////////////////////////////////
// DEFINES - ASSERT AND DEBUG
/////////////////////////////////////////////////////////////////////////////////////////
#define DNN_TRACE(fmt, ...) \
    fprintf(stderr, "%s:%d:%s(): " fmt, \
            __FILE__, __LINE__, __func__, \
            __VA_ARGS__);

#define DNN_ABORT(fmt, ...) \
    DNN_TRACE(fmt, __VA_ARGS__); \
    abort();

#define DNN_ASSERT(cond) \
    if (unlikely(!(cond))) { \
        printf ("\n-----------------------------------------------\n"); \
        printf ("\nAssertion failure: %s:%d '%s'\n", __FILE__, __LINE__, #cond); \
        abort(); \
    }

#define DNN_ASSERT_MSG(cond, fmt, ...) \
    if (unlikely(!(cond))) { \
        printf ("\n-----------------------------------------------\n"); \
        printf ("\nAssertion failure: %s:%d '%s'\n", __FILE__, __LINE__, #cond); \
        DNN_TRACE(fmt, __VA_ARGS__); \
        abort(); \
    }

#ifdef DNN_ENABLE_TRACE_TIMER
#define TIMER_VAR(timer_name) struct timeval timer_name;
#define START_TIMER(timer) start_timer((timer))
#define STOP_TIMER(timer, msg) stop_timer((timer), msg)
#else
#define TIMER_VAR(timer_name)
#define START_TIMER(timer)
#define STOP_TIMER(timer, msg)
#endif

#ifdef DNN_ENABLE_TRACE_TIMER_2
#define TIMER_VAR_2(timer_name) struct timeval timer_name
#define START_TIMER_2(timer) start_timer((timer))
#define STOP_TIMER_2(timer, msg) stop_timer((timer), msg)
#else
#define TIMER_VAR_2(timer_name)
#define START_TIMER_2(timer)
#define STOP_TIMER_2(timer, msg)
#endif

#ifdef DNN_ENABLE_TRACE_1
#define DNN_TRACE_1(fmt, ...) DNN_TRACE(fmt, __VA_ARGS__)
#else
#define DNN_TRACE_1(fmt, ...)
#endif

#ifdef DNN_ENABLE_TRACE_2
#define DNN_TRACE_2(fmt, ...) DNN_TRACE(fmt, __VA_ARGS__)
#else
#define DNN_TRACE_2(fmt, ...)
#endif

#ifdef DNN_ENABLE_TRACE_3
#define DNN_TRACE_3(fmt, ...) DNN_TRACE(fmt, __VA_ARGS__)
#else
#define DNN_TRACE_3(fmt, ...)
#endif

#ifdef DNN_ENABLE_TRACE_4
#define DNN_TRACE_4(fmt, ...) DNN_TRACE(fmt, __VA_ARGS__)
#else
#define DNN_TRACE_4(fmt, ...)
#endif

/////////////////////////////////////////////////////////////////////////////////////////
// TYPES
/////////////////////////////////////////////////////////////////////////////////////////
typedef struct _matrices_data {
    int d1;
    int d2;
    int d3;
    int n_matrices;
    long n_size;
    
    uint32_t *p_sp_out_data;
    uint32_t *p_sp_mask_data;
    
    float *p_aux_data;
    float *p_data;
} matrices_data_t;

/////////////////////////////////////////////////////////////////////////////////////////
// EXTERNAL FUNCTIONS
/////////////////////////////////////////////////////////////////////////////////////////
matrices_data_t *allocate_matrices(int n_matrices, int d1, int d2, int d3);
void allocate_matrices_data(matrices_data_t *p_matrices, int n_matrices, int d1, int d2, int d3);
void fp_print_matrices(FILE *fp, matrices_data_t *p_matrices);
void dump_matrices(matrices_data_t *p_matrices, int l_id, const char *layer_type_str, const char *type_str);

void set_matrices_size(matrices_data_t *p_matrices, int n_matrices, int d1, int d2, int d3);

void start_timer(struct timeval *p_timer);
void stop_timer(struct timeval *p_timer, const char *msg);

/////////////////////////////////////////////////////////////////////////////////////////
// EXTERNAL MACROS
/////////////////////////////////////////////////////////////////////////////////////////

#endif // COMMON_H
