/////////////////////////////////////////////////////////////////////////////////////////
//
//
//
/////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////
// INCLUDES
/////////////////////////////////////////////////////////////////////////////////////////
#include <stdlib.h>
#include <malloc.h>
#include <sys/time.h>
#include "common.h"

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
void allocate_matrices_data(matrices_data_t *p_matrices, int n_matrices, int d1, int d2, int d3) {
    long n_size = -1;
    
    DNN_TRACE_1("n_matrices [%d] of shape [%d,%d,%d]\n", n_matrices, d1, d2, d3);
    
    n_size = BUFFER_SIZE_FLOAT(n_matrices, d1, d2, d3);
    
    DNN_TRACE_1("n_size = %ld\n", n_size);
    
    p_matrices->p_data = (float *)memalign(MATRICES_DATA_ALIGNMENT, n_size);
    p_matrices->n_size = n_size;
    
    DNN_ASSERT(p_matrices->p_data != NULL);
    
    //memset(p_matrices->p_data, 0, n_size);
    
    p_matrices->p_aux_data = (float *)memalign(MATRICES_DATA_ALIGNMENT, sizeof(float) * n_matrices);
    DNN_ASSERT(p_matrices->p_aux_data != NULL);
    
    n_size = BUFFER_SIZE_UINT32(n_matrices, d1, d2, d3);
    
    p_matrices->p_sp_out_data = (uint32_t *)memalign(MATRICES_DATA_ALIGNMENT, n_size);
    DNN_ASSERT(p_matrices->p_sp_out_data != NULL);
    
    p_matrices->p_sp_mask_data = (uint32_t *)memalign(MATRICES_DATA_ALIGNMENT, n_size);
    DNN_ASSERT(p_matrices->p_sp_mask_data != NULL);
        
    //memset(p_matrices->p_aux_data, 0, sizeof(float) * n_matrices);
    
}

matrices_data_t *allocate_matrices(int n_matrices, int d1, int d2, int d3) {
    matrices_data_t *p_matrices;
    
    p_matrices = (matrices_data_t *)malloc(sizeof(matrices_data_t));
    
    allocate_matrices_data(p_matrices, n_matrices, d1, d2, d3);
    
    p_matrices->d1 = d1;
    p_matrices->d2 = d2;
    p_matrices->d3 = d3;
    p_matrices->n_matrices = n_matrices;
        
    return p_matrices;
}

void set_matrices_size(matrices_data_t *p_matrices, int n_matrices, int d1, int d2, int d3) {

    DNN_ASSERT(BUFFER_SIZE_FLOAT(n_matrices, d1, d2, d3) <= p_matrices->n_size);
        
    p_matrices->n_matrices = n_matrices;
    p_matrices->d1 = d1;
    p_matrices->d2 = d2;
    p_matrices->d3 = d3;
    
}

void fp_print_matrix(FILE *fp, float *p_matrix, int d1, int d2, int d3) {
    
	fprintf(fp, "-----------------------------------------");
	
	for (int cur_d1 = 0; cur_d1 < d1; cur_d1++) {
		
		int offset_d1 = D1_TO_OFFSET(cur_d1, d2, d3);
		
		fprintf(fp, "\n[%d]", cur_d1);
		for (int cur_d2 = 0; cur_d2 < d2; cur_d2++) {
			
			int offset_d2 = D2_TO_OFFSET(cur_d2, d3);
			
			fprintf(fp, " ( ");
			for (int cur_d3 = 0; cur_d3 < d3; cur_d3++) {
				fprintf(fp, "%g ", p_matrix[offset_d1 + offset_d2 + cur_d3]);
			}
			fprintf(fp, ")");
		}
	}
	fprintf(fp, "\n");
}

void fp_print_matrices(FILE *fp, matrices_data_t *p_matrices) {

	fprintf(fp, "=========================================\n");
	fprintf(fp, "n_matrices = %d of shape [%d,%d,%d]\n", 
	        p_matrices->n_matrices,
	        p_matrices->d1,
	        p_matrices->d2,
	        p_matrices->d3);
	fprintf(fp, "=========================================\n");
	for (int m_id = 0; m_id < p_matrices->n_matrices; m_id++) {
		fprintf(fp, "-----------------------------------------\n");
		fprintf(fp, "MATRIX[%d] aux_data = %g\n", m_id+1, *(GET_AUX_DATA_PTR(p_matrices, m_id)));
		fp_print_matrix(fp, 
		                GET_MATRIX_PTR(p_matrices, m_id), 
					    p_matrices->d1, 
					    p_matrices->d2, 
					    p_matrices->d3);
	}
	fprintf(fp, "=========================================\n");
}

void dump_matrices(matrices_data_t *p_matrices, int l_id, const char *layer_type_str, const char *type_str) {
    FILE *fp;
    char output_filename[2000];

    sprintf(output_filename, DUMP_FILE_FORMAT, l_id, layer_type_str, type_str);
    
    fp = fopen(output_filename, "wb");
    DNN_ASSERT_MSG(fp != NULL, "failed to open %s\n", output_filename);
    
    fp_print_matrices(fp, p_matrices);
    
    fclose(fp);
}

void start_timer(struct timeval *p_timer) {
    gettimeofday(p_timer, NULL);
}

void stop_timer(struct timeval *p_timer, const char *msg) {
    struct timeval timer2;
    double elapsed_time = 0;
    
    gettimeofday(&timer2, NULL);
    
    elapsed_time = (timer2.tv_sec - p_timer->tv_sec) * 1000000.0; // sec to microsecs
    elapsed_time += (timer2.tv_usec - p_timer->tv_usec); // microsecs
    
    printf("%s: %f [microsec]\n", msg, elapsed_time);
    
}

