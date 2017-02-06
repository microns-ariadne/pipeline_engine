/////////////////////////////////////////////////////////////////////////////////////////
//
//
//
/////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////
// INCLUDES
/////////////////////////////////////////////////////////////////////////////////////////
#include "common.h"
#include "conv.h"
#include "pool.h"
#include "softmax.h"
#include "interweave.h"
#include "dnn.h"

#include <vector>
#include <opencv2/opencv.hpp>

/////////////////////////////////////////////////////////////////////////////////////////
// DEFINES
/////////////////////////////////////////////////////////////////////////////////////////
#define MAX_INPUT_BUF (100)

//#define FLOAT_VAL_TO_UCHAR(f_val) (255 - (unsigned char)((f_val) * 255.0))
#define FLOAT_VAL_TO_UCHAR(f_val) ((unsigned char)((f_val) * 255.0))
#define FLIP_UCHAR(uc_val) (255 - (uc_val))

/////////////////////////////////////////////////////////////////////////////////////////
// TYPES
/////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////
// GLOBALS
/////////////////////////////////////////////////////////////////////////////////////////
volatile int g_n_input_files = 0;
char g_filepaths[MAX_INPUT_FILES][MAX_FILEPATH] = {0,};
cv::Mat *p_g_cv_images[MAX_INPUT_FILES];

/////////////////////////////////////////////////////////////////////////////////////////
// INTERNAL FUNCTIONS
/////////////////////////////////////////////////////////////////////////////////////////

void fp_init_matrix(
    float *p_matrix, 
    FILE *fp, 
    int d1, 
    int d2, 
    int d3,
    int is_pad_depth) {
    
    int c;
    int m_id;
    int n_objs_read;
    float num;
    
    DNN_TRACE_4("start [%d,%d,%d]\n", d1, d2, d3);
    
    m_id = 0;
    for (int i = 0; i < d1; i++) {

        for (int j = 0; j < d2; j++) {
            
            for (int k = 0; k < d3; k++) {
                if ((k == 0) && (d3 == 1)) {
                    n_objs_read = fscanf(fp, "(%f)", &num);
                } else if (k == 0) {
                    n_objs_read = fscanf(fp, "(%f ", &num);
                } else if (k == (d3-1)) {
                    n_objs_read = fscanf(fp, "%f)", &num);
                } else {
                    n_objs_read = fscanf(fp, "%f ", &num);
                }
                DNN_ASSERT(n_objs_read == 1);
            
                p_matrix[m_id] = num;
                m_id++;
            }
            
            if (is_pad_depth) {
                for (int k = 0; k < (MIN_AVX_DEPTH - d3); k++) {
                    p_matrix[m_id] = 0.0;
                    m_id++;
                }
            }
            
            if (j < (d2-1)) {
                c = fgetc(fp);
                DNN_ASSERT_MSG(c == ' ', "c == %c\n", c);
            }
        }
        
        c = fgetc(fp);
        DNN_ASSERT_MSG(c == '\n', "c == %c\n", c);
        
    }
    
    DNN_TRACE_4("finish [%d,%d,%d]\n", d1, d2, d3);
    
}

void fp_init_matrix_from_binary(float *p_matrix, FILE *fp, int d1, int d2, int d3) {
    int m_id;
    int n_objs_read;
    unsigned char byte;
    
    DNN_TRACE_3("start [%d,%d,%d]\n", d1, d2, d3);
    
    m_id = 0;
    for (int i = 0; i < d1; i++) {

        for (int j = 0; j < d2; j++) {
            
            for (int k = 0; k < d3; k++) {
                n_objs_read = fread(&byte, sizeof(unsigned char), 1, fp);
                DNN_ASSERT(n_objs_read == 1);
                
                p_matrix[m_id] = (float)byte / 255.0;
                m_id++;
            }
        }
    }
    
    DNN_TRACE_3("finish [%d,%d,%d]\n", d1, d2, d3);
    
}

void fp_output_matrix_to_binary(float *p_matrix, FILE *fp, int d1, int d2, int d3) {
    int m_id;
    int n_objs_written;
    unsigned char byte;
    
    DNN_TRACE_3("start [%d,%d,%d]\n", d1, d2, d3);
    
    m_id = 0;
    for (int i = 0; i < d1; i++) {

        for (int j = 0; j < d2; j++) {
            
            for (int k = 0; k < d3; k++) {
                
                byte = (unsigned char)(p_matrix[m_id] * 255.0);
                n_objs_written = fwrite(&byte, sizeof(unsigned char), 1, fp);
                DNN_ASSERT(n_objs_written == 1);
                
                m_id++;
            }
        }
    }
    
    DNN_TRACE_3("finish [%d,%d,%d]\n", d1, d2, d3);
    
}

void init_matrices_from_binary_file(matrices_data_t *p_matrices, char *init_filename) {
    FILE *fp;
    
    long magic_total_matrix = MAGIC_TOTAL_MATRIX;
    long magic_matrix_start = MAGIC_MATRIX_START;
    long magic_matrix_end = MAGIC_MATRIX_END;
    
    long l_magic;
    long l_n_matrices;
    long l_d1,l_d2,l_d3;
    
    int n_objs_read;
    
    DNN_TRACE_1("start reading %d matrices of [%d,%d,%d] from %s\n", 
                p_matrices->n_matrices,
                p_matrices->d1,
                p_matrices->d2,
                p_matrices->d3,
                init_filename);
    
    DNN_ASSERT_MSG(init_filename != NULL, "init_matrices_from_file: init_filename = %p\n", init_filename);
    
    fp = fopen(init_filename, "rb");
    DNN_ASSERT_MSG(fp != NULL, "failed to open %s\n", init_filename);
    
    n_objs_read = fread(&l_magic, sizeof(long), 1, fp);
    DNN_ASSERT(n_objs_read == 1);
    DNN_ASSERT(l_magic == magic_total_matrix);
    
    n_objs_read = fread(&l_n_matrices, sizeof(long), 1, fp);
    DNN_ASSERT(n_objs_read == 1);
    DNN_ASSERT(l_n_matrices == p_matrices->n_matrices);
    
    n_objs_read = fread(&l_d1, sizeof(long), 1, fp);
    DNN_ASSERT(n_objs_read == 1);
    DNN_ASSERT(l_d1 == p_matrices->d1);

    n_objs_read = fread(&l_d2, sizeof(long), 1, fp);
    DNN_ASSERT(n_objs_read == 1);
    DNN_ASSERT(l_d2 == p_matrices->d2);
    
    n_objs_read = fread(&l_d3, sizeof(long), 1, fp);
    DNN_ASSERT(n_objs_read == 1);
    DNN_ASSERT(l_d3 == p_matrices->d3);
    
    for (int i = 0; i < l_n_matrices; i++) {
        
        n_objs_read = fread(&l_magic, sizeof(long), 1, fp);
        DNN_ASSERT(n_objs_read == 1);
        DNN_ASSERT(l_magic == magic_matrix_start);
        
        fp_init_matrix_from_binary(GET_MATRIX_PTR(p_matrices, i), fp, l_d1, l_d2, l_d3);
        
        n_objs_read = fread(&l_magic, sizeof(long), 1, fp);
        DNN_ASSERT(n_objs_read == 1);
        DNN_ASSERT(l_magic == magic_matrix_end);
    }
    
    fclose(fp);
    
    DNN_TRACE_1("finished reading %d matrices of [%d,%d,%d] from %s\n", 
                p_matrices->n_matrices,
                p_matrices->d1,
                p_matrices->d2,
                p_matrices->d3,
                init_filename);
    
}

void output_matrices_to_binary_file(matrices_data_t *p_matrices,
                                    const char *output_filename) {
    FILE *fp;
    
    long magic_total_matrix = MAGIC_TOTAL_MATRIX;
    long magic_matrix_start = MAGIC_MATRIX_START;
    long magic_matrix_end = MAGIC_MATRIX_END;
    
    long l_n_matrices;
    long l_d1,l_d2,l_d3;
    
    int n_objs_written;
    
    l_n_matrices = p_matrices->n_matrices;
    l_d1 = p_matrices->d1;
    l_d2 = p_matrices->d2;
    l_d3 = p_matrices->d3;
    
    DNN_TRACE_1("start writing %d matrices of [%d,%d,%d] to %s\n", 
                p_matrices->n_matrices,
                p_matrices->d1,
                p_matrices->d2,
                p_matrices->d3,
                output_filename);
    
    DNN_ASSERT_MSG(output_filename != NULL, "output_filename = %p\n", output_filename);
    
    fp = fopen(output_filename, "wb");
    DNN_ASSERT_MSG(fp != NULL, "failed to open %s\n", output_filename);
    
    n_objs_written = fwrite(&magic_total_matrix, sizeof(long), 1, fp);
    DNN_ASSERT(n_objs_written == 1);
    

    n_objs_written = fwrite(&l_n_matrices, sizeof(long), 1, fp);
    DNN_ASSERT(n_objs_written == 1);
    
    n_objs_written = fwrite(&l_d1, sizeof(long), 1, fp);
    DNN_ASSERT(n_objs_written == 1);

    n_objs_written = fwrite(&l_d2, sizeof(long), 1, fp);
    DNN_ASSERT(n_objs_written == 1);

    n_objs_written = fwrite(&l_d3, sizeof(long), 1, fp);
    DNN_ASSERT(n_objs_written == 1);

    for (int i = 0; i < l_n_matrices; i++) {
        
        n_objs_written = fwrite(&magic_matrix_start, sizeof(long), 1, fp);
        DNN_ASSERT(n_objs_written == 1);
        
        fp_output_matrix_to_binary(GET_MATRIX_PTR(p_matrices, i), fp, l_d1, l_d2, l_d3);
        
        n_objs_written = fwrite(&magic_matrix_end, sizeof(long), 1, fp);
        DNN_ASSERT(n_objs_written == 1);
    }
    
    fclose(fp);
    
    DNN_TRACE_1("finished writing %d matrices of [%d,%d,%d] to %s\n", 
                p_matrices->n_matrices,
                p_matrices->d1,
                p_matrices->d2,
                p_matrices->d3,
                output_filename);
    
}

void output_matrices_to_files(matrices_data_t *p_out_matrices, 
                              int n_output_channels,
                              const char *output_dirname) {
    
    int out_n_matrices;
    int out_d1,out_d2,out_d3;
        
    out_d1 = p_out_matrices->d1;
    out_d2 = p_out_matrices->d2;
    out_d3 = p_out_matrices->d3;
    out_n_matrices = p_out_matrices->n_matrices;
    
    int n_out_groups = out_n_matrices / n_output_channels;
    
    char l_filepaths[MAX_INPUT_FILES][MAX_FILEPATH] = {0,};
    
    DNN_ASSERT((n_out_groups * n_output_channels) == out_n_matrices);
    DNN_ASSERT(out_d3 == 1);
    DNN_ASSERT(output_dirname != NULL);
    
    DNN_TRACE_1("start writing %d matrices of [%d,%d,%d] to dir: %s\n", 
                out_n_matrices,
                out_d1,
                out_d2,
                out_d3,
                output_dirname);
    
    memcpy(l_filepaths, g_filepaths, sizeof(char) * sizeof(g_filepaths));
    
    CILK_FOR_M (int out_group_id = 0; out_group_id < n_out_groups; out_group_id++) {
        
        CILK_FOR_M (int ch_id = 0; ch_id < n_output_channels; ch_id++) {
            
            int out_m_id = (out_group_id * n_output_channels) + ch_id;
                
            char out_prefix[MAX_FILEPATH]; 
            char out_filename[MAX_FILEPATH]; 
            char out_filepath[MAX_FILEPATH]; 

            float *p_out_matrix = GET_MATRIX_PTR(p_out_matrices, out_m_id);
            
            std::vector<int> compressionParams;
            compressionParams.push_back(CV_IMWRITE_PNG_COMPRESSION);
            compressionParams.push_back(0);
            cv::Mat out_mat(out_d1, out_d2, CV_8UC1);

            int cur_idx = 0;
            unsigned char uc_min_val = p_out_matrix[0];
            unsigned char uc_max_val = p_out_matrix[0];
            for (int i = 0; i < out_d1; i++) {
                for (int j = 0; j < out_d2; j++) {
                    float cur_val = p_out_matrix[cur_idx];

                    unsigned char uc_cur_val = FLOAT_VAL_TO_UCHAR(cur_val);
                    //uc_cur_val = FLIP_UCHAR(uc_cur_val);

                    if (uc_cur_val < uc_min_val) {
                        uc_min_val = uc_cur_val;
                    }

                    if (uc_cur_val > uc_max_val) {
                        uc_max_val = uc_cur_val;
                    }

                    cur_idx++;
                }
            }

            cur_idx = 0;
            for (int i = 0; i < out_d1; i++) {
                for (int j = 0; j < out_d2; j++) {
                    float cur_val = p_out_matrix[cur_idx];

                    unsigned char uc_cur_val = FLOAT_VAL_TO_UCHAR(cur_val);
                    //uc_cur_val = FLIP_UCHAR(uc_cur_val);

                    float cur_val_fixed = ((float)(uc_cur_val - uc_min_val)) / ((float)uc_max_val);

                    uc_cur_val = FLOAT_VAL_TO_UCHAR(cur_val_fixed);
                    out_mat.at<uchar>(i,j) = uc_cur_val;
                    cur_idx++;
                }

            }

            sprintf(out_prefix, "%s", basename(l_filepaths[out_group_id]));
            sprintf(out_filename, "%s-probs-%d.png", out_prefix, ch_id);
            sprintf(out_filepath, "%s/%s", output_dirname, out_filename);
            cv::imwrite(out_filepath, out_mat, compressionParams);

        }        
    }
    
    DNN_TRACE_1("finished writing %d matrices of [%d,%d,%d] to dir: %s\n", 
                out_n_matrices,
                out_d1,
                out_d2,
                out_d3,
                output_dirname);    
    
}

void init_matrices_from_file(
    matrices_data_t *p_matrices, 
    int in_d1,
    int in_d2,
    int in_d3,
    char *init_filename,
    int is_pad_depth) 
{
    FILE *fp;
    
    char magic_str_total_matrix[] = "total-matrices:";
    char magic_str_matrix_start[] = "matrix-start:";
    char magic_str_matrix_end[] = "matrix-end:";
    
    char str_total_matrix[MAX_INPUT_BUF] = {0,};
    char str_matrix[MAX_INPUT_BUF] = {0,};
    
    float aux_data;
    
    int n_objs_read;
    int n_matrices;
    int m_id,d1,d2,d3;
    
    DNN_TRACE_1("start reading %d matrices of [%d,%d,%d] from %s\n", 
                p_matrices->n_matrices,
                in_d1,
                in_d2,
                in_d3,
                init_filename);
    
    DNN_ASSERT_MSG(init_filename != NULL, "init_matrices_from_file: init_filename = %p\n", init_filename);
    
    fp = fopen(init_filename, "rb");
    DNN_ASSERT_MSG(fp != NULL, "failed to open %s\n", init_filename);
    
    n_objs_read = fscanf(fp, "%s %d\n", str_total_matrix, &n_matrices);
    DNN_ASSERT(n_objs_read == 2);
    
    DNN_ASSERT(n_matrices == p_matrices->n_matrices);
    DNN_ASSERT(0 == strcmp(str_total_matrix, magic_str_total_matrix));
    
    for (int i = 0; i < n_matrices; i++) {
        n_objs_read = fscanf(fp, "%s [%d] [%d %d %d] (%f)\n", str_matrix, &m_id, &d1, &d2, &d3, &aux_data);
        DNN_ASSERT(n_objs_read == 6);
        
        *(GET_AUX_DATA_PTR(p_matrices, i)) = aux_data;
        
        DNN_ASSERT(m_id == (i+1));
        DNN_ASSERT(0 == strcmp(str_matrix, magic_str_matrix_start));
        if ((d1 != in_d1) || 
            (d2 != in_d2) || 
            (d3 != in_d3)) {
            DNN_ABORT("Unexpected matrix format in %s [%d,%d,%d] (need: [%d,%d,%d])\n",
                      init_filename,
                      d1, d2, d3,
                      in_d1, in_d2, in_d3);
        }
        
        fp_init_matrix(
            GET_MATRIX_PTR(p_matrices, i), 
            fp, 
            in_d1, 
            in_d2, 
            in_d3,
            is_pad_depth);
        
        n_objs_read = fscanf(fp, "%s [%d]\n", str_matrix, &m_id);
        DNN_ASSERT(n_objs_read == 2);
        
        DNN_ASSERT(m_id == (i+1));
        DNN_ASSERT(0 == strcmp(str_matrix, magic_str_matrix_end));
                
    }
    
    fclose(fp);
    
    DNN_TRACE_1("finished reading %d matrices of [%d,%d,%d] from %s\n", 
                p_matrices->n_matrices,
                in_d1,
                in_d2,
                in_d3,
                init_filename);
    
    
}

void init_matrices_from_dir(matrices_data_t *p_matrices, char *dirname) {
    int n_files;
    struct dirent *d_entry;
    struct dirent **d_filenames;
    
    char l_filepaths[MAX_INPUT_FILES][MAX_FILEPATH] = {0,};
       
    DNN_ASSERT(dirname != NULL);
    DNN_ASSERT(1 == p_matrices->d3);
    
    n_files = scandir(dirname, &d_filenames, 0, alphasort);
    
    DNN_ASSERT(n_files > 0);
    
    DNN_TRACE_1("start reading %d matrices of [%d,%d,%d] from: \n -- DIR: %s\n", 
                p_matrices->n_matrices,
                p_matrices->d1,
                p_matrices->d2,
                p_matrices->d3,
                dirname);
    
    int idx = 0;
    //int max_idx = 0;
    
    for (int file_id = 0; file_id < n_files; file_id++) {
        d_entry = d_filenames[file_id];
        DNN_ASSERT(d_entry != NULL);
        
        if (d_entry->d_type != DT_REG) {
            continue;
        }
        
        DNN_TRACE_1("[%d] %s\n", idx, d_entry->d_name);
        
        sprintf(l_filepaths[idx], "%s/%s", dirname, d_entry->d_name);
        
        idx++;
    }
    
    DNN_ASSERT(idx == p_matrices->n_matrices);
    DNN_ASSERT(idx < MAX_INPUT_FILES);
    
    for (int i = 0; i < p_matrices->n_matrices; i++) {
        DNN_ASSERT(l_filepaths[i][0] != 0);
    }
    
    g_n_input_files = idx;
    
    CILK_FOR_M (int in_im_id = 0; in_im_id < idx; in_im_id++) {
                
        (*p_g_cv_images[in_im_id]) = cv::imread(l_filepaths[in_im_id], CV_LOAD_IMAGE_UNCHANGED);
        
        if ((*p_g_cv_images[in_im_id]).empty()) {
            DNN_ABORT("cv::imread: failed to load image %s\n", l_filepaths[in_im_id]);
    	}
        
        if ((*p_g_cv_images[in_im_id]).channels() != 1) {
    	    DNN_ABORT("cv::imread: unexpected number of channels [%d] (expected: %d)\n", (*p_g_cv_images[in_im_id]).channels(), 1);
    	}
                
        DNN_ASSERT((*p_g_cv_images[in_im_id]).rows == p_matrices->d1);
        DNN_ASSERT((*p_g_cv_images[in_im_id]).cols == p_matrices->d2);
        
    }
    
    memcpy(g_filepaths, l_filepaths, sizeof(char) * sizeof(l_filepaths));
    
    DNN_TRACE_1("finished reading %d matrices of [%d,%d,%d] from: \n -- DIR: %s\n", 
                p_matrices->n_matrices,
                p_matrices->d1,
                p_matrices->d2,
                p_matrices->d3,
                dirname);
    
}

void init_matrices_from_opencv(
    matrices_data_t *p_matrices, 
    int d1_start,
    int d2_start,
    int d1_finish,
    int d2_finish,
    int in_id, 
    int in_3D_depth, 
    int is_pad_3D,
    int is_pad_AVX) {
    
    *(GET_AUX_DATA_PTR(p_matrices, 0)) = 0;
        
    float *p_matrix = GET_MATRIX_PTR(p_matrices, 0);
    
    //int cur_idx = 0;
    
    if (is_pad_3D) {
        
        DNN_ABORT("FIX is_pad_3D %s", "");
        abort();
        
        // DNN_ASSERT(in_3D_depth == 3);
        //         
        //         int start_depth = in_id - 1;
        //         int finish_depth = in_id + 1;
        //         
        //         DNN_TRACE_1("set input 3D with padding: [%d] <=> [%d]\n", start_depth, finish_depth)
        //         for (int i = d1_start; i < d1_finish; i++) {
        //             
        //             for (int j = d2_start; j < d2_finish; j++) {
        //                 
        //                 
        //                 
        //                 for (int d_im_id = start_depth; d_im_id <= finish_depth; d_im_id++) {
        //                     
        //                     if ((d_im_id < 0) || (d_im_id >= g_n_input_files)) {
        //                         p_matrix[cur_idx] = 0.0;
        //                     } else {
        //                         p_matrix[cur_idx] = ((float)g_cv_images[d_im_id].at<uint8_t>(i,j)) / 255.0;
        //                     }
        //                     
        //                     cur_idx++;
        //                 }
        //             }
        //         }
        
    } else {    
    
        for (int i = d1_start; i < d1_finish; i++) {
            
            int out_offset_d1 = D1_TO_OFFSET(i, p_matrices->d2, p_matrices->d3);
            
            for (int j = d2_start; j < d2_finish; j++) {
                
                int out_offset_d2 = D2_TO_OFFSET(j, p_matrices->d3);
                
                int ch_id = 0;
                
                for (int d_id = 0; d_id < in_3D_depth; d_id++) {
                    p_matrix[out_offset_d1 + out_offset_d2 + ch_id] = ((float)(*p_g_cv_images[in_id + d_id]).at<uint8_t>(i,j)) / 255.0;
                    ch_id++;
                }
                
                if ((is_pad_AVX) && (in_3D_depth < MIN_AVX_DEPTH)) {
                    int pad_depth = MIN_AVX_DEPTH - in_3D_depth;
                    for (int d_id = 0; d_id < pad_depth; d_id++) {
                        p_matrix[out_offset_d1 + out_offset_d2 + ch_id] = 0.0;
                        ch_id++;
                    }
                }
                
            }
        }
    }
    
}

void layer_init_kernels(layer_t *p_layer) {
    
    if (p_layer->init_filename == NULL) {
        DNN_TRACE_1("layer_init_kernels: skip layer %d\n", p_layer->l_id);
        return;
    }
    
    int in_d1 = p_layer->kernels.d1;
    int in_d2 = p_layer->kernels.d2;
    int in_d3 = p_layer->kernels.d3;
    
    int is_pad_depth = 0;
    if ((IS_PAD_AVX) && (p_layer->kernels.d3 < MIN_AVX_DEPTH)) {
        
        p_layer->kernels.d3 = MIN_AVX_DEPTH;
        is_pad_depth = 1;
        
        DNN_TRACE_1(" -- layer_init_kernels[%d]: pad d3: %d => %d\n", 
            p_layer->l_id, 
            in_d3,
            p_layer->kernels.d3);
        
    }
    
    allocate_matrices_data(&(p_layer->kernels), 
                           p_layer->kernels.n_matrices, 
                           p_layer->kernels.d1,
                           p_layer->kernels.d2,
                           p_layer->kernels.d3);
    
    init_matrices_from_file(
        &(p_layer->kernels),
        in_d1,
        in_d2,
        in_d3,
        p_layer->init_filename,
        is_pad_depth);
    
    DNN_TRACE_1("layer_init_kernels: initialized layer %d from %s\n", p_layer->l_id, p_layer->init_filename);
    
}

void dnn_input_init(dnn_t *p_dnn) {
    
    for (int i = 0; i < MAX_INPUT_FILES; i++) {
        p_g_cv_images[i] = new cv::Mat();
        
        p_g_cv_images[i]->create(p_dnn->input_d1, p_dnn->input_d2, CV_8UC1);
    }
    
    p_dnn->p_in_matrices->d1 = p_dnn->input_d1;
    p_dnn->p_in_matrices->d2 = p_dnn->input_d2;
    p_dnn->p_in_matrices->d3 = p_dnn->input_d3;
    
    p_dnn->p_in_matrices->n_matrices = p_dnn->input_n_matrices;
    
    if (p_dnn->input_type == INPUT_TYPE_DIR) {
        init_matrices_from_dir(p_dnn->p_in_matrices, p_dnn->input_filename);
    } else if (p_dnn->input_type == INPUT_TYPE_BINARY) {
        init_matrices_from_binary_file(p_dnn->p_in_matrices, p_dnn->input_filename);
    } else {
        init_matrices_from_file(
            p_dnn->p_in_matrices,
            p_dnn->p_in_matrices->d1,
            p_dnn->p_in_matrices->d2,
            p_dnn->p_in_matrices->d3,
            p_dnn->input_filename,
            0);
    }
    
    DNN_TRACE_1("dnn_input_init: initialized from %s\n", p_dnn->input_filename);
    
}

// void func_io_thread(void *p_param) {
//     dnn_t *p_dnn = (dnn_t *)p_param;
//     
//     init_matrices_from_dir(p_dnn->p_in_matrices, p_dnn->input_filename);
//     
// }
// 
// void dnn_io_thread_init(dnn_t *p_dnn) {
//     int res;
//     pthread_attr_init(&p_dnn->th_attr);
//     pthread_attr_setdetachstate(&p_dnn->th_attr, PTHREAD_CREATE_JOINABLE);
//     
//     res = pthread_create(&p_dnn->th_io, &p_dnn->th_attr, func_io_thread, (void *)(p_dnn));
//     DNN_ASSERT(res == 0);
// }

void dnn_set_input(
    dnn_t *p_dnn, 
    int in_id, 
    int is_pad_AVX) {
    
    p_dnn->p_in_matrices->d1 = p_dnn->input_d1;
    p_dnn->p_in_matrices->d2 = p_dnn->input_d2;
    if (is_pad_AVX) {
        DNN_TRACE_1(" -- dnn_set_input[%d]: pad d3: %d => %d\n", 
            in_id,
            p_dnn->input_3D_depth,
            MIN_AVX_DEPTH);
        
        p_dnn->p_in_matrices->d3 = MIN_AVX_DEPTH;
    } else {
        p_dnn->p_in_matrices->d3 = p_dnn->input_3D_depth;
    }
    
    p_dnn->p_in_matrices->n_matrices = 1;
    
    int cur_window_d1 = WINDOW_D1;
    int cur_window_d2 = WINDOW_D2;
    
    CILK_FOR_M (
        int cur_d1 = 0; 
        cur_d1 < p_dnn->p_in_matrices->d1; 
        cur_d1 += cur_window_d1) {

        int d1_start = cur_d1;

        int d1_finish = d1_start + cur_window_d1;

        if (d1_finish > p_dnn->p_in_matrices->d1) {
            d1_finish = p_dnn->p_in_matrices->d1;
        }

        CILK_FOR_M (
            int cur_d2 = 0; 
            cur_d2 < p_dnn->p_in_matrices->d2; 
            cur_d2 += cur_window_d2) {
            
            int d2_start = cur_d2;
            
            int d2_finish = d2_start + cur_window_d2;

            if (d2_finish > p_dnn->p_in_matrices->d2) {
                d2_finish = p_dnn->p_in_matrices->d2;
            }
            
            init_matrices_from_opencv(
                   p_dnn->p_in_matrices,
                   d1_start,
                   d2_start,
                   d1_finish,
                   d2_finish,
                   in_id, 
                   p_dnn->input_3D_depth, 
                   p_dnn->is_pad_3D,
                   is_pad_AVX);
        }
    }
   
    
    DNN_TRACE_1("initialized input %d\n", in_id);
    
}

void dnn_set_output(matrices_data_t *p_res_matrices, matrices_data_t *p_out_matrices, int in_id, int ch_limit) {
    
    int res_d1 = p_res_matrices->d1;
    int res_d2 = p_res_matrices->d2;
    int res_d3 = p_res_matrices->d3;
    
    DNN_ASSERT(res_d1 == p_out_matrices->d1);
    DNN_ASSERT(res_d2 == p_out_matrices->d2);
    DNN_ASSERT(res_d3 == p_out_matrices->d3);
    DNN_ASSERT(res_d3 == 1);
    
    for (int ch_id = 0; ch_id < ch_limit; ch_id++) {
        
        float *p_res_matrix = GET_MATRIX_PTR(p_res_matrices, ch_id);
        float *p_out_matrix = GET_MATRIX_PTR(p_out_matrices, (in_id * ch_limit) + ch_id);
    
        memcpy(p_out_matrix, p_res_matrix, sizeof(float) * DIMS_TO_SIZE(res_d1, res_d2, res_d3));
    }
    
}

void dnn_kernels_init(dnn_t *p_dnn) {
    
    CILK_FOR_M (int l_id = 0; l_id < p_dnn->n_layers; l_id++) {
        layer_init_kernels(&(p_dnn->p_layers[l_id]));
    }
    
}

void combine_matrices_depth_internal_1ch_8m(
    float *p_in_matrices, 
    int d1,
    int d2,
    int d3,
    int d1_start,
    int d2_start,
    int d1_finish,
    int d2_finish,
    int n_matrices,
    float *p_out_matrix) {
    
    DNN_TRACE_4("start %s\n", "");
    DNN_TRACE_4(" p_in_matrices: %p\n", p_in_matrices);
    DNN_TRACE_4(" d1,d2,d3: [%d,%d,%d]\n", d1, d2, d3);
    DNN_TRACE_4(" n_matrices: [%d]\n", n_matrices);
    DNN_TRACE_4(" p_out_matrix: %p\n", p_out_matrix);
    
    DNN_ASSERT(d3 == 1);
    DNN_ASSERT(n_matrices == 8);
    
    int out_d2 = d2;
    int out_d3 = d3 * n_matrices;
    
    for (int cur_d1 = d1_start; cur_d1 < d1_finish; cur_d1++) {

        int offset_d1 = D1_TO_OFFSET(cur_d1, d2, d3);
        int out_offset_d1 = D1_TO_OFFSET(cur_d1, out_d2, out_d3);
        
        for (int cur_d2 = d2_start; cur_d2 < d2_finish; cur_d2++) {

            int offset_d2 = D2_TO_OFFSET(cur_d2, d3);
            int out_offset_d2 = D2_TO_OFFSET(cur_d2, out_d3);
            int cur_idx = 0;
            
            for (int cur_m = 0; cur_m < 8; cur_m++) {
                
                p_out_matrix[out_offset_d1 + out_offset_d2 + cur_idx] = p_in_matrices[(DIMS_TO_SIZE(d1, d2, d3) * cur_m) + offset_d1 + offset_d2];
                cur_idx++;
                
            }
        }
    }
    
    DNN_TRACE_4("finish %s\n", "");    
}

void combine_matrices_depth_internal_1ch(
    float *p_in_matrices, 
    int d1,
    int d2,
    int d3,
    int d1_start,
    int d2_start,
    int d1_finish,
    int d2_finish,
    int n_matrices,
    float *p_out_matrix) {
    
    DNN_TRACE_4("start %s\n", "");
    DNN_TRACE_4(" p_in_matrices: %p\n", p_in_matrices);
    DNN_TRACE_4(" d1,d2,d3: [%d,%d,%d]\n", d1, d2, d3);
    DNN_TRACE_4(" n_matrices: [%d]\n", n_matrices);
    DNN_TRACE_4(" p_out_matrix: %p\n", p_out_matrix);
    
    DNN_ASSERT(d3 == 1);
    
    int out_d2 = d2;
    int out_d3 = d3 * n_matrices;
    
    for (int cur_d1 = d1_start; cur_d1 < d1_finish; cur_d1++) {

        int offset_d1 = D1_TO_OFFSET(cur_d1, d2, d3);
        int out_offset_d1 = D1_TO_OFFSET(cur_d1, out_d2, out_d3);
        
        for (int cur_d2 = d2_start; cur_d2 < d2_finish; cur_d2++) {

            int offset_d2 = D2_TO_OFFSET(cur_d2, d3);
            int out_offset_d2 = D2_TO_OFFSET(cur_d2, out_d3);
            int cur_idx = 0;
            
            for (int cur_m = 0; cur_m < n_matrices; cur_m++) {
                
                p_out_matrix[out_offset_d1 + out_offset_d2 + cur_idx] = p_in_matrices[(DIMS_TO_SIZE(d1, d2, d3) * cur_m) + offset_d1 + offset_d2];
                cur_idx++;
                
            }
        }
    }
    
    DNN_TRACE_4("finish %s\n", "");    
}

void combine_matrices_depth_internal_general(
    float *p_in_matrices, 
    int d1,
    int d2,
    int d3,
    int d1_start,
    int d2_start,
    int d1_finish,
    int d2_finish,
    int n_matrices,
    float *p_out_matrix) {
    
    DNN_TRACE_4("start %s\n", "");
    DNN_TRACE_4(" p_in_matrices: %p\n", p_in_matrices);
    DNN_TRACE_4(" d1,d2,d3: [%d,%d,%d]\n", d1, d2, d3);
    DNN_TRACE_4(" n_matrices: [%d]\n", n_matrices);
    DNN_TRACE_4(" p_out_matrix: %p\n", p_out_matrix);
    
    int out_d2 = d2;
    int out_d3 = d3 * n_matrices;
    
    for (int cur_d1 = d1_start; cur_d1 < d1_finish; cur_d1++) {

        int offset_d1 = D1_TO_OFFSET(cur_d1, d2, d3);
        int out_offset_d1 = D1_TO_OFFSET(cur_d1, out_d2, out_d3);
        
        for (int cur_d2 = d2_start; cur_d2 < d2_finish; cur_d2++) {

            int offset_d2 = D2_TO_OFFSET(cur_d2, d3);
            int out_offset_d2 = D2_TO_OFFSET(cur_d2, out_d3);
            int cur_idx = 0;
            
            for (int cur_m = 0; cur_m < n_matrices; cur_m++) {
                
                for (int cur_d3 = 0; cur_d3 < d3; cur_d3++) {
                    p_out_matrix[out_offset_d1 + out_offset_d2 + cur_idx] = p_in_matrices[(DIMS_TO_SIZE(d1, d2, d3) * cur_m) + offset_d1 + offset_d2 + cur_d3];
                    cur_idx++;
                }
            }
        }
    }
    
    DNN_TRACE_4("finish %s\n", "");    
}

void combine_matrices_depth_internal(
    float *p_in_matrices, 
    int d1,
    int d2,
    int d3,
    int d1_start,
    int d2_start,
    int d1_finish,
    int d2_finish,
    int n_matrices,
    float *p_out_matrix) {
    
    if ((n_matrices == 8) && (d3 == 1)) {
        combine_matrices_depth_internal_1ch_8m(
            p_in_matrices, 
            d1,
            d2,
            d3,
            d1_start,
            d2_start,
            d1_finish,
            d2_finish,
            n_matrices,
            p_out_matrix);
            
    } else if (d3 == 1) {
        combine_matrices_depth_internal_1ch(
            p_in_matrices, 
            d1,
            d2,
            d3,
            d1_start,
            d2_start,
            d1_finish,
            d2_finish,
            n_matrices,
            p_out_matrix);
            
    } else {
        combine_matrices_depth_internal_general(
            p_in_matrices, 
            d1,
            d2,
            d3,
            d1_start,
            d2_start,
            d1_finish,
            d2_finish,
            n_matrices,
            p_out_matrix);
            
    } 
}

void combine_matrices_depth(float *p_in_matrices, 
                            int d1,
                            int d2,
                            int d3,
                            int n_matrices,
                            float *p_out_matrix) {
    
    DNN_TRACE_4("start %s\n", "");
    DNN_TRACE_4(" p_in_matrices: %p\n", p_in_matrices);
    DNN_TRACE_4(" d1,d2,d3: [%d,%d,%d]\n", d1, d2, d3);
    DNN_TRACE_4(" n_matrices: [%d]\n", n_matrices);
    DNN_TRACE_4(" p_out_matrix: %p\n", p_out_matrix);
    
    int cur_window_d1 = WINDOW_D1;
    int cur_window_d2 = WINDOW_D2;
    
    CILK_FOR_M (int cur_d1 = 0; 
              cur_d1 < d1; 
              cur_d1 += cur_window_d1) {
                 
        int d1_start = cur_d1;
        int d1_finish = d1_start + cur_window_d1;

        if (d1_finish > d1) {
            d1_finish = d1;
        }

        CILK_FOR_M (int cur_d2 = 0; 
                  cur_d2 < d2; 
                  cur_d2 += cur_window_d2) {

            int d2_start = cur_d2;

            int d2_finish = d2_start + cur_window_d2;

            if (d2_finish > d2) {
                d2_finish = d2;
            }
            
            combine_matrices_depth_internal(
                p_in_matrices, 
                d1,
                d2,
                d3,
                d1_start,
                d2_start,
                d1_finish,
                d2_finish,
                n_matrices,
                p_out_matrix);
        }
    }
    
    DNN_TRACE_4("finish %s\n", "");    
}

void exec_conv(dnn_t *p_dnn, int l_id) {
    int out_d1, tmp_d1;
    int out_d2, tmp_d2;
    int out_d3, tmp_d3;
    int out_n_matrices, tmp_n_matrices;
    int n_channels_per_worker;
    TIMER_VAR(timer_1);
    TIMER_VAR_2(timer_2);
    layer_t * p_layer;
    
    p_layer = &(p_dnn->p_layers[l_id]);
    
    DNN_TRACE_2("layer[%d] start\n", p_layer->l_id);
    DNN_TRACE_2("   in_matrices: %d of shape[%d,%d,%d]\n", 
                p_dnn->p_in_matrices->n_matrices, 
                p_dnn->p_in_matrices->d1,
                p_dnn->p_in_matrices->d2,
                p_dnn->p_in_matrices->d3);
    
    DNN_TRACE_2("   kernels: %d of shape[%d,%d,%d]\n", 
                        p_layer->kernels.n_matrices, 
                        p_layer->kernels.d1,
                        p_layer->kernels.d2,
                        p_layer->kernels.d3);
    
    DNN_TRACE_2("   n_channels: %d n_stride: %d\n", 
                        p_layer->n_channels, 
                        p_layer->n_stride);
    
    DNN_ASSERT((p_layer->n_channels * p_layer->n_maxout) == p_layer->kernels.n_matrices);
    DNN_ASSERT(p_layer->kernels.d3 == p_dnn->p_in_matrices->d3);
    
    DNN_ASSERT(p_dnn->p_in_matrices->d1 >= p_layer->kernels.d1);
    DNN_ASSERT(p_dnn->p_in_matrices->d2 >= p_layer->kernels.d2);
    
    if ((p_layer->kernels.d3 < CONV_DEPTH_LIMIT) || (p_layer->n_channels != 32)) {
        n_channels_per_worker = 1;
    } else {
        n_channels_per_worker = N_CHANNELS_PER_WORKER;
    }
    
    out_d1 = OUT_DIM(p_dnn->p_in_matrices->d1, p_layer->kernels.d1, p_layer->n_stride);
    out_d2 = OUT_DIM(p_dnn->p_in_matrices->d2, p_layer->kernels.d2, p_layer->n_stride);
    out_d3 = n_channels_per_worker;
    
    out_n_matrices = (p_layer->n_channels / n_channels_per_worker) * p_dnn->p_in_matrices->n_matrices;
    
    set_matrices_size(p_dnn->p_out_matrices, out_n_matrices, out_d1, out_d2, out_d3);
    
    int is_conv3D_depth = 0;
    if (p_layer->kernels.d3 < CONV_DEPTH_LIMIT) {
        is_conv3D_depth = 1;
        abort();
    }
    
    if (is_conv3D_depth) {
        tmp_d1 = out_d1;
        tmp_d2 = out_d2;
        tmp_d3 = DIMS_TO_SIZE(p_layer->kernels.d1, p_layer->kernels.d2, p_layer->kernels.d3);
        tmp_n_matrices = p_dnn->p_in_matrices->n_matrices;
    
        set_matrices_size(p_dnn->p_tmp_matrices, tmp_n_matrices, tmp_d1, tmp_d2, tmp_d3);
    }
    
    DUMP_MATRICES(p_dnn->p_in_matrices, l_id, conv, DUMP_INPUT);
    DUMP_MATRICES(&(p_layer->kernels), l_id, conv, DUMP_KERNELS);
    
    DNN_TRACE_3("layer[%d] convolutions start\n", p_layer->l_id);
    
    START_TIMER(&timer_1);
    
    int cur_window_d1 = WINDOW_D1;
    int cur_window_d2 = WINDOW_D2;
        
    int k_d1 = p_layer->kernels.d1;
    int k_d2 = p_layer->kernels.d2;
    int k_d3 = p_layer->kernels.d3;

#ifdef IS_QTD
    // QTD
    //if (p_dnn->p_in_matrices->d3 == p_layer->n_channels) {
        DNN_TRACE_3("layer[%d] qtd_decompose start\n", p_layer->l_id);
        START_TIMER(&timer_1);
        
        float qtd_threashold = QTD_THREASHOLD;
        if (p_dnn->p_in_matrices->d1 == 2048) {
            qtd_threashold = QTD_THREASHOLD_INPUT;
        }
        
        for (int m_id = 0; m_id < p_dnn->p_in_matrices->n_matrices; m_id++) {

            RESET_SP_OUT_MATRIX_PTR(p_dnn->p_in_matrices, m_id);
            RESET_SP_MASK_MATRIX_PTR(p_dnn->p_in_matrices, m_id);

            qtd_decompose(
                GET_MATRIX_PTR(p_dnn->p_in_matrices, m_id),
                p_dnn->p_in_matrices->d1,
                p_dnn->p_in_matrices->d2,
                p_dnn->p_in_matrices->d3,
                qtd_threashold, 
                GET_SP_OUT_MATRIX_PTR(p_dnn->p_in_matrices, m_id),
                GET_SP_MASK_MATRIX_PTR(p_dnn->p_in_matrices, m_id));

        }

        STOP_TIMER(&timer_1, "qtd_decompose time:");
        DNN_TRACE_3("layer[%d] qtd_decompose finish\n", p_layer->l_id);

    //}
    // QTD
#endif
    CILK_FOR_M (int m_id = 0; m_id < p_dnn->p_in_matrices->n_matrices; m_id++) {
        
        if (is_conv3D_depth) {
            abort();
            START_TIMER_2(&timer_2);
            
            CILK_FOR_M (int cur_d1 = 0; 
                      cur_d1 < p_dnn->p_in_matrices->d1; 
                      cur_d1 += (cur_window_d1 - k_d1 + 1)) {

                int d1_start = cur_d1;

                if ((d1_start + k_d1) > p_dnn->p_in_matrices->d1) {
                    continue;
                }

                int d1_finish = d1_start + cur_window_d1;

                if (d1_finish > p_dnn->p_in_matrices->d1) {
                    d1_finish = p_dnn->p_in_matrices->d1;
                }

                CILK_FOR_M (int cur_d2 = 0; 
                          cur_d2 < p_dnn->p_in_matrices->d2; 
                          cur_d2 += (cur_window_d2 - k_d2 + 1)) {
                    
                    int d2_start = cur_d2;

                    if ((d2_start + k_d2) > p_dnn->p_in_matrices->d2) {
                        continue;
                    }

                    int d2_finish = d2_start + cur_window_d2;

                    if (d2_finish > p_dnn->p_in_matrices->d2) {
                        d2_finish = p_dnn->p_in_matrices->d2;
                    }

                    matrix_to_conv_depth(
                        GET_MATRIX_PTR(p_dnn->p_in_matrices, m_id),
                        p_dnn->p_in_matrices->d1,
                        p_dnn->p_in_matrices->d2,
                        d1_start,
                        d2_start,
                        d1_finish,
                        d2_finish,
                        p_layer->kernels.d1,
                        p_layer->kernels.d2,
                        p_layer->kernels.d3,
                        p_layer->n_stride,
                        GET_MATRIX_PTR(p_dnn->p_tmp_matrices, m_id));   
                }
            }
            
            STOP_TIMER_2(&timer_2, "matrix_to_conv_depth time:");         
        }
        
        START_TIMER_2(&timer_2);                
        
        matrices_data_t *p_in_matrices = p_dnn->p_in_matrices;
        if (is_conv3D_depth) {
            abort();
            k_d1 = 1;
            k_d2 = 1;
            k_d3 = p_layer->kernels.d1 * p_layer->kernels.d2 * p_layer->kernels.d3;

            p_in_matrices = p_dnn->p_tmp_matrices;
            
            // // QTD
            //             DNN_TRACE_3("layer[%d] qtd_decompose start\n", p_layer->l_id);
            //             START_TIMER(&timer_1);
            // 
            //             for (int m_id = 0; m_id < p_dnn->p_in_matrices->n_matrices; m_id++) {
            // 
            //                 RESET_SP_OUT_MATRIX_PTR(p_dnn->p_in_matrices, m_id);
            //                 RESET_SP_MASK_MATRIX_PTR(p_dnn->p_in_matrices, m_id);
            // 
            //                 qtd_decompose(
            //                     GET_MATRIX_PTR(p_dnn->p_in_matrices, m_id),
            //                     p_dnn->p_out_matrices->d1,
            //                     p_dnn->p_out_matrices->d2,
            //                     p_dnn->p_out_matrices->d3,
            //                     QTD_THREASHOLD, 
            //                     GET_SP_OUT_MATRIX_PTR(p_dnn->p_in_matrices, m_id),
            //                     GET_SP_MASK_MATRIX_PTR(p_dnn->p_in_matrices, m_id));
            // 
            //             }
            // 
            //             STOP_TIMER(&timer_1, "qtd_decompose time:");
            //             DNN_TRACE_3("layer[%d] qtd_decompose finish\n", p_layer->l_id);
            //             // QTD
            
        }
        
        if ((p_in_matrices->d1 <= WINDOW_D1) || 
            (p_in_matrices->d2 <= WINDOW_D2)) {
            cur_window_d1 = p_in_matrices->d1;
            cur_window_d2 = p_in_matrices->d2;
        }

        // if (/*(k_d3 != 32) ||*/ (p_layer->n_maxout != 2)) {
        //             cur_window_d1 = p_in_matrices->d1;
        //             cur_window_d2 = p_in_matrices->d2;
        //         }
               
        CILK_FOR_M (int cur_d1 = 0; 
                  cur_d1 < p_in_matrices->d1; 
                  cur_d1 += (cur_window_d1 - k_d1 + 1)) {
                     
            int d1_start = cur_d1;

            if ((d1_start + k_d1) > p_in_matrices->d1) {
                continue;
            }

            int d1_finish = d1_start + cur_window_d1;

            if (d1_finish > p_in_matrices->d1) {
                d1_finish = p_in_matrices->d1;
            }

            CILK_FOR_M (int cur_d2 = 0; 
                      cur_d2 < p_in_matrices->d2; 
                      cur_d2 += (cur_window_d2 - k_d2 + 1)) {

                int d2_start = cur_d2;

                if ((d2_start + k_d2) > p_in_matrices->d2) {
                    continue;
                }

                int d2_finish = d2_start + cur_window_d2;

                if (d2_finish > p_in_matrices->d2) {
                    d2_finish = p_in_matrices->d2;
                }

                CILK_FOR_M (int k_id = 0; k_id < p_layer->n_channels; k_id += n_channels_per_worker) {

                    conv3D(

                        GET_MATRIX_PTR(p_in_matrices, m_id),
                        
                        GET_SP_OUT_MATRIX_PTR(p_dnn->p_in_matrices, m_id),
                        GET_SP_MASK_MATRIX_PTR(p_dnn->p_in_matrices, m_id),
                        
                        p_in_matrices->d1,
                        p_in_matrices->d2,
                        d1_start,
                        d2_start,
                        d1_finish,
                        d2_finish,

                        GET_MATRIX_PTR(&(p_layer->kernels), (k_id * p_layer->n_maxout)),
                        GET_AUX_DATA_PTR(&(p_layer->kernels), (k_id * p_layer->n_maxout)),
                        (n_channels_per_worker * p_layer->n_maxout),
                        p_layer->n_maxout,
                        k_d1,
                        k_d2,
                        k_d3,
                        p_layer->n_stride,

                        GET_MATRIX_PTR(p_dnn->p_out_matrices, 
                        (m_id * (p_layer->n_channels / n_channels_per_worker)) + (k_id / n_channels_per_worker))
                        );
                }

            }
        }

        STOP_TIMER_2(&timer_2, "conv3D time:");
    }
    
    STOP_TIMER(&timer_1, "conv-time:");
    
    DNN_TRACE_3("layer[%d] convolutions finish\n", p_layer->l_id);
    
    set_matrices_size(p_dnn->p_in_matrices, 
                      p_dnn->p_in_matrices->n_matrices, 
                      out_d1, 
                      out_d2, 
                      p_layer->n_channels);
    
    DNN_TRACE_3("layer[%d] combine_matrices_depth start\n", p_layer->l_id);
    
    START_TIMER(&timer_1);
        
    CILK_FOR_M (int m_id = 0; m_id < p_dnn->p_in_matrices->n_matrices; m_id++) {
        
        combine_matrices_depth(
            GET_MATRIX_PTR(p_dnn->p_out_matrices, m_id * (p_layer->n_channels / n_channels_per_worker)),
            out_d1,
            out_d2,
            out_d3,
            (p_layer->n_channels / n_channels_per_worker),
            GET_MATRIX_PTR(p_dnn->p_in_matrices, m_id));
    }
    
    STOP_TIMER(&timer_1, "2D-to-3D time:");
    
    DNN_TRACE_3("layer[%d] combine_matrices_depth finish\n", p_layer->l_id);
    
    DUMP_MATRICES(p_dnn->p_in_matrices, l_id, conv, DUMP_OUTPUT);
    
    DNN_TRACE_2("layer[%d] finish\n", p_layer->l_id);
}

void exec_pool(dnn_t *p_dnn, int l_id) {
    int out_d1;
    int out_d2;
    int out_d3;
    int n_matrices_in_block;
    int out_n_matrices;
    int n_stride_d1 = 0;
    int n_stride_d2 = 0;
    TIMER_VAR(timer);
    layer_t * p_layer;
    matrices_data_t *p_temp;
    
    p_layer = &(p_dnn->p_layers[l_id]);
    
    DNN_TRACE_2("layer[%d] start\n", p_layer->l_id);
    DNN_TRACE_2("   in_matrices: %d of shape[%d,%d,%d]\n", 
                p_dnn->p_in_matrices->n_matrices, 
                p_dnn->p_in_matrices->d1,
                p_dnn->p_in_matrices->d2,
                p_dnn->p_in_matrices->d3);
    
    DNN_TRACE_2("   kernels: %d of shape[%d,%d,%d]\n", 
                        p_layer->kernels.n_matrices, 
                        p_layer->kernels.d1,
                        p_layer->kernels.d2,
                        p_layer->kernels.d3);
    
    DNN_TRACE_2("   n_channels: %d n_stride: %d\n", 
                        p_layer->n_channels, 
                        p_layer->n_stride);
    
    DNN_ASSERT(p_layer->kernels.d3 == -1);
    DNN_ASSERT(p_layer->kernels.n_matrices == -1);
    //DNN_ASSERT(p_layer->kernels.n_size == -1);
    DNN_ASSERT(p_layer->kernels.p_data == NULL);
    DNN_ASSERT(p_layer->n_channels == p_dnn->p_in_matrices->d3);
    
    DNN_ASSERT(p_dnn->p_in_matrices->d1 >= p_layer->kernels.d1);
    DNN_ASSERT(p_dnn->p_in_matrices->d2 >= p_layer->kernels.d2);
    
    out_d1 = OUT_DIM(p_dnn->p_in_matrices->d1, p_layer->kernels.d1, p_layer->n_stride);
    out_d2 = OUT_DIM(p_dnn->p_in_matrices->d2, p_layer->kernels.d2, p_layer->n_stride);
    out_d3 = p_layer->n_channels;
    
    if (p_layer->kernels.n_size == 0) {
        n_matrices_in_block = 1;
        n_stride_d1 += 1;
        n_stride_d2 += 1;
        out_n_matrices = n_matrices_in_block * p_dnn->p_in_matrices->n_matrices;
    } else {
        n_stride_d1 += p_layer->n_stride;
        n_stride_d2 += p_layer->n_stride;
        n_matrices_in_block = p_layer->n_stride * p_layer->n_stride;
        out_n_matrices = n_matrices_in_block * p_dnn->p_in_matrices->n_matrices;
    }
    
    set_matrices_size(p_dnn->p_out_matrices, out_n_matrices, out_d1, out_d2, out_d3);
    
    DUMP_MATRICES(p_dnn->p_in_matrices, p_layer->l_id, pool, DUMP_INPUT);
    
    DNN_TRACE_3("layer[%d] max-pool start\n", p_layer->l_id);
    
    START_TIMER(&timer);
    
    int cur_window_d1 = WINDOW_POOL_D1;
    int cur_window_d2 = WINDOW_POOL_D2;
    
    CILK_FOR_M (int m_id = 0; m_id < p_dnn->p_in_matrices->n_matrices; m_id++) {
        
        CILK_FOR_M (int shift_d1 = 0; shift_d1 < n_stride_d1; shift_d1++) {

            CILK_FOR_M (int shift_d2 = 0; shift_d2 < n_stride_d1; shift_d2++) {
                
                int k_d1 = p_layer->kernels.d1;
                int k_d2 = p_layer->kernels.d2;
                
                int m_d1 = p_dnn->p_in_matrices->d1;
                int m_d2 = p_dnn->p_in_matrices->d2;
                
                int pool_d1 = m_d1 + shift_d1;
                int pool_d2 = m_d2 + shift_d2;
    
                CILK_FOR_M (int cur_d1 = shift_d1; 
                          cur_d1 < pool_d1; 
                          cur_d1 += (cur_window_d1 - k_d1 + 1)) {

                    int d1_start = cur_d1;
                    
                    if (((d1_start - shift_d1) % 2) != 0) {
                        d1_start--;
                    }
                    
                    if ((d1_start + k_d1) > pool_d1) {
                        continue;
                    }

                    int d1_finish = d1_start + cur_window_d1;
                    
                    if (d1_finish > pool_d1) {
                        d1_finish = pool_d1;
                    }

                    CILK_FOR_M (int cur_d2 = shift_d2; 
                              cur_d2 < pool_d2; 
                              cur_d2 += (cur_window_d2 - k_d2 + 1)) {

                        int d2_start = cur_d2;
                        
                        if (((d2_start - shift_d2) % 2) != 0) {
                            d2_start--;
                        }
                        
                        if ((d2_start + k_d2) > pool_d2) {
                            continue;
                        }

                        int d2_finish = d2_start + cur_window_d2;
                        
                        if (d2_finish > pool_d2) {
                            d2_finish = pool_d2;
                        }
                                                
                        max_pool(
                            GET_MATRIX_PTR(p_dnn->p_in_matrices, m_id),  
                            m_d1,
                            m_d2,
                            d1_start,
                            d2_start,
                            d1_finish,
                            d2_finish,
                            k_d1,
                            k_d2,
                            p_layer->n_channels,
                            p_layer->n_stride,
                            GET_MATRIX_PTR(p_dnn->p_out_matrices, 
                                (n_matrices_in_block * m_id) + (shift_d1 * p_layer->n_stride) + shift_d2));
                    }
                }
            }
        }
    }
    
    STOP_TIMER(&timer, "pool-time:");
    
    DNN_TRACE_3("layer[%d] max-pool finish\n", p_layer->l_id);
    
    p_temp = p_dnn->p_in_matrices;
    p_dnn->p_in_matrices = p_dnn->p_out_matrices;
    p_dnn->p_out_matrices = p_temp;
    
    DUMP_MATRICES(p_dnn->p_in_matrices, p_layer->l_id, pool, DUMP_OUTPUT);
    
    DNN_TRACE_3("layer[%d] finish\n", p_layer->l_id);
    
}

void exec_softmax(dnn_t *p_dnn, int l_id) {
    int out_d1;
    int out_d2;
    int out_d3;
    int out_n_matrices;
    TIMER_VAR(timer);
    layer_t * p_layer;
    matrices_data_t *p_temp;
    
    p_layer = &(p_dnn->p_layers[l_id]);
    
    DNN_TRACE_2("layer[%d] start\n", p_layer->l_id);
    DNN_TRACE_2("   in_matrices: %d of shape[%d,%d,%d]\n", 
                p_dnn->p_in_matrices->n_matrices, 
                p_dnn->p_in_matrices->d1,
                p_dnn->p_in_matrices->d2,
                p_dnn->p_in_matrices->d3);
    
    DNN_TRACE_2("   n_channels: %d n_stride: %d\n", 
                        p_layer->n_channels, 
                        p_layer->n_stride);
    
    DNN_ASSERT(p_layer->n_channels == p_dnn->p_in_matrices->d3);
        
    out_d1 = p_dnn->p_in_matrices->d1;
    out_d2 = p_dnn->p_in_matrices->d2;
    out_d3 = p_dnn->p_in_matrices->d3;
    
    out_n_matrices = p_dnn->p_in_matrices->n_matrices;
    
    set_matrices_size(p_dnn->p_out_matrices, out_n_matrices, out_d1, out_d2, out_d3);
    
    DUMP_MATRICES(p_dnn->p_in_matrices, p_layer->l_id, softmax, DUMP_INPUT);
    
    DNN_TRACE_3("layer[%d] softmax start\n", p_layer->l_id);
    
    START_TIMER(&timer);
    
    CILK_FOR_M (int m_id = 0; m_id < p_dnn->p_in_matrices->n_matrices; m_id++) {
        softmax(GET_MATRIX_PTR(p_dnn->p_in_matrices, m_id),
                p_dnn->p_in_matrices->d1,
                p_dnn->p_in_matrices->d2,
                p_dnn->p_in_matrices->d3, 
                GET_MATRIX_PTR(p_dnn->p_out_matrices, m_id));
    }
    
    STOP_TIMER(&timer, "softmax-time:");
    
    DNN_TRACE_3("layer[%d] softmax finish\n", p_layer->l_id);
    
    p_temp = p_dnn->p_in_matrices;
    p_dnn->p_in_matrices = p_dnn->p_out_matrices;
    p_dnn->p_out_matrices = p_temp;
    
    DUMP_MATRICES(p_dnn->p_in_matrices, p_layer->l_id, softmax, DUMP_OUTPUT);
    
    DNN_TRACE_3("layer[%d] finish\n", p_layer->l_id);
    
}

void exec_interweave(dnn_t *p_dnn, int l_id) {
    int out_d1;
    int out_d2;
    int out_d3;
    int out_n_matrices;
    TIMER_VAR(timer);
    layer_t * p_layer;
    matrices_data_t *p_temp;
    
    p_layer = &(p_dnn->p_layers[l_id]);
    
    DNN_TRACE_2("layer[%d] start\n", p_layer->l_id);
    DNN_TRACE_2("   in_matrices: %d of shape[%d,%d,%d]\n", 
                p_dnn->p_in_matrices->n_matrices, 
                p_dnn->p_in_matrices->d1,
                p_dnn->p_in_matrices->d2,
                p_dnn->p_in_matrices->d3);
    
    DNN_TRACE_2("   n_channels: %d n_stride: %d\n", 
                        p_layer->n_channels, 
                        p_layer->n_stride);
    
    DNN_ASSERT(p_layer->n_channels == p_dnn->p_in_matrices->n_matrices);
    
    // FIX FIX    
    if ((p_dnn->input_patch_leg != -1) && (FIX_FC_OUT_SIZE)) {
        out_d1 = (p_dnn->input_d1 - (p_dnn->input_patch_leg * 2)) / FIX_FC_OUT_SIZE_DIVIDE;
        out_d2 = (p_dnn->input_d2 - (p_dnn->input_patch_leg * 2)) / FIX_FC_OUT_SIZE_DIVIDE;
    } else {
        out_d1 = p_dnn->p_in_matrices->d1 * p_layer->n_stride;
        out_d2 = p_dnn->p_in_matrices->d2 * p_layer->n_stride;   
    }
    out_d3 = 1;
    
    out_n_matrices = p_dnn->n_output_channels;
        
    set_matrices_size(p_dnn->p_out_matrices, out_n_matrices, out_d1, out_d2, out_d3);
    
    DUMP_MATRICES(p_dnn->p_in_matrices, p_layer->l_id, interweave, DUMP_INPUT);
    
    DNN_TRACE_3("layer[%d] interweave start\n", p_layer->l_id);
    
    START_TIMER(&timer);

    for (int ch_id = 0; ch_id < out_n_matrices; ch_id++) {
        recursive_interweave(
            GET_MATRIX_PTR(p_dnn->p_in_matrices, 0),
            p_dnn->p_in_matrices->n_matrices,
            p_dnn->p_in_matrices->d1,
            p_dnn->p_in_matrices->d2,
            p_dnn->p_in_matrices->d3,
            ch_id,
            0,
            0,
            1,
            GET_MATRIX_PTR(p_dnn->p_out_matrices, ch_id),
            out_d1,
            out_d2);
    }
    STOP_TIMER(&timer, "interweave-time:");
    
    DNN_TRACE_3("layer[%d] interweave finish\n", p_layer->l_id);
    
    p_temp = p_dnn->p_in_matrices;
    p_dnn->p_in_matrices = p_dnn->p_out_matrices;
    p_dnn->p_out_matrices = p_temp;
    
    DUMP_MATRICES(p_dnn->p_in_matrices, p_layer->l_id, interweave, DUMP_OUTPUT);
    
    DNN_TRACE_3("layer[%d] finish\n", p_layer->l_id);
    
}

/////////////////////////////////////////////////////////////////////////////////////////
// EXTERNAL FUNCTIONS
/////////////////////////////////////////////////////////////////////////////////////////
void dnn_execute(dnn_t *p_dnn)
{    
    TIMER_VAR(timer1);
    TIMER_VAR(timer2);
    matrices_data_t *p_matrices_1 = NULL;
    matrices_data_t *p_matrices_2 = NULL;
    matrices_data_t *p_matrices_3 = NULL;
    matrices_data_t *p_matrices_4 = NULL;
    layer_t *p_layer = NULL;
    
    DNN_TRACE_1("Matrices buffer max size: %ld\n", 
        DIMS_TO_SIZE(p_dnn->input_d1, 
                     p_dnn->input_d2, 
                     p_dnn->input_d3) * MAX_CHANNELS * p_dnn->input_n_matrices * sizeof(float));
    
    p_matrices_1 = allocate_matrices(MAX_CHANNELS * 1, p_dnn->input_d1, p_dnn->input_d2, p_dnn->input_d3);
    p_matrices_2 = allocate_matrices(MAX_CHANNELS * 1, p_dnn->input_d1, p_dnn->input_d2, p_dnn->input_d3);
    p_matrices_3 = allocate_matrices(MAX_CHANNELS * 1, p_dnn->input_d1, p_dnn->input_d2, p_dnn->input_d3);
            
    if (p_dnn->input_type == INPUT_TYPE_DIR) {
        p_matrices_4 = allocate_matrices(MAX_INPUT_FILES, p_dnn->input_d1, p_dnn->input_d2, p_dnn->input_d3);
    }
    
    p_dnn->p_in_matrices = p_matrices_1;
    p_dnn->p_out_matrices = p_matrices_2;
    p_dnn->p_tmp_matrices = p_matrices_3;
    
    //dnn_init_io_thread(p_dnn)
    dnn_input_init(p_dnn);    
    dnn_kernels_init(p_dnn);
    
    //printf("INPUT MATRIX\n");
    //fp_print_matrices(stdout, p_dnn->p_in_matrices);
    //printf("CONV1 KERNELS\n");
    //fp_print_matrices(stdout, &(p_dnn->p_layers[0].kernels));
    
    START_TIMER(&timer1);
    
    int n_in_matrices = 1;
    if (p_dnn->input_type == INPUT_TYPE_DIR) {
        n_in_matrices = p_dnn->input_n_matrices;
    }
    
    int n_in_matrices_3D = n_in_matrices - p_dnn->input_3D_depth + 1;
    
    if (p_dnn->is_pad_3D) {
        n_in_matrices_3D = n_in_matrices;
    }
    
    for (int in_id = 0; in_id < n_in_matrices_3D; in_id++) {
        if (p_dnn->input_type == INPUT_TYPE_DIR) {
            START_TIMER(&timer2);
            dnn_set_input(p_dnn, in_id, IS_PAD_AVX);
            STOP_TIMER(&timer2, "set-input-time:");
        }
        
        for (int l_id = 0; l_id < p_dnn->n_layers; l_id++) {

            p_layer = &(p_dnn->p_layers[l_id]);

            if (p_layer->op_type == OP_CONV) {
                exec_conv(p_dnn, l_id);

            } else if (p_layer->op_type == OP_POOL) {
                exec_pool(p_dnn, l_id);

            } else if (p_layer->op_type == OP_SOFTMAX) {
                exec_softmax(p_dnn, l_id);

            } else if (p_layer->op_type == OP_INTERWEAVE) {
                exec_interweave(p_dnn, l_id);

            } else {
                DNN_ABORT("layer[%d] op_type[%d] is undefined\n", p_layer->l_id, p_layer->op_type);
            }
        }
        
        if (p_dnn->input_type == INPUT_TYPE_DIR) {
            if (in_id == 0) {    
                set_matrices_size(
                    p_matrices_4, 
                    (n_in_matrices_3D * p_dnn->n_output_channels), 
                    p_dnn->p_in_matrices->d1, 
                    p_dnn->p_in_matrices->d2,
                    1);
            }
            
            START_TIMER(&timer2);
            dnn_set_output(p_dnn->p_in_matrices, p_matrices_4, in_id, p_dnn->n_output_channels);
            STOP_TIMER(&timer2, "set-output-time:");
        }
        
    }
    
    STOP_TIMER(&timer1, "total-time:");
    
    if (p_dnn->input_type == INPUT_TYPE_DIR) { 
        output_matrices_to_files(p_matrices_4, 
                                 p_dnn->n_output_channels,
                                 p_dnn->output_filename);
    } else {
        output_matrices_to_binary_file(p_dnn->p_in_matrices,
                                       p_dnn->output_filename);
    }
    
    printf("-= PROC SUCCESS =-\n");
    
}