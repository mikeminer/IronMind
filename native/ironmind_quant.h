#pragma once

#include <stddef.h>
#include <stdint.h>

enum im_ggml_type {
    IM_GGML_TYPE_F32 = 0,
    IM_GGML_TYPE_F16 = 1,
    IM_GGML_TYPE_Q4_0 = 2,
    IM_GGML_TYPE_Q4_1 = 3,
    IM_GGML_TYPE_Q5_0 = 6,
    IM_GGML_TYPE_Q5_1 = 7,
    IM_GGML_TYPE_Q8_0 = 8,
    IM_GGML_TYPE_Q8_1 = 9,
    IM_GGML_TYPE_Q2_K = 10,
    IM_GGML_TYPE_Q3_K = 11,
    IM_GGML_TYPE_Q4_K = 12,
    IM_GGML_TYPE_Q5_K = 13,
    IM_GGML_TYPE_Q6_K = 14,
    IM_GGML_TYPE_Q8_K = 15,
    IM_GGML_TYPE_I8 = 24,
    IM_GGML_TYPE_I16 = 25,
    IM_GGML_TYPE_I32 = 26,
    IM_GGML_TYPE_I64 = 27,
    IM_GGML_TYPE_F64 = 28,
    IM_GGML_TYPE_BF16 = 30
};

typedef struct im_quant_type_info {
    const char * name;
    uint32_t block_size;
    uint32_t type_size;
    int quantized;
    int matvec_supported;
} im_quant_type_info;

const im_quant_type_info * im_quant_type_info_for(int32_t type);
const char * im_quant_type_name(int32_t type);
size_t im_quant_row_size(int32_t type, size_t cols);
size_t im_quant_tensor_size(int32_t type, const uint64_t * dims, uint32_t n_dims);
float im_f16_to_f32(uint16_t h);
float im_bf16_to_f32(uint16_t h);
int im_dequantize_row(float * out, const void * row, int32_t type, size_t cols);
int im_quant_has_direct_dot(int32_t type);
int im_quant_matvec(float * out, const void * matrix, int32_t type, size_t rows, size_t cols, const float * vector);
