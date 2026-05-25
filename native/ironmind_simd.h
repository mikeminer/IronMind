#pragma once

#include <stddef.h>

typedef enum im_simd_backend {
    IM_SIMD_SCALAR = 0,
    IM_SIMD_AVX2 = 1,
    IM_SIMD_AVX512F = 2
} im_simd_backend;

float im_dot_f32(const float * a, const float * b, size_t n);
im_simd_backend im_simd_selected_backend(void);
const char * im_simd_backend_name(im_simd_backend backend);
