#pragma once

#include <stddef.h>

void im_rms_norm(float * out, const float * input, const float * weight, size_t n, float eps);
void im_apply_rope(float * vector, size_t len, size_t head_dim, double position, double freq_base);
void im_softmax(float * values, size_t n);
void im_attention(float * out, const float * query, const float * keys, const float * values, size_t rows, size_t head_dim, size_t value_dim, float scale);

