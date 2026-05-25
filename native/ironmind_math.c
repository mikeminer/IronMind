#include "ironmind_math.h"

#include <math.h>
#include <string.h>

void im_rms_norm(float * out, const float * input, const float * weight, size_t n, float eps) {
    double sum_sq = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum_sq += (double) input[i] * (double) input[i];
    }
    const float scale = 1.0f / sqrtf((float) (sum_sq / (double) n) + eps);
    for (size_t i = 0; i < n; i++) {
        out[i] = input[i] * scale * weight[i];
    }
}

void im_apply_rope(float * vector, size_t len, size_t head_dim, double position, double freq_base) {
    if (head_dim == 0 || (head_dim % 2) != 0) return;
    for (size_t offset = 0; offset + head_dim <= len; offset += head_dim) {
        for (size_t i = 0; i < head_dim; i += 2) {
            const double theta = position * pow(freq_base, -((double) i / (double) head_dim));
            const float c = (float) cos(theta);
            const float s = (float) sin(theta);
            const float x0 = vector[offset + i];
            const float x1 = vector[offset + i + 1];
            vector[offset + i] = x0 * c - x1 * s;
            vector[offset + i + 1] = x0 * s + x1 * c;
        }
    }
}

void im_softmax(float * values, size_t n) {
    if (n == 0) return;
    float max = values[0];
    for (size_t i = 1; i < n; i++) {
        if (values[i] > max) max = values[i];
    }

    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        values[i] = expf(values[i] - max);
        sum += values[i];
    }
    for (size_t i = 0; i < n; i++) {
        values[i] = (float) ((double) values[i] / sum);
    }
}

static float im_dot(const float * a, const float * b, size_t n) {
    float out = 0.0f;
    for (size_t i = 0; i < n; i++) out += a[i] * b[i];
    return out;
}

void im_attention(float * out, const float * query, const float * keys, const float * values, size_t rows, size_t head_dim, size_t value_dim, float scale) {
    memset(out, 0, value_dim * sizeof(float));
    if (rows == 0) return;

    float scores[1024];
    if (rows > sizeof(scores) / sizeof(scores[0])) return;

    for (size_t row = 0; row < rows; row++) {
        scores[row] = im_dot(query, keys + row * head_dim, head_dim) * scale;
    }
    im_softmax(scores, rows);

    for (size_t row = 0; row < rows; row++) {
        const float p = scores[row];
        const float * value = values + row * value_dim;
        for (size_t col = 0; col < value_dim; col++) {
            out[col] += p * value[col];
        }
    }
}

