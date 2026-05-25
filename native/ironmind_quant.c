#include "ironmind_quant.h"

#include "ironmind_simd.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static const im_quant_type_info IM_TYPE_INFO[] = {
    {"F32", 1, 4, 0, 1},
    {"F16", 1, 2, 0, 1},
    {"Q4_0", 32, 18, 1, 1},
    {"Q4_1", 32, 20, 1, 1},
    {"UNKNOWN", 0, 0, 0, 0},
    {"UNKNOWN", 0, 0, 0, 0},
    {"Q5_0", 32, 22, 1, 1},
    {"Q5_1", 32, 24, 1, 1},
    {"Q8_0", 32, 34, 1, 1},
    {"Q8_1", 32, 40, 1, 1},
    {"Q2_K", 256, 84, 1, 0},
    {"Q3_K", 256, 110, 1, 0},
    {"Q4_K", 256, 144, 1, 1},
    {"Q5_K", 256, 176, 1, 0},
    {"Q6_K", 256, 210, 1, 1},
    {"Q8_K", 256, 292, 1, 0},
};

const im_quant_type_info * im_quant_type_info_for(int32_t type) {
    static const im_quant_type_info i8 = {"I8", 1, 1, 0, 0};
    static const im_quant_type_info i16 = {"I16", 1, 2, 0, 0};
    static const im_quant_type_info i32 = {"I32", 1, 4, 0, 0};
    static const im_quant_type_info i64 = {"I64", 1, 8, 0, 0};
    static const im_quant_type_info f64 = {"F64", 1, 8, 0, 0};
    static const im_quant_type_info bf16 = {"BF16", 1, 2, 0, 1};

    if (type >= 0 && type < (int32_t)(sizeof(IM_TYPE_INFO) / sizeof(IM_TYPE_INFO[0])) && IM_TYPE_INFO[type].block_size) return &IM_TYPE_INFO[type];
    if (type == IM_GGML_TYPE_I8) return &i8;
    if (type == IM_GGML_TYPE_I16) return &i16;
    if (type == IM_GGML_TYPE_I32) return &i32;
    if (type == IM_GGML_TYPE_I64) return &i64;
    if (type == IM_GGML_TYPE_F64) return &f64;
    if (type == IM_GGML_TYPE_BF16) return &bf16;
    return NULL;
}

const char * im_quant_type_name(int32_t type) {
    const im_quant_type_info * info = im_quant_type_info_for(type);
    return info ? info->name : "UNKNOWN";
}

size_t im_quant_row_size(int32_t type, size_t cols) {
    const im_quant_type_info * info = im_quant_type_info_for(type);
    if (!info || !info->block_size || !info->type_size) return 0;
    const size_t blocks = (cols + info->block_size - 1u) / info->block_size;
    return blocks * info->type_size;
}

size_t im_quant_tensor_size(int32_t type, const uint64_t * dims, uint32_t n_dims) {
    if (!dims || n_dims == 0) return 0;
    const im_quant_type_info * info = im_quant_type_info_for(type);
    if (!info) return 0;
    uint64_t rows = 1;
    for (uint32_t i = 1; i < n_dims; i++) {
        if (dims[i] && rows > UINT64_MAX / dims[i]) return 0;
        rows *= dims[i];
    }
    const size_t row = im_quant_row_size(type, (size_t)dims[0]);
    if (row == 0 || rows > SIZE_MAX / row) return 0;
    return (size_t)rows * row;
}

static uint16_t read_le16(const void * p) {
    const uint8_t * b = (const uint8_t *)p;
    return (uint16_t)b[0] | ((uint16_t)b[1] << 8);
}

static uint32_t read_le32(const void * p) {
    const uint8_t * b = (const uint8_t *)p;
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

float im_f16_to_f32(uint16_t h) {
    const uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)h & 0x03ffu;
    uint32_t out;

    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            out = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        out = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &out, sizeof(f));
    return f;
}

float im_bf16_to_f32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static void dequant_q4_0(float * out, const uint8_t * row, size_t cols) {
    const size_t blocks = cols / 32u;
    for (size_t block = 0; block < blocks; block++) {
        const uint8_t * p = row + block * 18u;
        const float d = im_f16_to_f32(read_le16(p));
        const uint8_t * qs = p + 2;
        float * dst = out + block * 32u;
        for (size_t i = 0; i < 16; i++) {
            dst[i] = (float)((int)(qs[i] & 0x0f) - 8) * d;
            dst[i + 16] = (float)((int)(qs[i] >> 4) - 8) * d;
        }
    }
}

static void dequant_q4_1(float * out, const uint8_t * row, size_t cols) {
    const size_t blocks = cols / 32u;
    for (size_t block = 0; block < blocks; block++) {
        const uint8_t * p = row + block * 20u;
        const float d = im_f16_to_f32(read_le16(p));
        const float m = im_f16_to_f32(read_le16(p + 2));
        const uint8_t * qs = p + 4;
        float * dst = out + block * 32u;
        for (size_t i = 0; i < 16; i++) {
            dst[i] = (float)(qs[i] & 0x0f) * d + m;
            dst[i + 16] = (float)(qs[i] >> 4) * d + m;
        }
    }
}

static void dequant_q5_0(float * out, const uint8_t * row, size_t cols) {
    const size_t blocks = cols / 32u;
    for (size_t block = 0; block < blocks; block++) {
        const uint8_t * p = row + block * 22u;
        const float d = im_f16_to_f32(read_le16(p));
        const uint32_t qh = read_le32(p + 2);
        const uint8_t * qs = p + 6;
        float * dst = out + block * 32u;
        for (size_t i = 0; i < 16; i++) {
            const uint8_t xh0 = (uint8_t)(((qh >> i) & 1u) << 4);
            const uint8_t xh1 = (uint8_t)(((qh >> (i + 16u)) & 1u) << 4);
            dst[i] = (float)(((qs[i] & 0x0f) | xh0) - 16) * d;
            dst[i + 16] = (float)(((qs[i] >> 4) | xh1) - 16) * d;
        }
    }
}

static void dequant_q5_1(float * out, const uint8_t * row, size_t cols) {
    const size_t blocks = cols / 32u;
    for (size_t block = 0; block < blocks; block++) {
        const uint8_t * p = row + block * 24u;
        const float d = im_f16_to_f32(read_le16(p));
        const float m = im_f16_to_f32(read_le16(p + 2));
        const uint32_t qh = read_le32(p + 4);
        const uint8_t * qs = p + 8;
        float * dst = out + block * 32u;
        for (size_t i = 0; i < 16; i++) {
            const uint8_t xh0 = (uint8_t)(((qh >> i) & 1u) << 4);
            const uint8_t xh1 = (uint8_t)(((qh >> (i + 16u)) & 1u) << 4);
            dst[i] = (float)((qs[i] & 0x0f) | xh0) * d + m;
            dst[i + 16] = (float)((qs[i] >> 4) | xh1) * d + m;
        }
    }
}

static void dequant_q8_0(float * out, const uint8_t * row, size_t cols) {
    const size_t blocks = cols / 32u;
    for (size_t block = 0; block < blocks; block++) {
        const uint8_t * p = row + block * 34u;
        const float d = im_f16_to_f32(read_le16(p));
        const int8_t * qs = (const int8_t *)(p + 2);
        float * dst = out + block * 32u;
        for (size_t i = 0; i < 32; i++) dst[i] = (float)qs[i] * d;
    }
}

static void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63u;
        *m = q[j + 4] & 63u;
    } else {
        *d = (q[j + 4] & 0x0fu) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static void dequant_q4_k(float * out, const uint8_t * row, size_t cols) {
    const size_t blocks = cols / 256u;
    float * y = out;
    for (size_t block = 0; block < blocks; block++) {
        const uint8_t * p = row + block * 144u;
        const float d = im_f16_to_f32(read_le16(p));
        const float min = im_f16_to_f32(read_le16(p + 2));
        const uint8_t * scales = p + 4;
        const uint8_t * q = p + 16;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc = 0;
            uint8_t m = 0;
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const float d1 = d * (float)sc;
            const float m1 = min * (float)m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            const float d2 = d * (float)sc;
            const float m2 = min * (float)m;
            for (int l = 0; l < 32; l++) *y++ = d1 * (float)(q[l] & 0x0f) - m1;
            for (int l = 0; l < 32; l++) *y++ = d2 * (float)(q[l] >> 4) - m2;
            q += 32;
            is += 2;
            (void)j;
        }
    }
}

static void dequant_q6_k(float * out, const uint8_t * row, size_t cols) {
    const size_t blocks = cols / 256u;
    for (size_t block = 0; block < blocks; block++) {
        const uint8_t * p = row + block * 210u;
        const uint8_t * ql = p;
        const uint8_t * qh = p + 128;
        const int8_t * sc = (const int8_t *)(p + 192);
        const float d = im_f16_to_f32(read_le16(p + 208));
        float * y = out + block * 256u;
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; l++) {
                const int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l + 0] & 0x0f) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0] = d * (float)sc[is + 0] * (float)q1;
                y[l + 32] = d * (float)sc[is + 2] * (float)q2;
                y[l + 64] = d * (float)sc[is + 4] * (float)q3;
                y[l + 96] = d * (float)sc[is + 6] * (float)q4;
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
            (void)n;
        }
    }
}

static float dot_q4_k(const uint8_t * row, const float * vector, size_t cols) {
    const size_t blocks = cols / 256u;
    float sum = 0.0f;
    for (size_t block = 0; block < blocks; block++) {
        const uint8_t * p = row + block * 144u;
        const float d = im_f16_to_f32(read_le16(p));
        const float min = im_f16_to_f32(read_le16(p + 2));
        const uint8_t * scales = p + 4;
        const uint8_t * q = p + 16;
        const size_t base = block * 256u;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc = 0;
            uint8_t m = 0;
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const float d1 = d * (float)sc;
            const float m1 = min * (float)m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            const float d2 = d * (float)sc;
            const float m2 = min * (float)m;
            const size_t off = base + (size_t)j;
            for (int l = 0; l < 32; l++) {
                sum += (d1 * (float)(q[l] & 0x0f) - m1) * vector[off + (size_t)l];
                sum += (d2 * (float)(q[l] >> 4) - m2) * vector[off + 32u + (size_t)l];
            }
            q += 32;
            is += 2;
        }
    }
    return sum;
}

static float dot_q6_k(const uint8_t * row, const float * vector, size_t cols) {
    const size_t blocks = cols / 256u;
    float sum = 0.0f;
    for (size_t block = 0; block < blocks; block++) {
        const uint8_t * p = row + block * 210u;
        const uint8_t * ql = p;
        const uint8_t * qh = p + 128;
        const int8_t * sc = (const int8_t *)(p + 192);
        const float d = im_f16_to_f32(read_le16(p + 208));
        const size_t base = block * 256u;
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; l++) {
                const int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l + 0] & 0x0f) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                const size_t off = base + (size_t)n + (size_t)l;
                sum += d * (float)sc[is + 0] * (float)q1 * vector[off + 0u];
                sum += d * (float)sc[is + 2] * (float)q2 * vector[off + 32u];
                sum += d * (float)sc[is + 4] * (float)q3 * vector[off + 64u];
                sum += d * (float)sc[is + 6] * (float)q4 * vector[off + 96u];
            }
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
    return sum;
}

int im_quant_has_direct_dot(int32_t type) {
    return type == IM_GGML_TYPE_Q4_K || type == IM_GGML_TYPE_Q6_K;
}

int im_dequantize_row(float * out, const void * row, int32_t type, size_t cols) {
    if (!out || !row) return -1;
    const uint8_t * bytes = (const uint8_t *)row;
    if (type == IM_GGML_TYPE_F32) {
        memcpy(out, row, cols * sizeof(float));
        return 0;
    }
    if (type == IM_GGML_TYPE_F16) {
        const uint8_t * src = bytes;
        for (size_t i = 0; i < cols; i++) out[i] = im_f16_to_f32(read_le16(src + i * 2u));
        return 0;
    }
    if (type == IM_GGML_TYPE_BF16) {
        const uint8_t * src = bytes;
        for (size_t i = 0; i < cols; i++) out[i] = im_bf16_to_f32(read_le16(src + i * 2u));
        return 0;
    }
    const im_quant_type_info * info = im_quant_type_info_for(type);
    if (!info || !info->matvec_supported || cols % info->block_size != 0) return -1;
    if (type == IM_GGML_TYPE_Q4_0) dequant_q4_0(out, bytes, cols);
    else if (type == IM_GGML_TYPE_Q4_1) dequant_q4_1(out, bytes, cols);
    else if (type == IM_GGML_TYPE_Q5_0) dequant_q5_0(out, bytes, cols);
    else if (type == IM_GGML_TYPE_Q5_1) dequant_q5_1(out, bytes, cols);
    else if (type == IM_GGML_TYPE_Q8_0) dequant_q8_0(out, bytes, cols);
    else if (type == IM_GGML_TYPE_Q4_K) dequant_q4_k(out, bytes, cols);
    else if (type == IM_GGML_TYPE_Q6_K) dequant_q6_k(out, bytes, cols);
    else return -1;
    return 0;
}

int im_quant_matvec(float * out, const void * matrix, int32_t type, size_t rows, size_t cols, const float * vector) {
    if (!out || !matrix || !vector || rows == 0 || cols == 0) return -1;
    const size_t row_bytes = im_quant_row_size(type, cols);
    if (row_bytes == 0) return -1;
    const uint8_t * base = (const uint8_t *)matrix;
    if (im_quant_has_direct_dot(type)) {
        const im_quant_type_info * info = im_quant_type_info_for(type);
        if (!info || cols % info->block_size != 0) return -1;
        for (size_t r = 0; r < rows; r++) {
            const uint8_t * row = base + r * row_bytes;
            out[r] = type == IM_GGML_TYPE_Q4_K ? dot_q4_k(row, vector, cols) : dot_q6_k(row, vector, cols);
        }
        return 0;
    }
    float * row = (float *)malloc(cols * sizeof(float));
    if (!row) return -1;
    for (size_t r = 0; r < rows; r++) {
        if (im_dequantize_row(row, base + r * row_bytes, type, cols) != 0) {
            free(row);
            return -1;
        }
        out[r] = im_dot_f32(row, vector, cols);
    }
    free(row);
    return 0;
}
