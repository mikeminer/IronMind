#include "ironmind_quant.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void expect(int condition, const char * message) {
    if (!condition) {
        fprintf(stderr, "native quant test failed: %s\n", message);
        exit(1);
    }
}

static int closef_local(float a, float b, float eps) {
    return fabsf(a - b) <= eps;
}

int main(void) {
    expect(closef_local(im_f16_to_f32(0x3c00), 1.0f, 1e-6f), "f16 one");
    expect(closef_local(im_f16_to_f32(0x3800), 0.5f, 1e-6f), "f16 half");
    expect(closef_local(im_bf16_to_f32(0x3f80), 1.0f, 1e-6f), "bf16 one");

    uint8_t q8[34];
    memset(q8, 0, sizeof(q8));
    q8[0] = 0x00;
    q8[1] = 0x38;
    for (int i = 0; i < 32; i++) q8[2 + i] = (uint8_t)(int8_t)(i - 16);
    float row[32];
    expect(im_dequantize_row(row, q8, IM_GGML_TYPE_Q8_0, 32) == 0, "q8 dequant");
    expect(closef_local(row[0], -8.0f, 1e-6f), "q8 first");
    expect(closef_local(row[31], 7.5f, 1e-6f), "q8 last");

    uint8_t q4[18];
    memset(q4, 0, sizeof(q4));
    q4[0] = 0x00;
    q4[1] = 0x3c;
    for (int i = 0; i < 16; i++) q4[2 + i] = (uint8_t)(i | ((15 - i) << 4));
    expect(im_dequantize_row(row, q4, IM_GGML_TYPE_Q4_0, 32) == 0, "q4_0 dequant");
    expect(closef_local(row[0], -8.0f, 1e-6f), "q4_0 low");
    expect(closef_local(row[15], 7.0f, 1e-6f), "q4_0 low high");
    expect(closef_local(row[16], 7.0f, 1e-6f), "q4_0 high");
    expect(closef_local(row[31], -8.0f, 1e-6f), "q4_0 high low");

    float vec[32];
    for (int i = 0; i < 32; i++) vec[i] = 1.0f;
    float out[1];
    expect(im_quant_matvec(out, q8, IM_GGML_TYPE_Q8_0, 1, 32, vec) == 0, "q8 matvec");
    expect(closef_local(out[0], -8.0f, 1e-6f), "q8 matvec sum");

    expect(im_quant_row_size(IM_GGML_TYPE_Q4_K, 256) == 144, "q4_k row bytes");
    expect(im_quant_row_size(IM_GGML_TYPE_Q6_K, 256) == 210, "q6_k row bytes");
    expect(im_quant_type_info_for(IM_GGML_TYPE_Q4_K)->matvec_supported, "q4_k supported");

    uint8_t q4k[144];
    memset(q4k, 0, sizeof(q4k));
    q4k[0] = 0x00;
    q4k[1] = 0x3c;
    q4k[4] = 1;
    q4k[5] = 2;
    memset(q4k + 16, 0x11, 128);
    float krow[256];
    expect(im_dequantize_row(krow, q4k, IM_GGML_TYPE_Q4_K, 256) == 0, "q4_k dequant");
    expect(closef_local(krow[0], 1.0f, 1e-6f), "q4_k first scale");
    expect(closef_local(krow[32], 2.0f, 1e-6f), "q4_k second scale");

    uint8_t q6k[210];
    memset(q6k, 0, sizeof(q6k));
    memset(q6k + 128, 0xaa, 64);
    memset(q6k + 192, 1, 16);
    q6k[208] = 0x00;
    q6k[209] = 0x3c;
    expect(im_dequantize_row(krow, q6k, IM_GGML_TYPE_Q6_K, 256) == 0, "q6_k dequant");
    expect(closef_local(krow[0], 0.0f, 1e-6f), "q6_k zero point");
    expect(closef_local(krow[255], 0.0f, 1e-6f), "q6_k zero point tail");

    puts("native quant tests passed");
    return 0;
}
