#include "ironmind_forward.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int closef_local(float a, float b, float eps) {
    return fabsf(a - b) < eps;
}

static void expect(int condition, const char * message) {
    if (!condition) {
        fprintf(stderr, "native forward test failed: %s\n", message);
        exit(1);
    }
}

static void identity(float * m, size_t n) {
    memset(m, 0, n * n * sizeof(float));
    for (size_t i = 0; i < n; i++) m[i * n + i] = 1.0f;
}

static void select_rows(float * m, size_t rows, size_t cols, size_t start_col) {
    memset(m, 0, rows * cols * sizeof(float));
    for (size_t r = 0; r < rows; r++) {
        m[r * cols + ((start_col + r) % cols)] = 1.0f;
    }
}

static void fill_ones(float * v, size_t n) {
    for (size_t i = 0; i < n; i++) v[i] = 1.0f;
}

int main(void) {
    const im_forward_config cfg = {
        .n_vocab = 4,
        .n_layer = 2,
        .n_embd = 4,
        .n_head = 2,
        .n_head_kv = 1,
        .head_dim = 2,
        .n_ff = 4,
        .max_seq = 8,
        .rms_eps = 1e-6f,
        .rope_freq_base = 10000.0f,
    };

    float token_embedding[16] = {
        1.0f, 0.2f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.2f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.2f,
        0.2f, 0.0f, 0.0f, 1.0f,
    };
    float output_norm[4];
    float output[16];
    float attn_norm[4];
    float attn_q[16];
    float attn_k[8];
    float attn_v[8];
    float attn_q_norm[2];
    float attn_k_norm[2];
    float attn_o[16];
    float ffn_norm[4];
    float ffn_gate[16] = {0};
    float ffn_up[16] = {0};
    float ffn_down[16] = {0};

    fill_ones(output_norm, 4);
    identity(output, 4);
    fill_ones(attn_norm, 4);
    identity(attn_q, 4);
    select_rows(attn_k, 2, 4, 0);
    select_rows(attn_v, 2, 4, 2);
    fill_ones(attn_q_norm, 2);
    fill_ones(attn_k_norm, 2);
    identity(attn_o, 4);
    fill_ones(ffn_norm, 4);

    const im_layer_weights layer = {
        .attn_norm = attn_norm,
        .attn_q = attn_q,
        .attn_k = attn_k,
        .attn_v = attn_v,
        .attn_q_norm = attn_q_norm,
        .attn_k_norm = attn_k_norm,
        .attn_o = attn_o,
        .ffn_norm = ffn_norm,
        .ffn_gate = ffn_gate,
        .ffn_up = ffn_up,
        .ffn_down = ffn_down,
    };
    const im_layer_weights layers[2] = {layer, layer};
    const im_forward_model model = {
        .cfg = cfg,
        .token_embedding = token_embedding,
        .output_norm = output_norm,
        .output = output,
        .layers = layers,
    };

    im_kv_cache cache;
    expect(im_kv_cache_init(&cache, &cfg) == 0, "cache init");
    float logits0[4];
    float logits1[4];
    expect(im_forward_decode(&model, &cache, 0, logits0) == 0, "first decode");
    expect(cache.token_count == 1, "first token is cached");
    expect(im_forward_decode(&model, &cache, 1, logits1) == 0, "second decode");
    expect(cache.token_count == 2, "second token is cached");
    expect(!closef_local(logits0[0], logits1[0], 1e-4f) || !closef_local(logits0[1], logits1[1], 1e-4f), "second logits depend on KV history");

    const char * path = "ironmind-forward-test.imkv";
    expect(im_kv_cache_save(&cache, path) == 0, "cache save");
    {
        FILE * saved = fopen(path, "rb");
        char magic[8] = {0};
        expect(saved != NULL, "open saved IronKV");
        expect(fread(magic, 1, sizeof(magic), saved) == sizeof(magic), "read saved IronKV magic");
        fclose(saved);
        expect(memcmp(magic, "IRONKV1", 7) == 0, "saved cache uses IronKV container");
    }

    im_kv_cache restored;
    memset(&restored, 0, sizeof(restored));
    expect(im_kv_cache_load(&restored, &cfg, path) == 0, "cache load");
    expect(restored.token_count == cache.token_count, "restored token count");

    float logits_a[4];
    float logits_b[4];
    expect(im_forward_decode(&model, &cache, 2, logits_a) == 0, "decode from original cache");
    expect(im_forward_decode(&model, &restored, 2, logits_b) == 0, "decode from restored cache");
    for (size_t i = 0; i < 4; i++) {
        expect(closef_local(logits_a[i], logits_b[i], 1e-6f), "restored logits match");
    }

    remove(path);
    im_kv_cache_free(&cache);
    im_kv_cache_free(&restored);
    puts("native forward tests passed");
    return 0;
}
