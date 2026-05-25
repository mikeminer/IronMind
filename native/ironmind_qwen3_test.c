#include "ironmind_qwen3.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct tensor_spec {
    const char * name;
    uint32_t n_dims;
    uint64_t dims[4];
    const float * data;
    uint64_t values;
    uint64_t offset;
} tensor_spec;

static void expect(int condition, const char * message) {
    if (!condition) {
        fprintf(stderr, "qwen3 reference test failed: %s\n", message);
        exit(1);
    }
}

static int closef_local(float a, float b, float eps) {
    return fabsf(a - b) <= eps;
}

static uint64_t align_u64_local(uint64_t value, uint64_t alignment) {
    const uint64_t mask = alignment - 1u;
    return (value + mask) & ~mask;
}

static void write_u32(FILE * f, uint32_t v) {
    uint8_t b[4] = {(uint8_t)v, (uint8_t)(v >> 8), (uint8_t)(v >> 16), (uint8_t)(v >> 24)};
    fwrite(b, 1, sizeof(b), f);
}

static void write_i32(FILE * f, int32_t v) {
    write_u32(f, (uint32_t)v);
}

static void write_u64(FILE * f, uint64_t v) {
    uint8_t b[8] = {
        (uint8_t)v, (uint8_t)(v >> 8), (uint8_t)(v >> 16), (uint8_t)(v >> 24),
        (uint8_t)(v >> 32), (uint8_t)(v >> 40), (uint8_t)(v >> 48), (uint8_t)(v >> 56)
    };
    fwrite(b, 1, sizeof(b), f);
}

static void write_f32(FILE * f, float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    write_u32(f, bits);
}

static void write_string(FILE * f, const char * s) {
    write_u64(f, (uint64_t)strlen(s));
    fwrite(s, 1, strlen(s), f);
}

static void write_string_kv(FILE * f, const char * key, const char * value) {
    write_string(f, key);
    write_i32(f, 8);
    write_string(f, value);
}

static void write_u32_kv(FILE * f, const char * key, uint32_t value) {
    write_string(f, key);
    write_i32(f, 4);
    write_u32(f, value);
}

static void write_f32_kv(FILE * f, const char * key, float value) {
    write_string(f, key);
    write_i32(f, 6);
    write_f32(f, value);
}

static void write_tokens_kv(FILE * f) {
    write_string(f, "tokenizer.ggml.tokens");
    write_i32(f, 9);
    write_i32(f, 8);
    write_u64(f, 4);
    write_string(f, "a");
    write_string(f, "b");
    write_string(f, "c");
    write_string(f, "d");
}

static void pad_to(FILE * f, uint64_t absolute) {
    while ((uint64_t)ftell(f) < absolute) fputc(0, f);
}

static void add_tensor(tensor_spec * specs, size_t * count, const char * name, uint32_t n_dims, uint64_t d0, uint64_t d1, const float * data) {
    tensor_spec * s = &specs[(*count)++];
    memset(s, 0, sizeof(*s));
    s->name = name;
    s->n_dims = n_dims;
    s->dims[0] = d0;
    s->dims[1] = d1;
    s->data = data;
    s->values = d0 * (n_dims > 1 ? d1 : 1u);
}

static void write_fixture(const char * path, const tensor_spec * input_specs, size_t tensor_count) {
    tensor_spec specs[32];
    memcpy(specs, input_specs, tensor_count * sizeof(specs[0]));
    uint64_t next_offset = 0;
    for (size_t i = 0; i < tensor_count; i++) {
        next_offset = align_u64_local(next_offset, 32);
        specs[i].offset = next_offset;
        next_offset += specs[i].values * sizeof(float);
    }

    FILE * f = fopen(path, "wb");
    expect(f != NULL, "create fixture");
    fwrite("GGUF", 1, 4, f);
    write_u32(f, 3);
    write_u64(f, (uint64_t)tensor_count);
    write_u64(f, 13);
    write_string_kv(f, "general.architecture", "qwen3");
    write_string_kv(f, "general.name", "IronMind Qwen3 tiny");
    write_string_kv(f, "tokenizer.ggml.model", "gpt2");
    write_u32_kv(f, "qwen3.context_length", 8);
    write_u32_kv(f, "qwen3.embedding_length", 4);
    write_u32_kv(f, "qwen3.block_count", 2);
    write_u32_kv(f, "qwen3.attention.head_count", 2);
    write_u32_kv(f, "qwen3.attention.head_count_kv", 1);
    write_u32_kv(f, "qwen3.attention.key_length", 2);
    write_u32_kv(f, "qwen3.feed_forward_length", 4);
    write_f32_kv(f, "qwen3.rope.freq_base", 10000.0f);
    write_f32_kv(f, "qwen3.attention.layer_norm_rms_epsilon", 1e-6f);
    write_tokens_kv(f);

    for (size_t i = 0; i < tensor_count; i++) {
        write_string(f, specs[i].name);
        write_u32(f, specs[i].n_dims);
        for (uint32_t d = 0; d < specs[i].n_dims; d++) write_u64(f, specs[i].dims[d]);
        write_i32(f, 0);
        write_u64(f, specs[i].offset);
    }

    while ((ftell(f) % 32) != 0) fputc(0, f);
    const uint64_t data_start = (uint64_t)ftell(f);
    for (size_t i = 0; i < tensor_count; i++) {
        pad_to(f, data_start + specs[i].offset);
        for (uint64_t v = 0; v < specs[i].values; v++) write_f32(f, specs[i].data[v]);
    }
    fclose(f);
}

static void identity(float * m, size_t n) {
    memset(m, 0, n * n * sizeof(float));
    for (size_t i = 0; i < n; i++) m[i * n + i] = 1.0f;
}

static void select_rows(float * m, size_t rows, size_t cols, size_t start_col) {
    memset(m, 0, rows * cols * sizeof(float));
    for (size_t r = 0; r < rows; r++) m[r * cols + ((start_col + r) % cols)] = 1.0f;
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

    tensor_spec specs[32];
    char layer_names[22][64];
    size_t count = 0;
    add_tensor(specs, &count, "token_embd.weight", 2, 4, 4, token_embedding);
    add_tensor(specs, &count, "output_norm.weight", 1, 4, 1, output_norm);
    add_tensor(specs, &count, "output.weight", 2, 4, 4, output);
    for (int layer = 0; layer < 2; layer++) {
        const char * suffixes[11] = {
            "attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_q_norm.weight", "attn_k_norm.weight",
            "attn_output.weight", "ffn_norm.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"
        };
        for (int i = 0; i < 11; i++) snprintf(layer_names[layer * 11 + i], sizeof(layer_names[0]), "blk.%d.%s", layer, suffixes[i]);
        add_tensor(specs, &count, layer_names[layer * 11 + 0], 1, 4, 1, attn_norm);
        add_tensor(specs, &count, layer_names[layer * 11 + 1], 2, 4, 4, attn_q);
        add_tensor(specs, &count, layer_names[layer * 11 + 2], 2, 4, 2, attn_k);
        add_tensor(specs, &count, layer_names[layer * 11 + 3], 2, 4, 2, attn_v);
        add_tensor(specs, &count, layer_names[layer * 11 + 4], 1, 2, 1, attn_q_norm);
        add_tensor(specs, &count, layer_names[layer * 11 + 5], 1, 2, 1, attn_k_norm);
        add_tensor(specs, &count, layer_names[layer * 11 + 6], 2, 4, 4, attn_o);
        add_tensor(specs, &count, layer_names[layer * 11 + 7], 1, 4, 1, ffn_norm);
        add_tensor(specs, &count, layer_names[layer * 11 + 8], 2, 4, 4, ffn_gate);
        add_tensor(specs, &count, layer_names[layer * 11 + 9], 2, 4, 4, ffn_up);
        add_tensor(specs, &count, layer_names[layer * 11 + 10], 2, 4, 4, ffn_down);
    }

    const char * path = "ironmind-qwen3-reference.gguf";
    write_fixture(path, specs, count);

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
    const im_forward_model ref_model = {
        .cfg = cfg,
        .token_embedding = token_embedding,
        .output_norm = output_norm,
        .output = output,
        .layers = layers,
    };

    im_qwen3_model gguf_model;
    expect(im_qwen3_model_load(&gguf_model, path, 8) == 0, "load qwen3 fixture");
    im_kv_cache ref_cache;
    im_kv_cache gguf_cache;
    expect(im_kv_cache_init(&ref_cache, &cfg) == 0, "ref cache");
    expect(im_kv_cache_init(&gguf_cache, &gguf_model.cfg) == 0, "gguf cache");

    float ref_logits[4];
    float gguf_logits[4];
    expect(im_forward_decode(&ref_model, &ref_cache, 0, ref_logits) == 0, "ref decode 0");
    expect(im_qwen3_decode(&gguf_model, &gguf_cache, 0, gguf_logits) == 0, "gguf decode 0");
    for (size_t i = 0; i < 4; i++) expect(closef_local(ref_logits[i], gguf_logits[i], 1e-6f), "first logits match");
    expect(im_qwen3_argmax(ref_logits, 4) == im_qwen3_argmax(gguf_logits, 4), "first token match");

    expect(im_forward_decode(&ref_model, &ref_cache, 1, ref_logits) == 0, "ref decode 1");
    expect(im_qwen3_decode(&gguf_model, &gguf_cache, 1, gguf_logits) == 0, "gguf decode 1");
    for (size_t i = 0; i < 4; i++) expect(closef_local(ref_logits[i], gguf_logits[i], 1e-6f), "second logits match");
    expect(im_qwen3_argmax(ref_logits, 4) == im_qwen3_argmax(gguf_logits, 4), "second token match");

    im_kv_cache_free(&ref_cache);
    im_kv_cache_free(&gguf_cache);
    im_qwen3_model_free(&gguf_model);
    remove(path);
    puts("native qwen3 reference tests passed");
    return 0;
}
