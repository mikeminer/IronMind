#pragma once

#include <stddef.h>
#include <stdint.h>

typedef struct im_forward_config {
    uint32_t n_vocab;
    uint32_t n_layer;
    uint32_t n_embd;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    uint32_t n_ff;
    uint32_t max_seq;
    float rms_eps;
    float rope_freq_base;
} im_forward_config;

typedef struct im_layer_weights {
    const float * attn_norm;
    const float * attn_q;
    const float * attn_k;
    const float * attn_v;
    const float * attn_q_norm;
    const float * attn_k_norm;
    const float * attn_o;
    const float * ffn_norm;
    const float * ffn_gate;
    const float * ffn_up;
    const float * ffn_down;
} im_layer_weights;

typedef struct im_forward_model {
    im_forward_config cfg;
    const float * token_embedding;
    const float * output_norm;
    const float * output;
    const im_layer_weights * layers;
} im_forward_model;

typedef struct im_kv_cache {
    im_forward_config cfg;
    uint32_t token_count;
    float * key;
    float * value;
} im_kv_cache;

int im_kv_cache_init(im_kv_cache * cache, const im_forward_config * cfg);
void im_kv_cache_free(im_kv_cache * cache);
void im_kv_cache_reset(im_kv_cache * cache);
size_t im_kv_cache_floats(const im_forward_config * cfg);
size_t im_kv_cache_bytes(const im_forward_config * cfg);
int im_kv_cache_save(const im_kv_cache * cache, const char * path);
int im_kv_cache_load(im_kv_cache * cache, const im_forward_config * cfg, const char * path);

int im_forward_decode(const im_forward_model * model, im_kv_cache * cache, uint32_t token_id, float * logits_out);
