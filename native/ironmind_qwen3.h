#pragma once

#include "ironmind_forward.h"
#include "ironmind_gguf.h"

#include <stdint.h>

typedef struct im_qwen3_layer_tensors {
    const im_gguf_tensor * attn_norm;
    const im_gguf_tensor * attn_q;
    const im_gguf_tensor * attn_k;
    const im_gguf_tensor * attn_v;
    const im_gguf_tensor * attn_q_norm;
    const im_gguf_tensor * attn_k_norm;
    const im_gguf_tensor * attn_o;
    const im_gguf_tensor * ffn_norm;
    const im_gguf_tensor * ffn_gate;
    const im_gguf_tensor * ffn_up;
    const im_gguf_tensor * ffn_down;
    const im_gguf_tensor * moe_gate_inp;
    const im_gguf_tensor * moe_gate_exps;
    const im_gguf_tensor * moe_up_exps;
    const im_gguf_tensor * moe_down_exps;
} im_qwen3_layer_tensors;

typedef struct im_qwen3_model {
    im_gguf_file gguf;
    im_forward_config cfg;
    int is_moe;
    uint32_t n_expert;
    uint32_t n_expert_used;
    uint32_t n_ff_exp;
    const im_gguf_tensor * token_embedding;
    const im_gguf_tensor * output_norm;
    const im_gguf_tensor * output;
    im_qwen3_layer_tensors * layers;
} im_qwen3_model;

int im_qwen3_model_load(im_qwen3_model * model, const char * path, uint32_t max_seq);
void im_qwen3_model_free(im_qwen3_model * model);
int im_qwen3_decode(const im_qwen3_model * model, im_kv_cache * cache, uint32_t token_id, float * logits_out);
uint32_t im_qwen3_argmax(const float * logits, uint32_t n_vocab);
