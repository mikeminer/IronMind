#include "ironmind_forward.h"

#include "ironmind_math.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct im_kv_file_header {
    char magic[8];
    uint32_t version;
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
    uint32_t token_count;
    uint64_t payload_floats;
} im_kv_file_header;

static size_t im_q_dim(const im_forward_config * cfg) {
    return (size_t) cfg->n_head * (size_t) cfg->head_dim;
}

static size_t im_kv_dim(const im_forward_config * cfg) {
    return (size_t) cfg->n_head_kv * (size_t) cfg->head_dim;
}

static size_t im_kv_token_floats(const im_forward_config * cfg) {
    return (size_t) cfg->n_layer * im_kv_dim(cfg);
}

size_t im_kv_cache_floats(const im_forward_config * cfg) {
    return (size_t) cfg->max_seq * im_kv_token_floats(cfg) * 2u;
}

size_t im_kv_cache_bytes(const im_forward_config * cfg) {
    return im_kv_cache_floats(cfg) * sizeof(float);
}

static size_t im_kv_offset(const im_forward_config * cfg, uint32_t layer, uint32_t pos, uint32_t kv_head) {
    return ((((size_t) layer * cfg->max_seq + pos) * cfg->n_head_kv + kv_head) * cfg->head_dim);
}

static int im_valid_config(const im_forward_config * cfg) {
    if (!cfg) return 0;
    if (!cfg->n_vocab || !cfg->n_layer || !cfg->n_embd || !cfg->n_head || !cfg->n_head_kv || !cfg->head_dim || !cfg->n_ff || !cfg->max_seq) return 0;
    if (cfg->n_head % cfg->n_head_kv != 0) return 0;
    if (cfg->n_head * cfg->head_dim != cfg->n_embd) return 0;
    if (cfg->rms_eps < 0.0f || cfg->rope_freq_base <= 0.0f) return 0;
    return 1;
}

int im_kv_cache_init(im_kv_cache * cache, const im_forward_config * cfg) {
    if (!cache || !im_valid_config(cfg)) return -1;
    memset(cache, 0, sizeof(*cache));
    cache->cfg = *cfg;
    const size_t single = (size_t) cfg->max_seq * im_kv_token_floats(cfg);
    cache->key = (float *) calloc(single, sizeof(float));
    cache->value = (float *) calloc(single, sizeof(float));
    if (!cache->key || !cache->value) {
        im_kv_cache_free(cache);
        return -1;
    }
    return 0;
}

void im_kv_cache_free(im_kv_cache * cache) {
    if (!cache) return;
    free(cache->key);
    free(cache->value);
    memset(cache, 0, sizeof(*cache));
}

void im_kv_cache_reset(im_kv_cache * cache) {
    if (!cache || !cache->key || !cache->value) return;
    const size_t single = (size_t) cache->cfg.max_seq * im_kv_token_floats(&cache->cfg);
    memset(cache->key, 0, single * sizeof(float));
    memset(cache->value, 0, single * sizeof(float));
    cache->token_count = 0;
}

static int im_same_config(const im_forward_config * a, const im_forward_config * b) {
    return a->n_vocab == b->n_vocab &&
           a->n_layer == b->n_layer &&
           a->n_embd == b->n_embd &&
           a->n_head == b->n_head &&
           a->n_head_kv == b->n_head_kv &&
           a->head_dim == b->head_dim &&
           a->n_ff == b->n_ff &&
           a->max_seq == b->max_seq &&
           a->rms_eps == b->rms_eps &&
           a->rope_freq_base == b->rope_freq_base;
}

static int im_kv_write_active(FILE * file, const float * data, const im_forward_config * cfg, uint32_t token_count) {
    const size_t kv_dim = im_kv_dim(cfg);
    const size_t floats = (size_t) token_count * kv_dim;
    for (uint32_t layer = 0; layer < cfg->n_layer; layer++) {
        const float * src = data + im_kv_offset(cfg, layer, 0, 0);
        if (floats && fwrite(src, sizeof(float), floats, file) != floats) return 0;
    }
    return 1;
}

static int im_kv_read_active(FILE * file, float * data, const im_forward_config * cfg, uint32_t token_count) {
    const size_t kv_dim = im_kv_dim(cfg);
    const size_t floats = (size_t) token_count * kv_dim;
    for (uint32_t layer = 0; layer < cfg->n_layer; layer++) {
        float * dst = data + im_kv_offset(cfg, layer, 0, 0);
        if (floats && fread(dst, sizeof(float), floats, file) != floats) return 0;
    }
    return 1;
}

int im_kv_cache_save(const im_kv_cache * cache, const char * path) {
    if (!cache || !path || !cache->key || !cache->value) return -1;
    if (!im_valid_config(&cache->cfg) || cache->token_count > cache->cfg.max_seq) return -1;
    FILE * file = fopen(path, "wb");
    if (!file) return -1;

    im_kv_file_header header;
    memset(&header, 0, sizeof(header));
    memcpy(header.magic, "IMKVFWD", 7);
    header.version = 1;
    header.n_vocab = cache->cfg.n_vocab;
    header.n_layer = cache->cfg.n_layer;
    header.n_embd = cache->cfg.n_embd;
    header.n_head = cache->cfg.n_head;
    header.n_head_kv = cache->cfg.n_head_kv;
    header.head_dim = cache->cfg.head_dim;
    header.n_ff = cache->cfg.n_ff;
    header.max_seq = cache->cfg.max_seq;
    header.rms_eps = cache->cfg.rms_eps;
    header.rope_freq_base = cache->cfg.rope_freq_base;
    header.token_count = cache->token_count;
    header.payload_floats = (uint64_t) cache->token_count * (uint64_t) im_kv_token_floats(&cache->cfg);

    int ok = fwrite(&header, sizeof(header), 1, file) == 1;
    ok = ok && im_kv_write_active(file, cache->key, &cache->cfg, cache->token_count);
    ok = ok && im_kv_write_active(file, cache->value, &cache->cfg, cache->token_count);
    fclose(file);
    return ok ? 0 : -1;
}

int im_kv_cache_load(im_kv_cache * cache, const im_forward_config * cfg, const char * path) {
    if (!cache || !cfg || !path) return -1;
    FILE * file = fopen(path, "rb");
    if (!file) return -1;

    im_kv_file_header header;
    int ok = fread(&header, sizeof(header), 1, file) == 1;
    if (!ok || memcmp(header.magic, "IMKVFWD", 7) != 0 || header.version != 1) {
        fclose(file);
        return -1;
    }

    im_forward_config saved;
    memset(&saved, 0, sizeof(saved));
    saved.n_vocab = header.n_vocab;
    saved.n_layer = header.n_layer;
    saved.n_embd = header.n_embd;
    saved.n_head = header.n_head;
    saved.n_head_kv = header.n_head_kv;
    saved.head_dim = header.head_dim;
    saved.n_ff = header.n_ff;
    saved.max_seq = header.max_seq;
    saved.rms_eps = header.rms_eps;
    saved.rope_freq_base = header.rope_freq_base;

    if (!im_same_config(&saved, cfg) || header.token_count > cfg->max_seq) {
        fclose(file);
        return -1;
    }

    if (cache->key || cache->value) im_kv_cache_free(cache);
    if (im_kv_cache_init(cache, cfg) != 0) {
        fclose(file);
        return -1;
    }

    const size_t floats = (size_t) header.payload_floats;
    const size_t expected = (size_t) header.token_count * im_kv_token_floats(cfg);
    ok = floats == expected;
    ok = ok && im_kv_read_active(file, cache->key, cfg, header.token_count);
    ok = ok && im_kv_read_active(file, cache->value, cfg, header.token_count);
    fclose(file);
    if (!ok) {
        im_kv_cache_free(cache);
        return -1;
    }
    cache->token_count = header.token_count;
    return 0;
}

static void im_mat_vec(float * out, const float * matrix, const float * vector, size_t rows, size_t cols) {
    for (size_t row = 0; row < rows; row++) {
        const float * w = matrix + row * cols;
        float sum = 0.0f;
        for (size_t col = 0; col < cols; col++) sum += w[col] * vector[col];
        out[row] = sum;
    }
}

static void im_add(float * out, const float * a, const float * b, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = a[i] + b[i];
}

static float im_dot_local(const float * a, const float * b, size_t n) {
    float out = 0.0f;
    for (size_t i = 0; i < n; i++) out += a[i] * b[i];
    return out;
}

static float im_silu(float x) {
    return x / (1.0f + expf(-x));
}

static int im_required_model_pointers(const im_forward_model * model) {
    if (!model || !im_valid_config(&model->cfg) || !model->token_embedding || !model->output_norm || !model->output || !model->layers) return 0;
    for (uint32_t i = 0; i < model->cfg.n_layer; i++) {
        const im_layer_weights * l = &model->layers[i];
        if (!l->attn_norm || !l->attn_q || !l->attn_k || !l->attn_v || !l->attn_q_norm || !l->attn_k_norm || !l->attn_o ||
            !l->ffn_norm || !l->ffn_gate || !l->ffn_up || !l->ffn_down) return 0;
    }
    return 1;
}

int im_forward_decode(const im_forward_model * model, im_kv_cache * cache, uint32_t token_id, float * logits_out) {
    if (!im_required_model_pointers(model) || !cache || !cache->key || !cache->value || !logits_out) return -1;
    const im_forward_config * cfg = &model->cfg;
    if (!im_same_config(cfg, &cache->cfg) || token_id >= cfg->n_vocab || cache->token_count >= cfg->max_seq) return -1;

    const size_t n_embd = cfg->n_embd;
    const size_t q_dim = im_q_dim(cfg);
    const size_t kv_dim = im_kv_dim(cfg);
    const uint32_t pos = cache->token_count;
    const float attn_scale = 1.0f / sqrtf((float) cfg->head_dim);

    float * hidden = (float *) malloc(n_embd * sizeof(float));
    float * residual = (float *) malloc(n_embd * sizeof(float));
    float * norm = (float *) malloc(n_embd * sizeof(float));
    float * q = (float *) malloc(q_dim * sizeof(float));
    float * k = (float *) malloc(kv_dim * sizeof(float));
    float * v = (float *) malloc(kv_dim * sizeof(float));
    float * attn_cat = (float *) malloc(q_dim * sizeof(float));
    float * attn_out = (float *) malloc(n_embd * sizeof(float));
    float * gate = (float *) malloc((size_t) cfg->n_ff * sizeof(float));
    float * up = (float *) malloc((size_t) cfg->n_ff * sizeof(float));
    float * ffn_hidden = (float *) malloc((size_t) cfg->n_ff * sizeof(float));
    float * ffn_out = (float *) malloc(n_embd * sizeof(float));
    float * scores = (float *) malloc((size_t) (pos + 1) * sizeof(float));
    if (!hidden || !residual || !norm || !q || !k || !v || !attn_cat || !attn_out || !gate || !up || !ffn_hidden || !ffn_out || !scores) {
        free(hidden); free(residual); free(norm); free(q); free(k); free(v); free(attn_cat); free(attn_out);
        free(gate); free(up); free(ffn_hidden); free(ffn_out); free(scores);
        return -1;
    }

    memcpy(hidden, model->token_embedding + (size_t) token_id * n_embd, n_embd * sizeof(float));

    for (uint32_t layer = 0; layer < cfg->n_layer; layer++) {
        const im_layer_weights * lw = &model->layers[layer];
        memcpy(residual, hidden, n_embd * sizeof(float));
        im_rms_norm(norm, hidden, lw->attn_norm, n_embd, cfg->rms_eps);

        im_mat_vec(q, lw->attn_q, norm, q_dim, n_embd);
        im_mat_vec(k, lw->attn_k, norm, kv_dim, n_embd);
        im_mat_vec(v, lw->attn_v, norm, kv_dim, n_embd);

        for (uint32_t head = 0; head < cfg->n_head; head++) {
            float * qh = q + (size_t) head * cfg->head_dim;
            im_rms_norm(qh, qh, lw->attn_q_norm, cfg->head_dim, cfg->rms_eps);
        }
        im_apply_rope(q, q_dim, cfg->head_dim, (double) pos, (double) cfg->rope_freq_base);

        for (uint32_t head = 0; head < cfg->n_head_kv; head++) {
            float * kh = k + (size_t) head * cfg->head_dim;
            im_rms_norm(kh, kh, lw->attn_k_norm, cfg->head_dim, cfg->rms_eps);
        }
        im_apply_rope(k, kv_dim, cfg->head_dim, (double) pos, (double) cfg->rope_freq_base);

        for (uint32_t kv_head = 0; kv_head < cfg->n_head_kv; kv_head++) {
            const size_t off = im_kv_offset(cfg, layer, pos, kv_head);
            memcpy(cache->key + off, k + (size_t) kv_head * cfg->head_dim, (size_t) cfg->head_dim * sizeof(float));
            memcpy(cache->value + off, v + (size_t) kv_head * cfg->head_dim, (size_t) cfg->head_dim * sizeof(float));
        }

        memset(attn_cat, 0, q_dim * sizeof(float));
        const uint32_t group = cfg->n_head / cfg->n_head_kv;
        for (uint32_t head = 0; head < cfg->n_head; head++) {
            const uint32_t kv_head = head / group;
            const float * qh = q + (size_t) head * cfg->head_dim;
            for (uint32_t t = 0; t <= pos; t++) {
                const size_t off = im_kv_offset(cfg, layer, t, kv_head);
                scores[t] = im_dot_local(qh, cache->key + off, cfg->head_dim) * attn_scale;
            }
            im_softmax(scores, (size_t) pos + 1);
            float * out_head = attn_cat + (size_t) head * cfg->head_dim;
            for (uint32_t t = 0; t <= pos; t++) {
                const size_t off = im_kv_offset(cfg, layer, t, kv_head);
                const float p = scores[t];
                for (uint32_t d = 0; d < cfg->head_dim; d++) {
                    out_head[d] += p * cache->value[off + d];
                }
            }
        }

        im_mat_vec(attn_out, lw->attn_o, attn_cat, n_embd, q_dim);
        im_add(hidden, residual, attn_out, n_embd);

        memcpy(residual, hidden, n_embd * sizeof(float));
        im_rms_norm(norm, hidden, lw->ffn_norm, n_embd, cfg->rms_eps);
        im_mat_vec(gate, lw->ffn_gate, norm, cfg->n_ff, n_embd);
        im_mat_vec(up, lw->ffn_up, norm, cfg->n_ff, n_embd);
        for (uint32_t i = 0; i < cfg->n_ff; i++) ffn_hidden[i] = im_silu(gate[i]) * up[i];
        im_mat_vec(ffn_out, lw->ffn_down, ffn_hidden, n_embd, cfg->n_ff);
        im_add(hidden, residual, ffn_out, n_embd);
    }

    im_rms_norm(norm, hidden, model->output_norm, n_embd, cfg->rms_eps);
    im_mat_vec(logits_out, model->output, norm, cfg->n_vocab, n_embd);
    cache->token_count++;

    free(hidden); free(residual); free(norm); free(q); free(k); free(v); free(attn_cat); free(attn_out);
    free(gate); free(up); free(ffn_hidden); free(ffn_out); free(scores);
    return 0;
}
