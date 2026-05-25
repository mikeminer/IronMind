#include "ironmind_qwen3.h"

#include "ironmind_math.h"
#include "ironmind_moe.h"
#include "ironmind_simd.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t q_dim(const im_forward_config * cfg) {
    return (size_t)cfg->n_head * cfg->head_dim;
}

static size_t kv_dim(const im_forward_config * cfg) {
    return (size_t)cfg->n_head_kv * cfg->head_dim;
}

static size_t kv_offset(const im_forward_config * cfg, uint32_t layer, uint32_t pos, uint32_t kv_head) {
    return ((((size_t)layer * cfg->max_seq + pos) * cfg->n_head_kv + kv_head) * cfg->head_dim);
}

static float dot_local(const float * a, const float * b, size_t n) {
    return im_dot_f32(a, b, n);
}

static float silu_local(float x) {
    return x / (1.0f + expf(-x));
}

static void add_local(float * out, const float * a, const float * b, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = a[i] + b[i];
}

static const im_gguf_tensor * require_tensor(const im_gguf_file * gguf, const char * name) {
    const im_gguf_tensor * tensor = im_gguf_find_tensor(gguf, name);
    if (!tensor) fprintf(stderr, "missing tensor %s\n", name);
    return tensor;
}

static const im_gguf_tensor * require_layer_tensor(const im_gguf_file * gguf, uint32_t layer, const char * suffix) {
    char name[128];
    snprintf(name, sizeof(name), "blk.%u.%s", layer, suffix);
    return require_tensor(gguf, name);
}

static uint64_t env_mb(const char * name, uint64_t fallback) {
    const char * value = getenv(name);
    if (!value || !*value) return fallback;
    char * end = NULL;
    const unsigned long long parsed = strtoull(value, &end, 10);
    return end && *end == '\0' ? (uint64_t)parsed : fallback;
}

static void pin_if_present(im_qwen3_model * model, const im_gguf_tensor * tensor) {
    if (tensor) (void)im_gguf_pin_tensor(&model->gguf, tensor);
}

static int tensor_rows_cols(const im_gguf_tensor * tensor, uint64_t rows, uint64_t cols) {
    return tensor && im_gguf_tensor_rows(tensor) == rows && im_gguf_tensor_cols(tensor) == cols;
}

static int tensor_matvec_ready(const im_gguf_tensor * tensor, uint64_t rows, uint64_t cols) {
    return tensor_rows_cols(tensor, rows, cols) && im_gguf_tensor_matvec_supported(tensor);
}

static int read_vector(const im_qwen3_model * model, const im_gguf_tensor * tensor, float * out, size_t n) {
    if (!tensor_rows_cols(tensor, 1, n)) return -1;
    return im_gguf_read_tensor_f32(&model->gguf, tensor, out, n);
}

static int matvec_tensor(const im_qwen3_model * model, const im_gguf_tensor * tensor, float * out, uint64_t rows, uint64_t cols, const float * vector) {
    if (!tensor_matvec_ready(tensor, rows, cols)) return -1;
    return im_gguf_tensor_matvec(&model->gguf, tensor, 0, rows, vector, out);
}

static int matvec_tensor_rows(const im_qwen3_model * model, const im_gguf_tensor * tensor, uint64_t row_start, uint64_t rows, uint64_t cols, const float * vector, float * out) {
    if (!tensor || im_gguf_tensor_cols(tensor) != cols || row_start + rows > im_gguf_tensor_rows(tensor) || !im_gguf_tensor_matvec_supported(tensor)) return -1;
    return im_gguf_tensor_matvec(&model->gguf, tensor, row_start, rows, vector, out);
}

static int validate_dense_layer(const im_qwen3_model * model, const im_qwen3_layer_tensors * l) {
    const im_forward_config * cfg = &model->cfg;
    return tensor_rows_cols(l->attn_norm, 1, cfg->n_embd) &&
           tensor_matvec_ready(l->attn_q, q_dim(cfg), cfg->n_embd) &&
           tensor_matvec_ready(l->attn_k, kv_dim(cfg), cfg->n_embd) &&
           tensor_matvec_ready(l->attn_v, kv_dim(cfg), cfg->n_embd) &&
           tensor_rows_cols(l->attn_q_norm, 1, cfg->head_dim) &&
           tensor_rows_cols(l->attn_k_norm, 1, cfg->head_dim) &&
           tensor_matvec_ready(l->attn_o, cfg->n_embd, q_dim(cfg)) &&
           tensor_rows_cols(l->ffn_norm, 1, cfg->n_embd) &&
           tensor_matvec_ready(l->ffn_gate, cfg->n_ff, cfg->n_embd) &&
           tensor_matvec_ready(l->ffn_up, cfg->n_ff, cfg->n_embd) &&
           tensor_matvec_ready(l->ffn_down, cfg->n_embd, cfg->n_ff);
}

static int validate_moe_layer(const im_qwen3_model * model, const im_qwen3_layer_tensors * l) {
    const im_forward_config * cfg = &model->cfg;
    const uint64_t expert_rows = (uint64_t)model->n_expert * model->n_ff_exp;
    const uint64_t down_rows = (uint64_t)model->n_expert * cfg->n_embd;
    return tensor_rows_cols(l->attn_norm, 1, cfg->n_embd) &&
           tensor_matvec_ready(l->attn_q, q_dim(cfg), cfg->n_embd) &&
           tensor_matvec_ready(l->attn_k, kv_dim(cfg), cfg->n_embd) &&
           tensor_matvec_ready(l->attn_v, kv_dim(cfg), cfg->n_embd) &&
           tensor_rows_cols(l->attn_q_norm, 1, cfg->head_dim) &&
           tensor_rows_cols(l->attn_k_norm, 1, cfg->head_dim) &&
           tensor_matvec_ready(l->attn_o, cfg->n_embd, q_dim(cfg)) &&
           tensor_rows_cols(l->ffn_norm, 1, cfg->n_embd) &&
           tensor_matvec_ready(l->moe_gate_inp, model->n_expert, cfg->n_embd) &&
           tensor_matvec_ready(l->moe_gate_exps, expert_rows, cfg->n_embd) &&
           tensor_matvec_ready(l->moe_up_exps, expert_rows, cfg->n_embd) &&
           tensor_matvec_ready(l->moe_down_exps, down_rows, model->n_ff_exp);
}

int im_qwen3_model_load(im_qwen3_model * model, const char * path, uint32_t max_seq) {
    if (!model || !path) return -1;
    memset(model, 0, sizeof(*model));
    if (im_gguf_load(path, &model->gguf) != 0) return -1;
    if (!im_gguf_is_qwen_target(&model->gguf)) {
        im_qwen3_model_free(model);
        return -1;
    }
    {
        const uint64_t cache_mb = env_mb("IRONMIND_NATIVE_CACHE_MB", 512);
        const uint64_t max_tensor_mb = env_mb("IRONMIND_NATIVE_CACHE_MAX_TENSOR_MB", 64);
        (void)im_gguf_set_residency(&model->gguf, cache_mb * 1024ull * 1024ull, max_tensor_mb * 1024ull * 1024ull);
    }

    model->is_moe = strcmp(model->gguf.architecture, "qwen3moe") == 0;
    model->token_embedding = require_tensor(&model->gguf, "token_embd.weight");
    model->output_norm = require_tensor(&model->gguf, "output_norm.weight");
    model->output = im_gguf_find_tensor(&model->gguf, "output.weight");
    if (!model->output) model->output = model->token_embedding;
    if (!model->token_embedding || !model->output_norm || !model->output) {
        im_qwen3_model_free(model);
        return -1;
    }

    const uint64_t token_cols = im_gguf_tensor_cols(model->token_embedding);
    const uint64_t token_rows = im_gguf_tensor_rows(model->token_embedding);
    const uint64_t head_dim = model->gguf.key_length ? model->gguf.key_length : model->gguf.embedding_length / model->gguf.head_count;
    const uint64_t ff = model->is_moe ? model->gguf.expert_feed_forward_length : model->gguf.feed_forward_length;
    if (!token_cols || !token_rows || !head_dim || !ff || token_cols != model->gguf.embedding_length) {
        im_qwen3_model_free(model);
        return -1;
    }

    model->cfg.n_vocab = (uint32_t)token_rows;
    model->cfg.n_layer = (uint32_t)model->gguf.block_count;
    model->cfg.n_embd = (uint32_t)model->gguf.embedding_length;
    model->cfg.n_head = (uint32_t)model->gguf.head_count;
    model->cfg.n_head_kv = (uint32_t)model->gguf.head_count_kv;
    model->cfg.head_dim = (uint32_t)head_dim;
    model->cfg.n_ff = (uint32_t)ff;
    model->cfg.max_seq = max_seq ? max_seq : (uint32_t)model->gguf.context_length;
    model->cfg.rms_eps = (float)model->gguf.rms_norm_eps;
    model->cfg.rope_freq_base = (float)model->gguf.rope_freq_base;
    model->n_expert = (uint32_t)model->gguf.expert_count;
    model->n_expert_used = (uint32_t)model->gguf.expert_used_count;
    model->n_ff_exp = (uint32_t)model->gguf.expert_feed_forward_length;

    if (model->cfg.n_head * model->cfg.head_dim != model->cfg.n_embd || model->cfg.n_head % model->cfg.n_head_kv != 0) {
        im_qwen3_model_free(model);
        return -1;
    }
    if (!tensor_rows_cols(model->output_norm, 1, model->cfg.n_embd) ||
        !tensor_matvec_ready(model->output, model->cfg.n_vocab, model->cfg.n_embd)) {
        im_qwen3_model_free(model);
        return -1;
    }
    pin_if_present(model, model->output_norm);

    model->layers = (im_qwen3_layer_tensors *)calloc(model->cfg.n_layer, sizeof(*model->layers));
    if (!model->layers) {
        im_qwen3_model_free(model);
        return -1;
    }

    for (uint32_t layer = 0; layer < model->cfg.n_layer; layer++) {
        im_qwen3_layer_tensors * l = &model->layers[layer];
        l->attn_norm = require_layer_tensor(&model->gguf, layer, "attn_norm.weight");
        l->attn_q = require_layer_tensor(&model->gguf, layer, "attn_q.weight");
        l->attn_k = require_layer_tensor(&model->gguf, layer, "attn_k.weight");
        l->attn_v = require_layer_tensor(&model->gguf, layer, "attn_v.weight");
        l->attn_q_norm = require_layer_tensor(&model->gguf, layer, "attn_q_norm.weight");
        l->attn_k_norm = require_layer_tensor(&model->gguf, layer, "attn_k_norm.weight");
        l->attn_o = require_layer_tensor(&model->gguf, layer, "attn_output.weight");
        l->ffn_norm = require_layer_tensor(&model->gguf, layer, "ffn_norm.weight");
        pin_if_present(model, l->attn_norm);
        pin_if_present(model, l->attn_q_norm);
        pin_if_present(model, l->attn_k_norm);
        pin_if_present(model, l->ffn_norm);
        if (model->is_moe) {
            l->moe_gate_inp = require_layer_tensor(&model->gguf, layer, "ffn_gate_inp.weight");
            l->moe_gate_exps = require_layer_tensor(&model->gguf, layer, "ffn_gate_exps.weight");
            l->moe_up_exps = require_layer_tensor(&model->gguf, layer, "ffn_up_exps.weight");
            l->moe_down_exps = require_layer_tensor(&model->gguf, layer, "ffn_down_exps.weight");
            if (!validate_moe_layer(model, l)) {
                im_qwen3_model_free(model);
                return -1;
            }
        } else {
            l->ffn_gate = require_layer_tensor(&model->gguf, layer, "ffn_gate.weight");
            l->ffn_up = require_layer_tensor(&model->gguf, layer, "ffn_up.weight");
            l->ffn_down = require_layer_tensor(&model->gguf, layer, "ffn_down.weight");
            if (!validate_dense_layer(model, l)) {
                im_qwen3_model_free(model);
                return -1;
            }
        }
    }
    return 0;
}

void im_qwen3_model_free(im_qwen3_model * model) {
    if (!model) return;
    free(model->layers);
    im_gguf_free(&model->gguf);
    memset(model, 0, sizeof(*model));
}

static int run_dense_ffn(const im_qwen3_model * model, const im_qwen3_layer_tensors * lw, const float * norm, float * ffn_out, float * gate, float * up, float * hidden) {
    const im_forward_config * cfg = &model->cfg;
    if (matvec_tensor(model, lw->ffn_gate, gate, cfg->n_ff, cfg->n_embd, norm) != 0) return -1;
    if (matvec_tensor(model, lw->ffn_up, up, cfg->n_ff, cfg->n_embd, norm) != 0) return -1;
    for (uint32_t i = 0; i < cfg->n_ff; i++) hidden[i] = silu_local(gate[i]) * up[i];
    return matvec_tensor(model, lw->ffn_down, ffn_out, cfg->n_embd, cfg->n_ff, hidden);
}

static int run_moe_ffn(const im_qwen3_model * model, const im_qwen3_layer_tensors * lw, const float * norm, float * ffn_out, float * gate, float * up, float * hidden, float * router_logits, im_moe_route * routes, float * expert_out) {
    const im_forward_config * cfg = &model->cfg;
    memset(ffn_out, 0, (size_t)cfg->n_embd * sizeof(float));
    if (matvec_tensor(model, lw->moe_gate_inp, router_logits, model->n_expert, cfg->n_embd, norm) != 0) return -1;
    if (im_moe_topk(routes, model->n_expert_used, router_logits, model->n_expert) != 0) return -1;
    for (uint32_t r = 0; r < model->n_expert_used; r++) {
        const uint32_t expert = routes[r].expert;
        const uint64_t expert_row_start = (uint64_t)expert * model->n_ff_exp;
        if (matvec_tensor_rows(model, lw->moe_gate_exps, expert_row_start, model->n_ff_exp, cfg->n_embd, norm, gate) != 0) return -1;
        if (matvec_tensor_rows(model, lw->moe_up_exps, expert_row_start, model->n_ff_exp, cfg->n_embd, norm, up) != 0) return -1;
        for (uint32_t i = 0; i < model->n_ff_exp; i++) hidden[i] = silu_local(gate[i]) * up[i];
        const uint64_t down_row_start = (uint64_t)expert * cfg->n_embd;
        if (matvec_tensor_rows(model, lw->moe_down_exps, down_row_start, cfg->n_embd, model->n_ff_exp, hidden, expert_out) != 0) return -1;
        for (uint32_t i = 0; i < cfg->n_embd; i++) ffn_out[i] += routes[r].weight * expert_out[i];
    }
    return 0;
}

int im_qwen3_decode(const im_qwen3_model * model, im_kv_cache * cache, uint32_t token_id, float * logits_out) {
    if (!model || !model->layers || !cache || !cache->key || !cache->value || !logits_out) return -1;
    const im_forward_config * cfg = &model->cfg;
    if (token_id >= cfg->n_vocab || cache->token_count >= cfg->max_seq) return -1;
    if (memcmp(&cache->cfg, cfg, sizeof(*cfg)) != 0) return -1;

    const size_t n_embd = cfg->n_embd;
    const size_t qd = q_dim(cfg);
    const size_t kvd = kv_dim(cfg);
    const uint32_t pos = cache->token_count;
    const float attn_scale = 1.0f / sqrtf((float)cfg->head_dim);
    const uint32_t ffn_work = model->is_moe ? model->n_ff_exp : cfg->n_ff;

    float * hidden = (float *)malloc(n_embd * sizeof(float));
    float * residual = (float *)malloc(n_embd * sizeof(float));
    float * norm = (float *)malloc(n_embd * sizeof(float));
    float * norm_w = (float *)malloc(n_embd * sizeof(float));
    float * q = (float *)malloc(qd * sizeof(float));
    float * k = (float *)malloc(kvd * sizeof(float));
    float * v = (float *)malloc(kvd * sizeof(float));
    float * qn = (float *)malloc((size_t)cfg->head_dim * sizeof(float));
    float * kn = (float *)malloc((size_t)cfg->head_dim * sizeof(float));
    float * attn_cat = (float *)malloc(qd * sizeof(float));
    float * attn_out = (float *)malloc(n_embd * sizeof(float));
    float * gate = (float *)malloc((size_t)ffn_work * sizeof(float));
    float * up = (float *)malloc((size_t)ffn_work * sizeof(float));
    float * ffn_hidden = (float *)malloc((size_t)ffn_work * sizeof(float));
    float * ffn_out = (float *)malloc(n_embd * sizeof(float));
    float * scores = (float *)malloc((size_t)(pos + 1) * sizeof(float));
    float * router_logits = model->is_moe ? (float *)malloc((size_t)model->n_expert * sizeof(float)) : NULL;
    im_moe_route * routes = model->is_moe ? (im_moe_route *)malloc((size_t)model->n_expert_used * sizeof(*routes)) : NULL;
    float * expert_out = model->is_moe ? (float *)malloc(n_embd * sizeof(float)) : NULL;

    if (!hidden || !residual || !norm || !norm_w || !q || !k || !v || !qn || !kn || !attn_cat || !attn_out ||
        !gate || !up || !ffn_hidden || !ffn_out || !scores || (model->is_moe && (!router_logits || !routes || !expert_out))) {
        free(hidden); free(residual); free(norm); free(norm_w); free(q); free(k); free(v); free(qn); free(kn); free(attn_cat); free(attn_out);
        free(gate); free(up); free(ffn_hidden); free(ffn_out); free(scores); free(router_logits); free(routes); free(expert_out);
        return -1;
    }

    int ok = im_gguf_read_tensor_row_f32(&model->gguf, model->token_embedding, token_id, hidden, n_embd) == 0;
    for (uint32_t layer = 0; ok && layer < cfg->n_layer; layer++) {
        const im_qwen3_layer_tensors * lw = &model->layers[layer];
        memcpy(residual, hidden, n_embd * sizeof(float));
        ok = read_vector(model, lw->attn_norm, norm_w, n_embd) == 0;
        if (!ok) break;
        im_rms_norm(norm, hidden, norm_w, n_embd, cfg->rms_eps);

        ok = matvec_tensor(model, lw->attn_q, q, qd, n_embd, norm) == 0 &&
             matvec_tensor(model, lw->attn_k, k, kvd, n_embd, norm) == 0 &&
             matvec_tensor(model, lw->attn_v, v, kvd, n_embd, norm) == 0 &&
             read_vector(model, lw->attn_q_norm, qn, cfg->head_dim) == 0 &&
             read_vector(model, lw->attn_k_norm, kn, cfg->head_dim) == 0;
        if (!ok) break;

        for (uint32_t head = 0; head < cfg->n_head; head++) {
            float * qh = q + (size_t)head * cfg->head_dim;
            im_rms_norm(qh, qh, qn, cfg->head_dim, cfg->rms_eps);
        }
        im_apply_rope(q, qd, cfg->head_dim, (double)pos, (double)cfg->rope_freq_base);
        for (uint32_t head = 0; head < cfg->n_head_kv; head++) {
            float * kh = k + (size_t)head * cfg->head_dim;
            im_rms_norm(kh, kh, kn, cfg->head_dim, cfg->rms_eps);
        }
        im_apply_rope(k, kvd, cfg->head_dim, (double)pos, (double)cfg->rope_freq_base);

        for (uint32_t kv_head = 0; kv_head < cfg->n_head_kv; kv_head++) {
            const size_t off = kv_offset(cfg, layer, pos, kv_head);
            memcpy(cache->key + off, k + (size_t)kv_head * cfg->head_dim, (size_t)cfg->head_dim * sizeof(float));
            memcpy(cache->value + off, v + (size_t)kv_head * cfg->head_dim, (size_t)cfg->head_dim * sizeof(float));
        }

        memset(attn_cat, 0, qd * sizeof(float));
        const uint32_t group = cfg->n_head / cfg->n_head_kv;
        for (uint32_t head = 0; head < cfg->n_head; head++) {
            const uint32_t kv_head = head / group;
            const float * qh = q + (size_t)head * cfg->head_dim;
            for (uint32_t t = 0; t <= pos; t++) {
                const size_t off = kv_offset(cfg, layer, t, kv_head);
                scores[t] = dot_local(qh, cache->key + off, cfg->head_dim) * attn_scale;
            }
            im_softmax(scores, (size_t)pos + 1);
            float * out_head = attn_cat + (size_t)head * cfg->head_dim;
            for (uint32_t t = 0; t <= pos; t++) {
                const size_t off = kv_offset(cfg, layer, t, kv_head);
                const float p = scores[t];
                for (uint32_t d = 0; d < cfg->head_dim; d++) out_head[d] += p * cache->value[off + d];
            }
        }
        ok = matvec_tensor(model, lw->attn_o, attn_out, n_embd, qd, attn_cat) == 0;
        if (!ok) break;
        add_local(hidden, residual, attn_out, n_embd);

        memcpy(residual, hidden, n_embd * sizeof(float));
        ok = read_vector(model, lw->ffn_norm, norm_w, n_embd) == 0;
        if (!ok) break;
        im_rms_norm(norm, hidden, norm_w, n_embd, cfg->rms_eps);
        ok = model->is_moe ?
            run_moe_ffn(model, lw, norm, ffn_out, gate, up, ffn_hidden, router_logits, routes, expert_out) == 0 :
            run_dense_ffn(model, lw, norm, ffn_out, gate, up, ffn_hidden) == 0;
        if (!ok) break;
        add_local(hidden, residual, ffn_out, n_embd);
    }

    if (ok) {
        ok = read_vector(model, model->output_norm, norm_w, n_embd) == 0;
        if (ok) {
            im_rms_norm(norm, hidden, norm_w, n_embd, cfg->rms_eps);
            ok = matvec_tensor(model, model->output, logits_out, cfg->n_vocab, n_embd, norm) == 0;
        }
    }
    if (ok) cache->token_count++;

    free(hidden); free(residual); free(norm); free(norm_w); free(q); free(k); free(v); free(qn); free(kn); free(attn_cat); free(attn_out);
    free(gate); free(up); free(ffn_hidden); free(ffn_out); free(scores); free(router_logits); free(routes); free(expert_out);
    return ok ? 0 : -1;
}

uint32_t im_qwen3_argmax(const float * logits, uint32_t n_vocab) {
    uint32_t best = 0;
    for (uint32_t i = 1; i < n_vocab; i++) {
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}
