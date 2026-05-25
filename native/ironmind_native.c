#include "ironmind_gguf.h"

#include "ironmind_forward.h"
#include "ironmind_quant.h"
#include "ironmind_qwen3.h"
#include "ironmind_simd.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int is_moe(const im_gguf_file * file) {
    return strcmp(file->architecture, "qwen3moe") == 0;
}

static int check_tensor(const im_gguf_file * file, const char * name, uint64_t * mapped, uint64_t * unsupported) {
    const im_gguf_tensor * tensor = im_gguf_find_tensor(file, name);
    if (!tensor) {
        printf("  missing             %s\n", name);
        return 0;
    }
    (*mapped)++;
    if (!im_gguf_tensor_matvec_supported(tensor)) {
        (*unsupported)++;
        printf("  unsupported         %-42s %s\n", name, im_quant_type_name(tensor->type));
    }
    return 1;
}

static void check_layer_tensor(const im_gguf_file * file, uint64_t layer, const char * suffix, uint64_t * mapped, uint64_t * missing, uint64_t * unsupported) {
    char name[128];
    snprintf(name, sizeof(name), "blk.%" PRIu64 ".%s", layer, suffix);
    if (!check_tensor(file, name, mapped, unsupported)) (*missing)++;
}

static int native_probe(const char * path) {
    im_gguf_file file;
    if (im_gguf_load(path, &file) != 0) return 1;

    im_gguf_print_summary(&file);
    if (!im_gguf_is_qwen_target(&file)) {
        printf("  issue               unsupported architecture\n");
        im_gguf_free(&file);
        return 1;
    }

    uint64_t mapped = 0;
    uint64_t missing = 0;
    uint64_t unsupported = 0;
    if (!check_tensor(&file, "token_embd.weight", &mapped, &unsupported)) missing++;
    if (!check_tensor(&file, "output_norm.weight", &mapped, &unsupported)) missing++;
    if (!check_tensor(&file, "output.weight", &mapped, &unsupported)) {
        if (!im_gguf_find_tensor(&file, "token_embd.weight")) missing++;
    }

    const char * common[] = {
        "attn_norm.weight",
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "attn_q_norm.weight",
        "attn_k_norm.weight",
        "ffn_norm.weight",
    };
    for (uint64_t layer = 0; layer < file.block_count; layer++) {
        for (size_t i = 0; i < sizeof(common) / sizeof(common[0]); i++) {
            check_layer_tensor(&file, layer, common[i], &mapped, &missing, &unsupported);
        }
        if (is_moe(&file)) {
            check_layer_tensor(&file, layer, "ffn_gate_inp.weight", &mapped, &missing, &unsupported);
            check_layer_tensor(&file, layer, "ffn_gate_exps.weight", &mapped, &missing, &unsupported);
            check_layer_tensor(&file, layer, "ffn_up_exps.weight", &mapped, &missing, &unsupported);
            check_layer_tensor(&file, layer, "ffn_down_exps.weight", &mapped, &missing, &unsupported);
        } else {
            check_layer_tensor(&file, layer, "ffn_gate.weight", &mapped, &missing, &unsupported);
            check_layer_tensor(&file, layer, "ffn_up.weight", &mapped, &missing, &unsupported);
            check_layer_tensor(&file, layer, "ffn_down.weight", &mapped, &missing, &unsupported);
        }
    }

    int forward_ready = 0;
    if (missing == 0 && unsupported == 0) {
        im_qwen3_model model;
        forward_ready = im_qwen3_model_load(&model, path, 1) == 0;
        if (forward_ready) {
            printf("  simdBackend          %s\n", im_simd_backend_name(im_simd_selected_backend()));
            printf("  residencyBudgetMb    %" PRIu64 "\n", im_gguf_residency_budget(&model.gguf) / (1024u * 1024u));
            printf("  residencyMaxTensorMb %" PRIu64 "\n", im_gguf_residency_max_tensor(&model.gguf) / (1024u * 1024u));
            printf("  residencyUsedMb      %" PRIu64 "\n", im_gguf_residency_used(&model.gguf) / (1024u * 1024u));
            printf("  residencyEntries     %" PRIu64 "\n", im_gguf_residency_entries(&model.gguf));
        }
        im_qwen3_model_free(&model);
    }
    printf("  mappedTensors        %" PRIu64 "\n", mapped);
    printf("  missingTensors       %" PRIu64 "\n", missing);
    printf("  unsupportedMatvec    %" PRIu64 "\n", unsupported);
    printf("  nativeStatus         %s\n", forward_ready ? "ready-for-native-decode" : "blocked");

    im_gguf_free(&file);
    return forward_ready ? 0 : 1;
}

static int native_decode(const char * path, uint32_t token_id, uint32_t ctx) {
    im_qwen3_model model;
    if (im_qwen3_model_load(&model, path, ctx) != 0) {
        fprintf(stderr, "error: model is not ready for native Qwen3 decode\n");
        return 1;
    }
    im_kv_cache cache;
    if (im_kv_cache_init(&cache, &model.cfg) != 0) {
        im_qwen3_model_free(&model);
        fprintf(stderr, "error: cannot allocate KV cache\n");
        return 1;
    }
    float * logits = (float *)malloc((size_t)model.cfg.n_vocab * sizeof(float));
    if (!logits) {
        im_kv_cache_free(&cache);
        im_qwen3_model_free(&model);
        fprintf(stderr, "error: cannot allocate logits\n");
        return 1;
    }
    const int rc = im_qwen3_decode(&model, &cache, token_id, logits);
    if (rc != 0) {
        free(logits);
        im_kv_cache_free(&cache);
        im_qwen3_model_free(&model);
        fprintf(stderr, "error: native decode failed\n");
        return 1;
    }
    const uint32_t next = im_qwen3_argmax(logits, model.cfg.n_vocab);
    printf("IronMind native decode\n");
    printf("  token               %u\n", token_id);
    printf("  simdBackend         %s\n", im_simd_backend_name(im_simd_selected_backend()));
    printf("  residencyUsedMb     %" PRIu64 "\n", im_gguf_residency_used(&model.gguf) / (1024u * 1024u));
    printf("  residencyEntries    %" PRIu64 "\n", im_gguf_residency_entries(&model.gguf));
    printf("  logits              %u\n", model.cfg.n_vocab);
    printf("  nextTokenArgmax     %u\n", next);
    printf("  nextTokenLogit      %.9g\n", logits[next]);
    printf("  kvTokens            %u\n", cache.token_count);
    free(logits);
    im_kv_cache_free(&cache);
    im_qwen3_model_free(&model);
    return 0;
}

static uint32_t parse_u32(const char * s, uint32_t fallback) {
    char * end = NULL;
    const unsigned long v = strtoul(s, &end, 10);
    return end && *end == '\0' ? (uint32_t)v : fallback;
}

int main(int argc, char ** argv) {
    if (argc != 2 && argc != 4 && argc != 6) {
        fprintf(stderr, "usage: ironmind-native <model.gguf> [--decode TOKEN] [--ctx N]\n");
        return 2;
    }
    if (argc >= 4 && strcmp(argv[2], "--decode") == 0) {
        uint32_t ctx = 1;
        if (argc == 6 && strcmp(argv[4], "--ctx") == 0) ctx = parse_u32(argv[5], 1);
        return native_decode(argv[1], parse_u32(argv[3], 0), ctx);
    }
    return native_probe(argv[1]);
}
