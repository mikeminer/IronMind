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

typedef struct token_list {
    uint32_t * data;
    size_t count;
} token_list;

static void token_list_free(token_list * list) {
    if (!list) return;
    free(list->data);
    list->data = NULL;
    list->count = 0;
}

static int parse_token_csv(const char * text, token_list * out) {
    if (!text || !out) return -1;
    memset(out, 0, sizeof(*out));
    size_t cap = 16;
    out->data = (uint32_t *)malloc(cap * sizeof(uint32_t));
    if (!out->data) return -1;

    const char * p = text;
    while (*p) {
        while (*p == ',' || *p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') p++;
        if (!*p) break;
        char * end = NULL;
        const unsigned long value = strtoul(p, &end, 10);
        if (end == p || value > UINT32_MAX) {
            token_list_free(out);
            return -1;
        }
        if (out->count == cap) {
            cap *= 2u;
            uint32_t * next = (uint32_t *)realloc(out->data, cap * sizeof(uint32_t));
            if (!next) {
                token_list_free(out);
                return -1;
            }
            out->data = next;
        }
        out->data[out->count++] = (uint32_t)value;
        p = end;
    }

    if (!out->count) {
        token_list_free(out);
        return -1;
    }
    return 0;
}

static char * read_text_file(const char * path) {
    FILE * file = fopen(path, "rb");
    if (!file) return NULL;
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return NULL;
    }
    const long n = ftell(file);
    if (n < 0) {
        fclose(file);
        return NULL;
    }
    rewind(file);
    char * out = (char *)malloc((size_t)n + 1u);
    if (!out) {
        fclose(file);
        return NULL;
    }
    if (fread(out, 1, (size_t)n, file) != (size_t)n) {
        free(out);
        fclose(file);
        return NULL;
    }
    out[n] = '\0';
    fclose(file);
    return out;
}

static void print_token_json(const uint32_t * tokens, uint32_t count) {
    printf("[");
    for (uint32_t i = 0; i < count; i++) {
        if (i) printf(",");
        printf("%u", tokens[i]);
    }
    printf("]");
}

static int native_generate(const char * path, const token_list * prompt, uint32_t generate, uint32_t ctx, const char * save_kv, int as_json) {
    if (!prompt || !prompt->count || !generate) {
        fprintf(stderr, "error: --tokens/--tokens-file and --generate are required\n");
        return 2;
    }

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
    uint32_t * out = (uint32_t *)calloc((size_t)generate, sizeof(uint32_t));
    if (!logits || !out) {
        free(logits);
        free(out);
        im_kv_cache_free(&cache);
        im_qwen3_model_free(&model);
        fprintf(stderr, "error: cannot allocate generation buffers\n");
        return 1;
    }

    int ok = 1;
    for (size_t i = 0; i < prompt->count; i++) {
        if (im_qwen3_decode(&model, &cache, prompt->data[i], logits) != 0) {
            ok = 0;
            break;
        }
    }

    uint32_t produced = 0;
    while (ok && produced < generate && cache.token_count < model.cfg.max_seq) {
        const uint32_t next = im_qwen3_argmax(logits, model.cfg.n_vocab);
        out[produced++] = next;
        if (im_qwen3_decode(&model, &cache, next, logits) != 0) {
            ok = 0;
            break;
        }
    }

    if (ok && save_kv && im_kv_cache_save(&cache, save_kv) != 0) {
        fprintf(stderr, "error: cannot save native KV payload\n");
        ok = 0;
    }

    if (ok) {
        if (as_json) {
            printf("{\"promptTokens\":%zu,\"generatedTokens\":%u,\"kvTokens\":%u,\"simdBackend\":\"%s\","
                   "\"residencyUsedMb\":%" PRIu64 ",\"residencyEntries\":%" PRIu64 ",\"tokenIds\":",
                   prompt->count,
                   produced,
                   cache.token_count,
                   im_simd_backend_name(im_simd_selected_backend()),
                   im_gguf_residency_used(&model.gguf) / (1024u * 1024u),
                   im_gguf_residency_entries(&model.gguf));
            print_token_json(out, produced);
            printf("}\n");
        } else {
            print_token_json(out, produced);
            printf("\n");
        }
    }

    free(logits);
    free(out);
    im_kv_cache_free(&cache);
    im_qwen3_model_free(&model);
    return ok ? 0 : 1;
}

static uint32_t parse_u32(const char * s, uint32_t fallback) {
    char * end = NULL;
    const unsigned long v = strtoul(s, &end, 10);
    return end && *end == '\0' ? (uint32_t)v : fallback;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: ironmind-native <model.gguf> [--decode TOKEN] [--tokens CSV|--tokens-file PATH --generate N] [--ctx N] [--json] [--save-kv PATH]\n");
        return 2;
    }

    const char * tokens_text = NULL;
    const char * tokens_file = NULL;
    const char * save_kv = NULL;
    uint32_t decode_token = 0;
    uint32_t generate = 0;
    uint32_t ctx = 1;
    int do_decode = 0;
    int as_json = 0;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--decode") == 0 && i + 1 < argc) {
            do_decode = 1;
            decode_token = parse_u32(argv[++i], 0);
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            tokens_text = argv[++i];
        } else if (strcmp(argv[i], "--tokens-file") == 0 && i + 1 < argc) {
            tokens_file = argv[++i];
        } else if (strcmp(argv[i], "--generate") == 0 && i + 1 < argc) {
            generate = parse_u32(argv[++i], 0);
        } else if (strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) {
            ctx = parse_u32(argv[++i], 1);
        } else if (strcmp(argv[i], "--save-kv") == 0 && i + 1 < argc) {
            save_kv = argv[++i];
        } else if (strcmp(argv[i], "--json") == 0) {
            as_json = 1;
        } else {
            fprintf(stderr, "usage: ironmind-native <model.gguf> [--decode TOKEN] [--tokens CSV|--tokens-file PATH --generate N] [--ctx N] [--json] [--save-kv PATH]\n");
            return 2;
        }
    }

    if (do_decode) return native_decode(argv[1], decode_token, ctx);

    if (tokens_text || tokens_file) {
        char * file_text = NULL;
        if (tokens_file) {
            file_text = read_text_file(tokens_file);
            if (!file_text) {
                fprintf(stderr, "error: cannot read tokens file\n");
                return 1;
            }
            tokens_text = file_text;
        }
        token_list prompt;
        if (parse_token_csv(tokens_text, &prompt) != 0) {
            free(file_text);
            fprintf(stderr, "error: invalid token list\n");
            return 2;
        }
        const int rc = native_generate(argv[1], &prompt, generate, ctx, save_kv, as_json);
        token_list_free(&prompt);
        free(file_text);
        return rc;
    }
    return native_probe(argv[1]);
}
