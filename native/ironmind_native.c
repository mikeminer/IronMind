#include "ironmind_gguf.h"

#include "ironmind_quant.h"

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

    printf("  mappedTensors        %" PRIu64 "\n", mapped);
    printf("  missingTensors       %" PRIu64 "\n", missing);
    printf("  unsupportedMatvec    %" PRIu64 "\n", unsupported);
    printf("  nativeStatus         %s\n", missing == 0 && unsupported == 0 ? "ready-for-forward-wiring" : "blocked");

    im_gguf_free(&file);
    return (missing == 0 && unsupported == 0) ? 0 : 1;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: ironmind-native <model.gguf>\n");
        return 2;
    }
    return native_probe(argv[1]);
}
