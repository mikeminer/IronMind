#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#define IM_FSEEK _fseeki64
#define IM_FTELL _ftelli64
#else
#define IM_FSEEK fseeko
#define IM_FTELL ftello
#endif

enum gguf_type {
    GGUF_TYPE_UINT8 = 0,
    GGUF_TYPE_INT8 = 1,
    GGUF_TYPE_UINT16 = 2,
    GGUF_TYPE_INT16 = 3,
    GGUF_TYPE_UINT32 = 4,
    GGUF_TYPE_INT32 = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL = 7,
    GGUF_TYPE_STRING = 8,
    GGUF_TYPE_ARRAY = 9,
    GGUF_TYPE_UINT64 = 10,
    GGUF_TYPE_INT64 = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

struct im_summary {
    uint32_t version;
    uint64_t kv_count;
    uint64_t tensor_count;
    uint32_t alignment;
    uint64_t data_offset;

    char architecture[64];
    char name[256];
    char tokenizer[64];

    uint64_t context_length;
    uint64_t embedding_length;
    uint64_t block_count;
    uint64_t head_count;
    uint64_t head_count_kv;
    uint64_t expert_count;
    uint64_t expert_used_count;
    uint64_t expert_ffn_length;
    uint64_t vocab_items;
    uint64_t file_type;
    uint64_t quantization_version;

    bool has_token_embd;
    bool has_output_norm;
};

static bool read_exact(FILE * f, void * dst, size_t n) {
    return fread(dst, 1, n, f) == n;
}

static bool skip_bytes(FILE * f, uint64_t n) {
    if (n > INT64_MAX) return false;
    return IM_FSEEK(f, (int64_t)n, SEEK_CUR) == 0;
}

static bool read_u8(FILE * f, uint8_t * out) {
    return read_exact(f, out, 1);
}

static bool read_u32(FILE * f, uint32_t * out) {
    uint8_t b[4];
    if (!read_exact(f, b, sizeof(b))) return false;
    *out = ((uint32_t)b[0]) | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return true;
}

static bool read_i32(FILE * f, int32_t * out) {
    uint32_t tmp;
    if (!read_u32(f, &tmp)) return false;
    memcpy(out, &tmp, sizeof(tmp));
    return true;
}

static bool read_u64(FILE * f, uint64_t * out) {
    uint8_t b[8];
    if (!read_exact(f, b, sizeof(b))) return false;
    *out = ((uint64_t)b[0]) |
           ((uint64_t)b[1] << 8) |
           ((uint64_t)b[2] << 16) |
           ((uint64_t)b[3] << 24) |
           ((uint64_t)b[4] << 32) |
           ((uint64_t)b[5] << 40) |
           ((uint64_t)b[6] << 48) |
           ((uint64_t)b[7] << 56);
    return true;
}

static bool read_string(FILE * f, char * dst, size_t cap) {
    uint64_t n = 0;
    if (!read_u64(f, &n)) return false;
    if (n >= cap) {
        if (cap > 0) dst[0] = '\0';
        return skip_bytes(f, n);
    }
    if (!read_exact(f, dst, (size_t)n)) return false;
    dst[n] = '\0';
    return true;
}

static bool skip_string(FILE * f) {
    uint64_t n = 0;
    return read_u64(f, &n) && skip_bytes(f, n);
}

static size_t scalar_size(int32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            return 1;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            return 2;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            return 4;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            return 8;
        default:
            return 0;
    }
}

static bool skip_value(FILE * f, int32_t type) {
    if (type == GGUF_TYPE_STRING) return skip_string(f);
    const size_t n = scalar_size(type);
    return n != 0 && skip_bytes(f, n);
}

static bool skip_array(FILE * f, int32_t inner_type, uint64_t count) {
    if (inner_type == GGUF_TYPE_STRING) {
        for (uint64_t i = 0; i < count; i++) {
            if (!skip_string(f)) return false;
        }
        return true;
    }
    const size_t n = scalar_size(inner_type);
    if (n == 0 || count > UINT64_MAX / n) return false;
    return skip_bytes(f, count * n);
}

static bool has_suffix(const char * s, const char * suffix) {
    const size_t n = strlen(s);
    const size_t m = strlen(suffix);
    return n >= m && strcmp(s + n - m, suffix) == 0;
}

static uint64_t read_scalar_as_u64(FILE * f, int32_t type, bool * ok) {
    uint8_t u8 = 0;
    uint32_t u32 = 0;
    uint64_t u64 = 0;
    *ok = true;
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_BOOL:
            *ok = read_u8(f, &u8);
            return u8;
        case GGUF_TYPE_UINT32:
            *ok = read_u32(f, &u32);
            return u32;
        case GGUF_TYPE_UINT64:
            *ok = read_u64(f, &u64);
            return u64;
        case GGUF_TYPE_INT32:
            *ok = read_u32(f, &u32);
            return (int32_t)u32 < 0 ? 0 : (uint64_t)(int32_t)u32;
        case GGUF_TYPE_INT64:
            *ok = read_u64(f, &u64);
            return (int64_t)u64 < 0 ? 0 : u64;
        default:
            *ok = skip_value(f, type);
            return 0;
    }
}

static bool read_interesting_kv(FILE * f, const char * key, int32_t type, struct im_summary * s) {
    bool ok = true;

    if (strcmp(key, "general.architecture") == 0 && type == GGUF_TYPE_STRING) return read_string(f, s->architecture, sizeof(s->architecture));
    if (strcmp(key, "general.name") == 0 && type == GGUF_TYPE_STRING) return read_string(f, s->name, sizeof(s->name));
    if (strcmp(key, "tokenizer.ggml.model") == 0 && type == GGUF_TYPE_STRING) return read_string(f, s->tokenizer, sizeof(s->tokenizer));

    if (strcmp(key, "general.file_type") == 0) s->file_type = read_scalar_as_u64(f, type, &ok);
    else if (strcmp(key, "general.quantization_version") == 0) s->quantization_version = read_scalar_as_u64(f, type, &ok);
    else if (strcmp(key, "general.alignment") == 0) s->alignment = (uint32_t)read_scalar_as_u64(f, type, &ok);
    else if (has_suffix(key, ".context_length")) s->context_length = read_scalar_as_u64(f, type, &ok);
    else if (has_suffix(key, ".embedding_length")) s->embedding_length = read_scalar_as_u64(f, type, &ok);
    else if (has_suffix(key, ".block_count")) s->block_count = read_scalar_as_u64(f, type, &ok);
    else if (has_suffix(key, ".attention.head_count")) s->head_count = read_scalar_as_u64(f, type, &ok);
    else if (has_suffix(key, ".attention.head_count_kv")) s->head_count_kv = read_scalar_as_u64(f, type, &ok);
    else if (has_suffix(key, ".expert_count")) s->expert_count = read_scalar_as_u64(f, type, &ok);
    else if (has_suffix(key, ".expert_used_count")) s->expert_used_count = read_scalar_as_u64(f, type, &ok);
    else if (has_suffix(key, ".expert_feed_forward_length")) s->expert_ffn_length = read_scalar_as_u64(f, type, &ok);
    else ok = skip_value(f, type);

    return ok;
}

static bool read_kv(FILE * f, struct im_summary * s) {
    char key[512];
    int32_t type = 0;
    if (!read_string(f, key, sizeof(key))) return false;
    if (!read_i32(f, &type)) return false;

    if (type == GGUF_TYPE_ARRAY) {
        int32_t inner_type = 0;
        uint64_t count = 0;
        if (!read_i32(f, &inner_type) || !read_u64(f, &count)) return false;
        if (strcmp(key, "tokenizer.ggml.tokens") == 0) s->vocab_items = count;
        return skip_array(f, inner_type, count);
    }

    return read_interesting_kv(f, key, type, s);
}

static uint64_t align_u64(uint64_t value, uint32_t alignment) {
    const uint64_t mask = (uint64_t)alignment - 1;
    return (value + mask) & ~mask;
}

static bool inspect_file(const char * path, struct im_summary * s) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "error: cannot open %s: %s\n", path, strerror(errno));
        return false;
    }

    bool ok = false;
    char magic[4];
    if (!read_exact(f, magic, sizeof(magic)) || memcmp(magic, "GGUF", 4) != 0) {
        fprintf(stderr, "error: invalid GGUF magic\n");
        goto done;
    }

    s->alignment = 32;
    if (!read_u32(f, &s->version) || s->version < 2 || s->version > 3) {
        fprintf(stderr, "error: unsupported GGUF version %" PRIu32 "\n", s->version);
        goto done;
    }
    if (!read_u64(f, &s->tensor_count) || !read_u64(f, &s->kv_count)) goto done;

    for (uint64_t i = 0; i < s->kv_count; i++) {
        if (!read_kv(f, s)) {
            fprintf(stderr, "error: failed reading KV pair %" PRIu64 "\n", i);
            goto done;
        }
    }

    for (uint64_t i = 0; i < s->tensor_count; i++) {
        char name[256];
        uint32_t n_dims = 0;
        if (!read_string(f, name, sizeof(name)) || !read_u32(f, &n_dims)) goto done;
        for (uint32_t d = 0; d < n_dims; d++) {
            uint64_t dim = 0;
            if (!read_u64(f, &dim)) goto done;
        }
        int32_t tensor_type = 0;
        uint64_t tensor_offset = 0;
        if (!read_i32(f, &tensor_type) || !read_u64(f, &tensor_offset)) goto done;
        (void)tensor_type;
        (void)tensor_offset;
        if (strcmp(name, "token_embd.weight") == 0) s->has_token_embd = true;
        if (strcmp(name, "output_norm.weight") == 0) s->has_output_norm = true;
    }

    const int64_t current = IM_FTELL(f);
    if (current >= 0) s->data_offset = align_u64((uint64_t)current, s->alignment);
    ok = true;

done:
    fclose(f);
    return ok;
}

static bool is_target_arch(const char * arch) {
    return strcmp(arch, "qwen3") == 0 || strcmp(arch, "qwen3moe") == 0;
}

static void print_summary(const struct im_summary * s) {
    const bool is_moe = strstr(s->architecture, "moe") != NULL;
    bool compatible = true;
    compatible = compatible && is_target_arch(s->architecture);
    compatible = compatible && s->context_length != 0;
    compatible = compatible && s->embedding_length != 0;
    compatible = compatible && s->block_count != 0;
    compatible = compatible && s->head_count != 0;
    compatible = compatible && s->head_count_kv != 0;
    compatible = compatible && s->tokenizer[0] != '\0';
    compatible = compatible && s->vocab_items != 0;
    if (is_moe) compatible = compatible && s->expert_count != 0 && s->expert_used_count != 0;

    printf("IronMind native GGUF inspector\n");
    printf("  architecture         %s\n", s->architecture[0] ? s->architecture : "-");
    printf("  name                 %s\n", s->name[0] ? s->name : "-");
    printf("  fileType             %" PRIu64 "\n", s->file_type);
    printf("  quantizationVersion  %" PRIu64 "\n", s->quantization_version);
    printf("  contextLength        %" PRIu64 "\n", s->context_length);
    printf("  embeddingLength      %" PRIu64 "\n", s->embedding_length);
    printf("  blockCount           %" PRIu64 "\n", s->block_count);
    printf("  headCount            %" PRIu64 "\n", s->head_count);
    printf("  headCountKv          %" PRIu64 "\n", s->head_count_kv);
    printf("  expertCount          %" PRIu64 "\n", s->expert_count);
    printf("  expertUsedCount      %" PRIu64 "\n", s->expert_used_count);
    printf("  tokenizer            %s\n", s->tokenizer[0] ? s->tokenizer : "-");
    printf("  vocabItems           %" PRIu64 "\n", s->vocab_items);
    printf("  tensorCount          %" PRIu64 "\n", s->tensor_count);
    printf("  kvCount              %" PRIu64 "\n", s->kv_count);
    printf("  dataOffset           %" PRIu64 "\n", s->data_offset);
    printf("  target               %s\n", compatible ? "compatible" : "not compatible");
    if (!is_target_arch(s->architecture)) printf("  issue                unsupported architecture\n");
    if (!s->has_token_embd) printf("  warning              tensor token_embd.weight not found\n");
    if (!s->has_output_norm) printf("  warning              tensor output_norm.weight not found\n");
    if (!is_moe) printf("  warning              dense Qwen3 path; MoE router disabled\n");
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: ironmind-inspect <model.gguf>\n");
        return 2;
    }

    struct im_summary summary;
    memset(&summary, 0, sizeof(summary));
    if (!inspect_file(argv[1], &summary)) return 1;
    print_summary(&summary);
    return 0;
}
