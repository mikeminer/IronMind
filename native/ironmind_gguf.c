#include "ironmind_gguf.h"

#include "ironmind_quant.h"

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
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

enum im_gguf_kv_type {
    IM_GGUF_TYPE_UINT8 = 0,
    IM_GGUF_TYPE_INT8 = 1,
    IM_GGUF_TYPE_UINT16 = 2,
    IM_GGUF_TYPE_INT16 = 3,
    IM_GGUF_TYPE_UINT32 = 4,
    IM_GGUF_TYPE_INT32 = 5,
    IM_GGUF_TYPE_FLOAT32 = 6,
    IM_GGUF_TYPE_BOOL = 7,
    IM_GGUF_TYPE_STRING = 8,
    IM_GGUF_TYPE_ARRAY = 9,
    IM_GGUF_TYPE_UINT64 = 10,
    IM_GGUF_TYPE_INT64 = 11,
    IM_GGUF_TYPE_FLOAT64 = 12,
};

static char * im_strdup_local(const char * s) {
    const size_t n = strlen(s) + 1u;
    char * out = (char *)malloc(n);
    if (out) memcpy(out, s, n);
    return out;
}

static int read_exact(FILE * f, void * dst, size_t n) {
    return fread(dst, 1, n, f) == n;
}

static int skip_bytes(FILE * f, uint64_t n) {
    if (n > INT64_MAX) return 0;
    return IM_FSEEK(f, (int64_t)n, SEEK_CUR) == 0;
}

static int read_u8(FILE * f, uint8_t * out) {
    return read_exact(f, out, 1);
}

static int read_u32(FILE * f, uint32_t * out) {
    uint8_t b[4];
    if (!read_exact(f, b, sizeof(b))) return 0;
    *out = ((uint32_t)b[0]) | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return 1;
}

static int read_i32(FILE * f, int32_t * out) {
    uint32_t tmp = 0;
    if (!read_u32(f, &tmp)) return 0;
    memcpy(out, &tmp, sizeof(tmp));
    return 1;
}

static int read_u64(FILE * f, uint64_t * out) {
    uint8_t b[8];
    if (!read_exact(f, b, sizeof(b))) return 0;
    *out = ((uint64_t)b[0]) |
           ((uint64_t)b[1] << 8) |
           ((uint64_t)b[2] << 16) |
           ((uint64_t)b[3] << 24) |
           ((uint64_t)b[4] << 32) |
           ((uint64_t)b[5] << 40) |
           ((uint64_t)b[6] << 48) |
           ((uint64_t)b[7] << 56);
    return 1;
}

static int read_f32(FILE * f, float * out) {
    uint32_t bits = 0;
    if (!read_u32(f, &bits)) return 0;
    memcpy(out, &bits, sizeof(*out));
    return 1;
}

static int read_f64(FILE * f, double * out) {
    uint64_t bits = 0;
    if (!read_u64(f, &bits)) return 0;
    memcpy(out, &bits, sizeof(*out));
    return 1;
}

static int read_string_alloc(FILE * f, char ** out) {
    uint64_t n = 0;
    if (!read_u64(f, &n) || n > SIZE_MAX - 1u) return 0;
    char * s = (char *)malloc((size_t)n + 1u);
    if (!s) return 0;
    if (!read_exact(f, s, (size_t)n)) {
        free(s);
        return 0;
    }
    s[n] = '\0';
    *out = s;
    return 1;
}

static int read_string_fixed(FILE * f, char * dst, size_t cap) {
    char * s = NULL;
    if (!read_string_alloc(f, &s)) return 0;
    if (cap) {
        size_t n = strlen(s);
        if (n >= cap) n = cap - 1u;
        memcpy(dst, s, n);
        dst[n] = '\0';
    }
    free(s);
    return 1;
}

static int skip_string(FILE * f) {
    uint64_t n = 0;
    return read_u64(f, &n) && skip_bytes(f, n);
}

static size_t scalar_size(int32_t type) {
    switch (type) {
        case IM_GGUF_TYPE_UINT8:
        case IM_GGUF_TYPE_INT8:
        case IM_GGUF_TYPE_BOOL:
            return 1;
        case IM_GGUF_TYPE_UINT16:
        case IM_GGUF_TYPE_INT16:
            return 2;
        case IM_GGUF_TYPE_UINT32:
        case IM_GGUF_TYPE_INT32:
        case IM_GGUF_TYPE_FLOAT32:
            return 4;
        case IM_GGUF_TYPE_UINT64:
        case IM_GGUF_TYPE_INT64:
        case IM_GGUF_TYPE_FLOAT64:
            return 8;
        default:
            return 0;
    }
}

static int skip_value(FILE * f, int32_t type) {
    if (type == IM_GGUF_TYPE_STRING) return skip_string(f);
    const size_t n = scalar_size(type);
    return n != 0 && skip_bytes(f, n);
}

static int skip_array(FILE * f, int32_t inner_type, uint64_t count) {
    if (inner_type == IM_GGUF_TYPE_STRING) {
        for (uint64_t i = 0; i < count; i++) {
            if (!skip_string(f)) return 0;
        }
        return 1;
    }
    const size_t n = scalar_size(inner_type);
    if (n == 0 || count > UINT64_MAX / n) return 0;
    return skip_bytes(f, count * n);
}

static int has_suffix(const char * s, const char * suffix) {
    const size_t n = strlen(s);
    const size_t m = strlen(suffix);
    return n >= m && strcmp(s + n - m, suffix) == 0;
}

static uint64_t read_scalar_as_u64(FILE * f, int32_t type, int * ok) {
    uint8_t u8 = 0;
    uint32_t u32 = 0;
    uint64_t u64 = 0;
    *ok = 1;
    switch (type) {
        case IM_GGUF_TYPE_UINT8:
        case IM_GGUF_TYPE_BOOL:
            *ok = read_u8(f, &u8);
            return u8;
        case IM_GGUF_TYPE_UINT32:
            *ok = read_u32(f, &u32);
            return u32;
        case IM_GGUF_TYPE_UINT64:
            *ok = read_u64(f, &u64);
            return u64;
        case IM_GGUF_TYPE_INT32:
            *ok = read_u32(f, &u32);
            return (int32_t)u32 < 0 ? 0 : (uint64_t)(int32_t)u32;
        case IM_GGUF_TYPE_INT64:
            *ok = read_u64(f, &u64);
            return (int64_t)u64 < 0 ? 0 : u64;
        default:
            *ok = skip_value(f, type);
            return 0;
    }
}

static double read_scalar_as_double(FILE * f, int32_t type, int * ok) {
    float f32 = 0.0f;
    double f64 = 0.0;
    uint64_t u = 0;
    *ok = 1;
    if (type == IM_GGUF_TYPE_FLOAT32) {
        *ok = read_f32(f, &f32);
        return f32;
    }
    if (type == IM_GGUF_TYPE_FLOAT64) {
        *ok = read_f64(f, &f64);
        return f64;
    }
    u = read_scalar_as_u64(f, type, ok);
    return (double)u;
}

static int read_interesting_kv(FILE * f, const char * key, int32_t type, im_gguf_file * out) {
    int ok = 1;

    if (strcmp(key, "general.architecture") == 0 && type == IM_GGUF_TYPE_STRING) return read_string_fixed(f, out->architecture, sizeof(out->architecture));
    if (strcmp(key, "general.name") == 0 && type == IM_GGUF_TYPE_STRING) return read_string_fixed(f, out->name, sizeof(out->name));
    if (strcmp(key, "tokenizer.ggml.model") == 0 && type == IM_GGUF_TYPE_STRING) return read_string_fixed(f, out->tokenizer_model, sizeof(out->tokenizer_model));

    if (strcmp(key, "general.file_type") == 0) out->file_type = read_scalar_as_u64(f, type, &ok);
    else if (strcmp(key, "general.quantization_version") == 0) out->quantization_version = read_scalar_as_u64(f, type, &ok);
    else if (strcmp(key, "general.alignment") == 0) out->alignment = (uint32_t)read_scalar_as_u64(f, type, &ok);
    else if (has_suffix(key, ".context_length")) out->context_length = read_scalar_as_u64(f, type, &ok);
    else if (has_suffix(key, ".embedding_length")) out->embedding_length = read_scalar_as_u64(f, type, &ok);
    else if (has_suffix(key, ".block_count")) out->block_count = read_scalar_as_u64(f, type, &ok);
    else if (has_suffix(key, ".attention.head_count")) out->head_count = read_scalar_as_u64(f, type, &ok);
    else if (has_suffix(key, ".attention.head_count_kv")) out->head_count_kv = read_scalar_as_u64(f, type, &ok);
    else if (has_suffix(key, ".attention.key_length")) out->key_length = read_scalar_as_u64(f, type, &ok);
    else if (has_suffix(key, ".feed_forward_length")) out->feed_forward_length = read_scalar_as_u64(f, type, &ok);
    else if (has_suffix(key, ".expert_count")) out->expert_count = read_scalar_as_u64(f, type, &ok);
    else if (has_suffix(key, ".expert_used_count")) out->expert_used_count = read_scalar_as_u64(f, type, &ok);
    else if (has_suffix(key, ".expert_feed_forward_length")) out->expert_feed_forward_length = read_scalar_as_u64(f, type, &ok);
    else if (has_suffix(key, ".rope.freq_base")) out->rope_freq_base = read_scalar_as_double(f, type, &ok);
    else if (has_suffix(key, ".attention.layer_norm_rms_epsilon")) out->rms_norm_eps = read_scalar_as_double(f, type, &ok);
    else ok = skip_value(f, type);

    return ok;
}

static int read_kv(FILE * f, im_gguf_file * out) {
    char * key = NULL;
    int32_t type = 0;
    int ok = read_string_alloc(f, &key) && read_i32(f, &type);
    if (!ok) {
        free(key);
        return 0;
    }

    if (type == IM_GGUF_TYPE_ARRAY) {
        int32_t inner_type = 0;
        uint64_t count = 0;
        ok = read_i32(f, &inner_type) && read_u64(f, &count);
        if (ok && strcmp(key, "tokenizer.ggml.tokens") == 0) out->vocab_items = count;
        ok = ok && skip_array(f, inner_type, count);
    } else {
        ok = read_interesting_kv(f, key, type, out);
    }
    free(key);
    return ok;
}

static uint64_t align_u64(uint64_t value, uint32_t alignment) {
    const uint64_t mask = (uint64_t)alignment - 1u;
    return (value + mask) & ~mask;
}

int im_gguf_load(const char * path, im_gguf_file * out) {
    if (!path || !out) return -1;
    memset(out, 0, sizeof(*out));
    out->alignment = 32;
    out->rope_freq_base = 1000000.0;
    out->rms_norm_eps = 1e-6;

    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "error: cannot open %s: errno %d\n", path, errno);
        return -1;
    }

    int ok = 0;
    char magic[4];
    if (!read_exact(f, magic, sizeof(magic)) || memcmp(magic, "GGUF", 4) != 0) goto done;
    if (!read_u32(f, &out->version) || out->version < 2 || out->version > 3) goto done;
    if (!read_u64(f, &out->tensor_count) || !read_u64(f, &out->kv_count)) goto done;

    for (uint64_t i = 0; i < out->kv_count; i++) {
        if (!read_kv(f, out)) goto done;
    }

    if (out->tensor_count > SIZE_MAX / sizeof(im_gguf_tensor)) goto done;
    out->tensors = (im_gguf_tensor *)calloc((size_t)out->tensor_count, sizeof(im_gguf_tensor));
    if (!out->tensors) goto done;

    for (uint64_t i = 0; i < out->tensor_count; i++) {
        im_gguf_tensor * tensor = &out->tensors[i];
        if (!read_string_alloc(f, &tensor->name) || !read_u32(f, &tensor->n_dims)) goto done;
        if (tensor->n_dims > 4) goto done;
        for (uint32_t d = 0; d < tensor->n_dims; d++) {
            if (!read_u64(f, &tensor->dims[d])) goto done;
        }
        if (!read_i32(f, &tensor->type) || !read_u64(f, &tensor->relative_offset)) goto done;
        tensor->size_bytes = im_quant_tensor_size(tensor->type, tensor->dims, tensor->n_dims);
    }

    {
        const int64_t current = IM_FTELL(f);
        if (current < 0) goto done;
        out->data_offset = align_u64((uint64_t)current, out->alignment);
    }
    for (uint64_t i = 0; i < out->tensor_count; i++) {
        out->tensors[i].absolute_offset = out->data_offset + out->tensors[i].relative_offset;
    }
    out->path = im_strdup_local(path);
    ok = out->path != NULL;

done:
    fclose(f);
    if (!ok) {
        im_gguf_free(out);
        return -1;
    }
    return 0;
}

void im_gguf_free(im_gguf_file * file) {
    if (!file) return;
    free(file->path);
    if (file->tensors) {
        for (uint64_t i = 0; i < file->tensor_count; i++) free(file->tensors[i].name);
    }
    free(file->tensors);
    memset(file, 0, sizeof(*file));
}

const im_gguf_tensor * im_gguf_find_tensor(const im_gguf_file * file, const char * name) {
    if (!file || !name || !file->tensors) return NULL;
    for (uint64_t i = 0; i < file->tensor_count; i++) {
        if (file->tensors[i].name && strcmp(file->tensors[i].name, name) == 0) return &file->tensors[i];
    }
    return NULL;
}

int im_gguf_read_tensor_data(const im_gguf_file * file, const im_gguf_tensor * tensor, void * dst, size_t dst_size) {
    if (!file || !tensor || !dst || dst_size < tensor->size_bytes) return -1;
    FILE * f = fopen(file->path, "rb");
    if (!f) return -1;
    const int ok = IM_FSEEK(f, (int64_t)tensor->absolute_offset, SEEK_SET) == 0 &&
                   fread(dst, 1, (size_t)tensor->size_bytes, f) == (size_t)tensor->size_bytes;
    fclose(f);
    return ok ? 0 : -1;
}

int im_gguf_is_qwen_target(const im_gguf_file * file) {
    if (!file) return 0;
    return strcmp(file->architecture, "qwen3") == 0 || strcmp(file->architecture, "qwen3moe") == 0;
}

int im_gguf_tensor_matvec_supported(const im_gguf_tensor * tensor) {
    if (!tensor) return 0;
    const im_quant_type_info * info = im_quant_type_info_for(tensor->type);
    return info && info->matvec_supported;
}

void im_gguf_print_summary(const im_gguf_file * file) {
    if (!file) return;
    printf("IronMind native GGUF loader\n");
    printf("  architecture         %s\n", file->architecture[0] ? file->architecture : "-");
    printf("  name                 %s\n", file->name[0] ? file->name : "-");
    printf("  tokenizer            %s\n", file->tokenizer_model[0] ? file->tokenizer_model : "-");
    printf("  contextLength        %" PRIu64 "\n", file->context_length);
    printf("  embeddingLength      %" PRIu64 "\n", file->embedding_length);
    printf("  blockCount           %" PRIu64 "\n", file->block_count);
    printf("  headCount            %" PRIu64 "\n", file->head_count);
    printf("  headCountKv          %" PRIu64 "\n", file->head_count_kv);
    printf("  feedForwardLength    %" PRIu64 "\n", file->feed_forward_length);
    printf("  expertCount          %" PRIu64 "\n", file->expert_count);
    printf("  expertUsedCount      %" PRIu64 "\n", file->expert_used_count);
    printf("  tensorCount          %" PRIu64 "\n", file->tensor_count);
    printf("  dataOffset           %" PRIu64 "\n", file->data_offset);
    printf("  target               %s\n", im_gguf_is_qwen_target(file) ? "qwen-compatible" : "not-compatible");
}
