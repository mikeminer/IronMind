#pragma once

#include <stddef.h>
#include <stdint.h>

typedef struct im_gguf_tensor {
    char * name;
    uint32_t n_dims;
    uint64_t dims[4];
    int32_t type;
    uint64_t relative_offset;
    uint64_t absolute_offset;
    uint64_t size_bytes;
} im_gguf_tensor;

typedef struct im_gguf_file {
    char * path;
    uint32_t version;
    uint64_t kv_count;
    uint64_t tensor_count;
    uint32_t alignment;
    uint64_t data_offset;

    char architecture[64];
    char name[256];
    char tokenizer_model[64];

    uint64_t context_length;
    uint64_t embedding_length;
    uint64_t block_count;
    uint64_t head_count;
    uint64_t head_count_kv;
    uint64_t key_length;
    uint64_t feed_forward_length;
    uint64_t expert_count;
    uint64_t expert_used_count;
    uint64_t expert_feed_forward_length;
    double rope_freq_base;
    double rms_norm_eps;
    uint64_t vocab_items;
    uint64_t file_type;
    uint64_t quantization_version;

    im_gguf_tensor * tensors;
} im_gguf_file;

int im_gguf_load(const char * path, im_gguf_file * out);
void im_gguf_free(im_gguf_file * file);
const im_gguf_tensor * im_gguf_find_tensor(const im_gguf_file * file, const char * name);
int im_gguf_read_tensor_data(const im_gguf_file * file, const im_gguf_tensor * tensor, void * dst, size_t dst_size);
int im_gguf_is_qwen_target(const im_gguf_file * file);
int im_gguf_tensor_matvec_supported(const im_gguf_tensor * tensor);
void im_gguf_print_summary(const im_gguf_file * file);
