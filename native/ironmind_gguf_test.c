#include "ironmind_gguf.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void expect(int condition, const char * message) {
    if (!condition) {
        fprintf(stderr, "native gguf test failed: %s\n", message);
        exit(1);
    }
}

static void write_u32(FILE * f, uint32_t v) {
    uint8_t b[4] = {(uint8_t)v, (uint8_t)(v >> 8), (uint8_t)(v >> 16), (uint8_t)(v >> 24)};
    fwrite(b, 1, sizeof(b), f);
}

static void write_i32(FILE * f, int32_t v) {
    write_u32(f, (uint32_t)v);
}

static void write_u64(FILE * f, uint64_t v) {
    uint8_t b[8] = {
        (uint8_t)v, (uint8_t)(v >> 8), (uint8_t)(v >> 16), (uint8_t)(v >> 24),
        (uint8_t)(v >> 32), (uint8_t)(v >> 40), (uint8_t)(v >> 48), (uint8_t)(v >> 56)
    };
    fwrite(b, 1, sizeof(b), f);
}

static void write_f32(FILE * f, float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    write_u32(f, bits);
}

static void write_string(FILE * f, const char * s) {
    write_u64(f, (uint64_t)strlen(s));
    fwrite(s, 1, strlen(s), f);
}

static void write_string_kv(FILE * f, const char * key, const char * value) {
    write_string(f, key);
    write_i32(f, 8);
    write_string(f, value);
}

static void write_u32_kv(FILE * f, const char * key, uint32_t value) {
    write_string(f, key);
    write_i32(f, 4);
    write_u32(f, value);
}

static void write_f32_kv(FILE * f, const char * key, float value) {
    write_string(f, key);
    write_i32(f, 6);
    write_f32(f, value);
}

static void write_tokens_kv(FILE * f) {
    write_string(f, "tokenizer.ggml.tokens");
    write_i32(f, 9);
    write_i32(f, 8);
    write_u64(f, 2);
    write_string(f, "a");
    write_string(f, "b");
}

int main(void) {
    const char * path = "ironmind-gguf-test.gguf";
    FILE * f = fopen(path, "wb");
    expect(f != NULL, "create fixture");

    fwrite("GGUF", 1, 4, f);
    write_u32(f, 3);
    write_u64(f, 1);
    write_u64(f, 12);

    write_string_kv(f, "general.architecture", "qwen3");
    write_string_kv(f, "general.name", "IronMind tiny");
    write_string_kv(f, "tokenizer.ggml.model", "gpt2");
    write_u32_kv(f, "qwen3.context_length", 131072);
    write_u32_kv(f, "qwen3.embedding_length", 2);
    write_u32_kv(f, "qwen3.block_count", 1);
    write_u32_kv(f, "qwen3.attention.head_count", 1);
    write_u32_kv(f, "qwen3.attention.head_count_kv", 1);
    write_u32_kv(f, "qwen3.feed_forward_length", 4);
    write_f32_kv(f, "qwen3.rope.freq_base", 10000.0f);
    write_f32_kv(f, "qwen3.attention.layer_norm_rms_epsilon", 1e-6f);
    write_tokens_kv(f);

    write_string(f, "token_embd.weight");
    write_u32(f, 2);
    write_u64(f, 2);
    write_u64(f, 2);
    write_i32(f, 0);
    write_u64(f, 0);

    while ((ftell(f) % 32) != 0) fputc(0, f);
    write_f32(f, 1.0f);
    write_f32(f, 2.0f);
    write_f32(f, 3.0f);
    write_f32(f, 4.0f);
    fclose(f);

    im_gguf_file gguf;
    expect(im_gguf_load(path, &gguf) == 0, "load fixture");
    expect(strcmp(gguf.architecture, "qwen3") == 0, "architecture");
    expect(gguf.context_length == 131072, "context");
    expect(gguf.embedding_length == 2, "embedding");
    expect(gguf.vocab_items == 2, "vocab");
    expect(im_gguf_is_qwen_target(&gguf), "target");

    const im_gguf_tensor * tensor = im_gguf_find_tensor(&gguf, "token_embd.weight");
    expect(tensor != NULL, "find tensor");
    expect(tensor->size_bytes == 16, "tensor bytes");
    expect(im_gguf_set_residency(&gguf, 1024 * 1024, 1024 * 1024) == 0, "set residency");
    expect(im_gguf_pin_tensor(&gguf, tensor) == 0, "pin tensor");
    expect(im_gguf_residency_entries(&gguf) == 1, "residency entry");
    expect(im_gguf_residency_used(&gguf) == 16, "residency used");
    float data[4] = {0};
    expect(im_gguf_read_tensor_data(&gguf, tensor, data, sizeof(data)) == 0, "read tensor");
    expect(fabsf(data[3] - 4.0f) < 1e-6f, "tensor data");

    im_gguf_free(&gguf);
    remove(path);
    puts("native gguf tests passed");
    return 0;
}
