#pragma once

#include <stddef.h>
#include <stdint.h>

typedef struct im_moe_route {
    uint32_t expert;
    float weight;
} im_moe_route;

int im_moe_topk(im_moe_route * routes, size_t top_k, const float * logits, size_t n_expert);
int im_moe_mix(float * out, size_t n_embd, const im_moe_route * routes, size_t top_k, const float * expert_outputs, size_t n_expert);
