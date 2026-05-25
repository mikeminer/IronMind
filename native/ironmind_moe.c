#include "ironmind_moe.h"

#include <float.h>
#include <math.h>
#include <string.h>

int im_moe_topk(im_moe_route * routes, size_t top_k, const float * logits, size_t n_expert) {
    if (!routes || !logits || top_k == 0 || n_expert == 0 || top_k > n_expert) return -1;
    for (size_t k = 0; k < top_k; k++) {
        routes[k].expert = 0;
        routes[k].weight = -FLT_MAX;
    }

    for (size_t expert = 0; expert < n_expert; expert++) {
        const float score = logits[expert];
        for (size_t k = 0; k < top_k; k++) {
            if (score > routes[k].weight) {
                for (size_t s = top_k - 1; s > k; s--) routes[s] = routes[s - 1];
                routes[k].expert = (uint32_t)expert;
                routes[k].weight = score;
                break;
            }
        }
    }

    float max = routes[0].weight;
    double sum = 0.0;
    for (size_t k = 0; k < top_k; k++) {
        routes[k].weight = expf(routes[k].weight - max);
        sum += routes[k].weight;
    }
    if (sum == 0.0) return -1;
    for (size_t k = 0; k < top_k; k++) routes[k].weight = (float)((double)routes[k].weight / sum);
    return 0;
}

int im_moe_mix(float * out, size_t n_embd, const im_moe_route * routes, size_t top_k, const float * expert_outputs, size_t n_expert) {
    if (!out || !routes || !expert_outputs || n_embd == 0 || top_k == 0 || n_expert == 0) return -1;
    memset(out, 0, n_embd * sizeof(float));
    for (size_t k = 0; k < top_k; k++) {
        if (routes[k].expert >= n_expert) return -1;
        const float * src = expert_outputs + (size_t)routes[k].expert * n_embd;
        const float w = routes[k].weight;
        for (size_t i = 0; i < n_embd; i++) out[i] += w * src[i];
    }
    return 0;
}
