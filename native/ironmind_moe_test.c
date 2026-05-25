#include "ironmind_moe.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void expect(int condition, const char * message) {
    if (!condition) {
        fprintf(stderr, "native moe test failed: %s\n", message);
        exit(1);
    }
}

int main(void) {
    const float logits[4] = {0.0f, 3.0f, 1.0f, 2.0f};
    im_moe_route routes[2];
    expect(im_moe_topk(routes, 2, logits, 4) == 0, "topk");
    expect(routes[0].expert == 1, "best expert");
    expect(routes[1].expert == 3, "second expert");
    expect(fabsf((routes[0].weight + routes[1].weight) - 1.0f) < 1e-6f, "renormalized weights");
    expect(routes[0].weight > routes[1].weight, "softmax order");

    const float expert_outputs[8] = {
        0.0f, 0.0f,
        10.0f, 20.0f,
        0.0f, 0.0f,
        30.0f, 40.0f,
    };
    float out[2];
    expect(im_moe_mix(out, 2, routes, 2, expert_outputs, 4) == 0, "mix");
    expect(out[0] > 10.0f && out[0] < 30.0f, "mixed first");
    expect(out[1] > 20.0f && out[1] < 40.0f, "mixed second");

    puts("native moe tests passed");
    return 0;
}
