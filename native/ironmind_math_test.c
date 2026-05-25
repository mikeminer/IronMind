#include "ironmind_math.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

static int closef(float a, float b, float eps) {
    return fabsf(a - b) < eps;
}

int main(void) {
    const float x[2] = {3.0f, 4.0f};
    const float w[2] = {1.0f, 1.0f};
    float y[2] = {0};
    im_rms_norm(y, x, w, 2, 0.0f);
    assert(closef(y[0], 0.8485281f, 1e-5f));
    assert(closef(y[1], 1.1313708f, 1e-5f));

    float rope[2] = {1.0f, 0.0f};
    im_apply_rope(rope, 2, 2, 3.14159265358979323846 / 2.0, 10000.0);
    assert(closef(rope[0], 0.0f, 1e-5f));
    assert(closef(rope[1], 1.0f, 1e-5f));

    float probs[3] = {1.0f, 2.0f, 3.0f};
    im_softmax(probs, 3);
    assert(closef(probs[0] + probs[1] + probs[2], 1.0f, 1e-5f));

    const float q[2] = {1.0f, 0.0f};
    const float k[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float v[4] = {10.0f, 0.0f, 0.0f, 10.0f};
    float out[2] = {0};
    im_attention(out, q, k, v, 2, 2, 2, 100.0f);
    assert(out[0] > 9.999f);
    assert(out[1] < 0.001f);

    puts("native math tests passed");
    return 0;
}

