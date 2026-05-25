#include "ironmind_simd.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void expect(int condition, const char * message) {
    if (!condition) {
        fprintf(stderr, "native simd test failed: %s\n", message);
        exit(1);
    }
}

int main(void) {
    float a[65];
    float b[65];
    double expected = 0.0;
    for (size_t i = 0; i < 65; i++) {
        a[i] = (float)((int)i - 20) * 0.25f;
        b[i] = (float)((int)(i % 7) - 3) * 0.5f;
        expected += (double)a[i] * (double)b[i];
    }
    const float got = im_dot_f32(a, b, 65);
    expect(fabsf(got - (float)expected) < 1e-4f, "dot result");
    printf("native simd tests passed (%s)\n", im_simd_backend_name(im_simd_selected_backend()));
    return 0;
}
