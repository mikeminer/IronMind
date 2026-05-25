#include <immintrin.h>
#include <stddef.h>

float im_dot_f32_avx512_impl(const float * a, const float * b, size_t n) {
    __m512 acc = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m512 va = _mm512_loadu_ps(a + i);
        const __m512 vb = _mm512_loadu_ps(b + i);
        acc = _mm512_add_ps(acc, _mm512_mul_ps(va, vb));
    }
    float lanes[16];
    _mm512_storeu_ps(lanes, acc);
    float sum = 0.0f;
    for (size_t lane = 0; lane < 16; lane++) sum += lanes[lane];
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}
