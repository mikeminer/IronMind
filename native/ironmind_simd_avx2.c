#include <immintrin.h>
#include <stddef.h>

float im_dot_f32_avx2_impl(const float * a, const float * b, size_t n) {
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
    }
    float lanes[8];
    _mm256_storeu_ps(lanes, acc);
    float sum = lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5] + lanes[6] + lanes[7];
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}
