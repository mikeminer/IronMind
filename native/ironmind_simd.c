#include "ironmind_simd.h"

#include <stdint.h>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

#if !defined(_MSC_VER) && (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#endif

float im_dot_f32_avx2_impl(const float * a, const float * b, size_t n);
float im_dot_f32_avx512_impl(const float * a, const float * b, size_t n);

static float dot_f32_scalar(const float * a, const float * b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) sum += (double)a[i] * (double)b[i];
    return (float)sum;
}

static int cpuid_leaf(uint32_t leaf, uint32_t subleaf, uint32_t regs[4]) {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int out[4];
    __cpuidex(out, (int)leaf, (int)subleaf);
    regs[0] = (uint32_t)out[0];
    regs[1] = (uint32_t)out[1];
    regs[2] = (uint32_t)out[2];
    regs[3] = (uint32_t)out[3];
    return 1;
#elif !defined(_MSC_VER) && (defined(__x86_64__) || defined(__i386__))
    unsigned int a = 0, b = 0, c = 0, d = 0;
    if (!__get_cpuid_count(leaf, subleaf, &a, &b, &c, &d)) return 0;
    regs[0] = a;
    regs[1] = b;
    regs[2] = c;
    regs[3] = d;
    return 1;
#else
    (void)leaf;
    (void)subleaf;
    (void)regs;
    return 0;
#endif
}

static uint64_t xgetbv0(void) {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    return _xgetbv(0);
#elif !defined(_MSC_VER) && (defined(__x86_64__) || defined(__i386__))
    uint32_t eax = 0, edx = 0;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return ((uint64_t)edx << 32) | eax;
#else
    return 0;
#endif
}

static im_simd_backend detect_backend(void) {
    uint32_t regs[4] = {0, 0, 0, 0};
    if (!cpuid_leaf(1, 0, regs)) return IM_SIMD_SCALAR;
    const int osxsave = (regs[2] & (1u << 27)) != 0;
    const int avx = (regs[2] & (1u << 28)) != 0;
    if (!osxsave || !avx) return IM_SIMD_SCALAR;
    const uint64_t xcr0 = xgetbv0();
    if ((xcr0 & 0x6u) != 0x6u) return IM_SIMD_SCALAR;

    if (!cpuid_leaf(7, 0, regs)) return IM_SIMD_SCALAR;
    const int avx2 = (regs[1] & (1u << 5)) != 0;
    const int avx512f = (regs[1] & (1u << 16)) != 0;
    const int os_avx512 = (xcr0 & 0xe6u) == 0xe6u;
    if (avx512f && os_avx512) return IM_SIMD_AVX512F;
    if (avx2) return IM_SIMD_AVX2;
    return IM_SIMD_SCALAR;
}

im_simd_backend im_simd_selected_backend(void) {
    static int initialized = 0;
    static im_simd_backend backend = IM_SIMD_SCALAR;
    if (!initialized) {
        backend = detect_backend();
        initialized = 1;
    }
    return backend;
}

const char * im_simd_backend_name(im_simd_backend backend) {
    switch (backend) {
        case IM_SIMD_AVX512F: return "avx512f";
        case IM_SIMD_AVX2: return "avx2";
        default: return "scalar";
    }
}

float im_dot_f32(const float * a, const float * b, size_t n) {
    const im_simd_backend backend = im_simd_selected_backend();
    if (backend == IM_SIMD_AVX512F) return im_dot_f32_avx512_impl(a, b, n);
    if (backend == IM_SIMD_AVX2) return im_dot_f32_avx2_impl(a, b, n);
    return dot_f32_scalar(a, b, n);
}
