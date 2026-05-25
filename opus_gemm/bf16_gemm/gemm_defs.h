// Shared types between host and GEMM kernel TUs.
// Intentionally has NO dependency on <opus/opus.hpp> so the host TU can
// instantiate opus_gemm_traits without pulling in opus containers.
// Per-kernel derived constants (HALF_B_M, E_M, smem_m_rep, etc.) live
// inside each kernel template .hpp.
#pragma once

#include <type_traits>
#include <cstddef>

using bf16_t = __bf16;

__host__ __device__ constexpr inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

// Kernel arguments shared by all GEMM kernel variants.
struct opus_gemm_kargs {
    const void* __restrict__ ptr_a;
    const void* __restrict__ ptr_b;
    void* __restrict__ ptr_c;
    int m;
    int n;
    int k;
    int batch;
    int stride_a;
    int stride_b;
    int stride_c;
    int stride_a_batch;
    int stride_b_batch;
    int stride_c_batch;
};

// User-facing GEMM configuration: block tile (B_M, B_N, B_K), data types,
// global-memory vector widths, and workgroup size. Plain int / type template
// parameters — no opus dependency. Each kernel template hpp consumes one of
// these and computes its own derived constants (HALF_B_M, E_M, smem_*, ...).
template<int BLOCK_SIZE_,
         int B_M_, int B_N_, int B_K_,
         typename D_A_, typename D_B_, typename D_C_, typename D_ACC_>
struct opus_gemm_traits {
    static constexpr int BLOCK_SIZE = BLOCK_SIZE_;

    static constexpr int B_M = B_M_;
    static constexpr int B_N = B_N_;
    static constexpr int B_K = B_K_;

    using D_A   = D_A_;
    using D_B   = D_B_;
    using D_C   = D_C_;
    using D_ACC = D_ACC_;
    static_assert(std::is_same<D_A, D_B>::value);

    static constexpr int VEC_A = 8;
    static constexpr int VEC_B = 8;
    static constexpr int VEC_C = 8;
};
