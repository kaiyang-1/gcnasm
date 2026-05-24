// Unified host driver: launches both 256x256 and 192x256 BF16 GEMM kernels
// against the same input matrices, validates each against the CPU reference,
// and benchmarks each.
#include <opus/hip_minimal.hpp>
#include <random>
#include <iostream>
#include <memory>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <omp.h>

#include "gemm_defs.h"

// Device-stub declarations resolved by linking against the per-kernel TUs.
template<typename Traits>
__global__ void gemm_a16w16_256x256_kernel(opus_gemm_kargs kargs);
template<typename Traits>
__global__ void gemm_a16w16_192x256_kernel(opus_gemm_kargs kargs);

// Compile-time dispatch by tile size: select kernel from Traits::B_M.
template<typename Traits>
inline void gemm_launch(const opus_gemm_kargs& kargs, dim3 grid, dim3 block) {
    if constexpr (Traits::B_M == 256) {
        gemm_a16w16_256x256_kernel<Traits><<<grid, block>>>(kargs);
    } else {
        gemm_a16w16_192x256_kernel<Traits><<<grid, block>>>(kargs);
    }
}

#define CHECK_HIP(call)                                                                                   \
    do {                                                                                                  \
        hipError_t status_ = call;                                                                        \
        if (status_ != hipSuccess) {                                                                      \
            fprintf(stderr, "HIP error (%s:%d): %s\n", __FILE__, __LINE__, hipGetErrorString(status_));   \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while(0)

#define CHECK_HIP_KERNEL_LAUNCH() CHECK_HIP(hipGetLastError())

template<typename T>
void rand_vector_2d(T* ptr, int m, int n, int ld, float min_val = 0.0f, float max_val = 1.0f) {
    #pragma omp parallel
    {
        std::random_device rd;
        std::mt19937 gen(rd() + omp_get_thread_num());
        std::uniform_real_distribution<float> dis(min_val, max_val);
        #pragma omp for collapse(2)
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                ptr[i * ld + j] = static_cast<T>(dis(gen));
            }
        }
    }
}

template<typename T>
bool valid_vector(const T* ref, const T* result, int n, float threshold = 1e-3f) {
    int errors = 0;
    for(int i = 0; i < n; i++) {
        float diff = std::abs(static_cast<float>(ref[i]) - static_cast<float>(result[i]));
        if(diff > threshold) {
            if(errors < 10) {
                printf("Error at %d: ref=%.6f, result=%.6f, diff=%.6f\n",
                       i, static_cast<float>(ref[i]), static_cast<float>(result[i]), diff);
            }
            errors++;
            if(errors >= 10) break;
        }
    }
    return errors == 0;
}

// CPU reference GEMM (row-major, B is K-major like the device input).
void gemm_ref(const bf16_t* a, const bf16_t* b, bf16_t* c, int m, int n, int k, int lda, int ldb, int ldc) {
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            float sum = 0.0f;
            for(int p = 0; p < k; p++) {
                sum += static_cast<float>(a[i * lda + p]) * static_cast<float>(b[j * ldb + p]);
            }
            c[i * ldc + j] = static_cast<bf16_t>(sum);
        }
    }
}

template<typename Traits>
void benchmark_kernel(const opus_gemm_kargs& kargs, dim3 grid, dim3 block,
                      int warmup = 50, int iterations = 100) {
    for (int i = 0; i < warmup; ++i) {
        gemm_launch<Traits>(kargs, grid, block);
        CHECK_HIP_KERNEL_LAUNCH();
    }

    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipEventRecord(start));

    for (int i = 0; i < iterations; ++i) {
        gemm_launch<Traits>(kargs, grid, block);
        CHECK_HIP_KERNEL_LAUNCH();
    }

    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));

    float total_time = 0;
    CHECK_HIP(hipEventElapsedTime(&total_time, start, stop));

    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));

    const float avg_time = total_time / iterations;
    const std::size_t flop = std::size_t(2) * kargs.m * kargs.n * kargs.k * kargs.batch;
    const float tflops = static_cast<float>(flop) / 1.0e9f / avg_time;

    printf("Kernel Performance: avg_time=%.4f ms, %.2f TFlops\n", avg_time, tflops);
}

int main(int argc, char** argv) {
    // Default problem sizes
    int M = 256;
    int N = 512;
    int K = 128;
    int batch = 8;

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if ((std::strcmp(arg, "-m") == 0 || std::strcmp(arg, "--m") == 0) && i + 1 < argc) {
            M = std::atoi(argv[++i]);
        } else if ((std::strcmp(arg, "-n") == 0 || std::strcmp(arg, "--n") == 0) && i + 1 < argc) {
            N = std::atoi(argv[++i]);
        } else if ((std::strcmp(arg, "-k") == 0 || std::strcmp(arg, "--k") == 0) && i + 1 < argc) {
            K = std::atoi(argv[++i]);
        } else if ((std::strcmp(arg, "-b") == 0 || std::strcmp(arg, "--b") == 0) && i + 1 < argc) {
            batch = std::atoi(argv[++i]);
        }
    }

    if (M <= 0 || N <= 0 || K <= 0 || batch <= 0) {
        std::cerr << "Invalid problem size: M,N,K and batch must be positive.\n";
        return 1;
    }

    printf("GEMM: M=%d, N=%d, K=%d, batch=%d\n", M, N, K, batch);

    // Allocate host buffers (one shared A/B and one CPU reference; per-kernel C-out).
    auto host_a       = std::make_unique<bf16_t[]>(batch * M * K);
    auto host_b       = std::make_unique<bf16_t[]>(batch * N * K);
    auto host_c_ref   = std::make_unique<bf16_t[]>(batch * M * N);
    auto host_c_256   = std::make_unique<bf16_t[]>(batch * M * N);
    auto host_c_192   = std::make_unique<bf16_t[]>(batch * M * N);

    for(int b = 0; b < batch; b++) {
        rand_vector_2d(host_a.get() + b * M * K, M, K, K, 0.0f, 1.0f);
        rand_vector_2d(host_b.get() + b * N * K, N, K, K, -0.5f, 0.5f);
    }

    bf16_t *dev_a, *dev_b, *dev_c;
    CHECK_HIP(hipMalloc(&dev_a, batch * M * K * sizeof(bf16_t)));
    CHECK_HIP(hipMalloc(&dev_b, batch * N * K * sizeof(bf16_t)));
    CHECK_HIP(hipMalloc(&dev_c, batch * M * N * sizeof(bf16_t)));

    CHECK_HIP(hipMemcpy(dev_a, host_a.get(), batch * M * K * sizeof(bf16_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_b, host_b.get(), batch * N * K * sizeof(bf16_t), hipMemcpyHostToDevice));

    opus_gemm_kargs kargs{};
    kargs.ptr_a = dev_a;
    kargs.ptr_b = dev_b;
    kargs.ptr_c = dev_c;
    kargs.m = M;
    kargs.n = N;
    kargs.k = K;
    kargs.batch = batch;
    kargs.stride_a = K;
    kargs.stride_b = K;
    kargs.stride_c = N;
    kargs.stride_a_batch = M * K;
    kargs.stride_b_batch = N * K;
    kargs.stride_c_batch = M * N;

    // Compute CPU reference once; both kernels are validated against it.
    printf("\nComputing CPU reference...\n");
    for(int b = 0; b < batch; b++) {
        gemm_ref(
            host_a.get() + b * M * K,
            host_b.get() + b * N * K,
            host_c_ref.get() + b * M * N,
            M, N, K, K, K, N);
    }

    // Run one kernel variant: launch -> D->H -> validate per-batch -> benchmark.
    auto run = [&]<typename Traits>(Traits, bf16_t* host_c_out, const char* label) -> bool {
        const int num_tiles_m = ceil_div(M, Traits::B_M);
        const int num_tiles_n = ceil_div(N, Traits::B_N);
        dim3 grid(num_tiles_m * num_tiles_n, 1, batch);
        dim3 block(Traits::BLOCK_SIZE);

        printf("\n=== %s ===\n", label);
        printf("Launch: block_tile=%dx%dx%d, grid=(%d,%d,%d), block=%d\n",
               Traits::B_M, Traits::B_N, Traits::B_K,
               grid.x, grid.y, grid.z, (int)block.x);

        gemm_launch<Traits>(kargs, grid, block);
        CHECK_HIP_KERNEL_LAUNCH();

        CHECK_HIP(hipMemcpy(host_c_out, dev_c, batch * M * N * sizeof(bf16_t), hipMemcpyDeviceToHost));

        bool all_valid = true;
        for(int b = 0; b < batch; b++) {
            bool valid = valid_vector(
                host_c_ref.get() + b * M * N,
                host_c_out + b * M * N,
                M * N, 5e-1f);
            printf("[%s batch %d/%d: %dx%dx%d, block_%dx%dx%d] %s\n",
                   label, b + 1, batch, M, N, K,
                   Traits::B_M, Traits::B_N, Traits::B_K,
                   valid ? "✓ VALID" : "✗ FAIL");
            all_valid = all_valid && valid;
        }
        printf("[%s overall] %s\n", label,
               all_valid ? "✓ ALL BATCHES VALID" : "✗ SOME BATCHES FAILED");

        benchmark_kernel<Traits>(kargs, grid, block);
        return all_valid;
    };

    using Traits256 = opus_gemm_traits<512, 256, 256, 64, bf16_t, bf16_t, bf16_t, float, 8, 8, 4>;
    using Traits192 = opus_gemm_traits<512, 192, 256, 64, bf16_t, bf16_t, bf16_t, float, 8, 8, 4>;

    bool v1 = run(Traits256{}, host_c_256.get(), "256x256");
    bool v2 = run(Traits192{}, host_c_192.get(), "192x256");

    printf("\n=== Summary ===\n");
    printf("  256x256: %s\n", v1 ? "PASS" : "FAIL");
    printf("  192x256: %s\n", v2 ? "PASS" : "FAIL");

    CHECK_HIP(hipFree(dev_a));
    CHECK_HIP(hipFree(dev_b));
    CHECK_HIP(hipFree(dev_c));

    return (v1 && v2) ? 0 : 1;
}
