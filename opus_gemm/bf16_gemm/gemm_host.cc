// Unified host driver: launches both the quad-subtile and mono-tile BF16 GEMM
// kernels against the same input matrices, validates each against the CPU
// reference, and benchmarks each.
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
__global__ void gemm_a16w16_quad_subtile_kernel(opus_gemm_kargs kargs);
template<typename Traits>
__global__ void gemm_a16w16_mono_tile_kernel(opus_gemm_kargs kargs);

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

struct bench_result { float avg_ms; float tflops; };

template<typename Launch>
bench_result benchmark_kernel(Launch&& launch, const opus_gemm_kargs& kargs,
                              int warmup = 50, int iterations = 100) {
    for (int i = 0; i < warmup; ++i) {
        launch();
        CHECK_HIP_KERNEL_LAUNCH();
    }

    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipEventRecord(start));

    for (int i = 0; i < iterations; ++i) {
        launch();
        CHECK_HIP_KERNEL_LAUNCH();
    }

    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));

    float total_time = 0;
    CHECK_HIP(hipEventElapsedTime(&total_time, start, stop));

    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));

    const float avg_ms = total_time / iterations;
    const std::size_t flop = std::size_t(2) * kargs.m * kargs.n * kargs.k * kargs.batch;
    const float tflops = static_cast<float>(flop) / 1.0e9f / avg_ms;
    return {avg_ms, tflops};
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

    printf("BF16 GEMM: M=%d, N=%d, K=%d, Batch=%d\n", M, N, K, batch);

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

    printf("Computing CPU reference ...\n");
    for(int b = 0; b < batch; b++) {
        gemm_ref(
            host_a.get() + b * M * K,
            host_b.get() + b * N * K,
            host_c_ref.get() + b * M * N,
            M, N, K, K, K, N);
    }

    struct kernel_result { const char* label; bool ok; int passed; int total; float ms; float tflops; };

    // Run one (Traits, kernel) pair: launch -> D->H -> validate per-batch -> benchmark.
    auto run = [&]<typename Traits>(Traits, auto kernel, bf16_t* host_c_out, const char* label) -> kernel_result {
        const int num_tiles_m = ceil_div(M, Traits::B_M);
        const int num_tiles_n = ceil_div(N, Traits::B_N);
        dim3 grid(num_tiles_m * num_tiles_n, 1, batch);
        dim3 block(Traits::BLOCK_SIZE);

        auto launch = [&] { kernel<<<grid, block>>>(kargs); };

        printf("\n[%s]  tile=%dx%dx%d  grid=%dx%dx%d  block=%d\n",
               label, Traits::B_M, Traits::B_N, Traits::B_K,
               grid.x, grid.y, grid.z, (int)block.x);

        launch();
        CHECK_HIP_KERNEL_LAUNCH();
        CHECK_HIP(hipMemcpy(host_c_out, dev_c, batch * M * N * sizeof(bf16_t), hipMemcpyDeviceToHost));

        int passed = 0;
        for(int b = 0; b < batch; b++) {
            if (valid_vector(host_c_ref.get() + b * M * N, host_c_out + b * M * N, M * N, 5e-1f)) {
                ++passed;
            }
        }
        const bool ok = (passed == batch);
        printf("  validate : %s  (%d/%d batches)\n",
               ok ? "PASS" : "FAIL", passed, batch);

        auto bench = benchmark_kernel(launch, kargs);
        printf("  perf    : %7.4f ms  %8.2f TFlops\n", bench.avg_ms, bench.tflops);

        return {label, ok, passed, batch, bench.avg_ms, bench.tflops};
    };

    using TraitsQuad = opus_gemm_traits<512, 256, 256, 64, bf16_t, bf16_t, bf16_t, float>;
    using TraitsMono = opus_gemm_traits<512, 192, 256, 64, bf16_t, bf16_t, bf16_t, float>;

    auto r_quad = run(TraitsQuad{}, gemm_a16w16_quad_subtile_kernel<TraitsQuad>,
                      host_c_256.get(), "quad_subtile 256x256");
    auto r_mono = run(TraitsMono{}, gemm_a16w16_mono_tile_kernel<TraitsMono>,
                      host_c_192.get(), "mono_tile    192x256");

    printf("\nSummary\n");
    for (const auto& r : {r_quad, r_mono}) {
        printf("  %-22s  %s  %8.2f TFlops\n",
               r.label, r.ok ? "PASS" : "FAIL", r.tflops);
    }

    CHECK_HIP(hipFree(dev_a));
    CHECK_HIP(hipFree(dev_b));
    CHECK_HIP(hipFree(dev_c));

    return 0;
}
