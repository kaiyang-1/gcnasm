// Host-only: benchmark harness, CPU reference, main()
#include <opus/hip_minimal.hpp>
#include <random>
#include <iostream>
#include <numeric>
#include <memory>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <omp.h>

#include "pa_defs.h"

// Declared in pa_prefill_kernel.cc (device TUs)
template<class Traits>
__global__ void pa_prefill_kernel(pa_kargs kargs);

// Common launch wrapper
template<class Traits>
inline void pa_launch(const pa_kargs& kargs, dim3 grid, dim3 block) {
    pa_prefill_kernel<Traits><<<grid, block>>>(kargs);
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

// Fill a contiguous vector with random values
template<typename T>
void rand_vector(T* ptr, size_t size, float min_val = 0.0f, float max_val = 1.0f) {
    #pragma omp parallel
    {
        std::random_device rd;
        std::mt19937 gen(rd() + omp_get_thread_num());
        std::uniform_real_distribution<float> dis(min_val, max_val);
        #pragma omp for
        for (size_t i = 0; i < size; i++) {
            ptr[i] = static_cast<T>(dis(gen));
        }
    }
}

// Benchmark GQA kernel performance with warm-up and timing
template<class Traits>
void benchmark_pa_kernel(const pa_kargs& kargs, dim3 grid, dim3 block,
                          int warmup = 100, int iterations = 50) {
    for (int i = 0; i < warmup; ++i) {
        pa_launch<Traits>(kargs, grid, block);
        CHECK_HIP_KERNEL_LAUNCH();
    }
    CHECK_HIP(hipDeviceSynchronize());

    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    CHECK_HIP(hipEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        pa_launch<Traits>(kargs, grid, block);
        CHECK_HIP_KERNEL_LAUNCH();
    }
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));

    float total_time = 0;
    CHECK_HIP(hipEventElapsedTime(&total_time, start, stop));

    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));

    const float avg_time = total_time / iterations;
    //   full attention  -> 4 * H * N^2 * D
    const double flops = (4.0 * kargs.H * kargs.N * kargs.N * kargs.D);
    const double tflops = flops / (avg_time * 1e-3) / 1e12;

    printf("PA Prefill Kernel Performance: avg_time=%.3f ms, %.2f TFlops\n",
           avg_time, tflops);
}

// Validate PA GPU results against CPU reference
bool validate_pa_results(const bf16_t* ref, const bf16_t* gpu,
                          int N, int H, int D, float threshold = 5e-2f) {
    bool all_valid = true;
    size_t total_errors = 0;
    const size_t total_elements = (size_t)N * H * D;

    for (int i = 0; i < N; i++) {
        for (int h = 0; h < H; h++) {
            const size_t offset = ((size_t)i * H + h) * D;
            for (int d = 0; d < D; d++) {
                const float ref_val = static_cast<float>(ref[offset + d]);
                const float gpu_val = static_cast<float>(gpu[offset + d]);
                const float diff = std::abs(gpu_val - ref_val);
                if (diff > threshold) {
                    total_errors++;
                    all_valid = false;
                    printf("  mismatch [n=%d,h=%d,d=%d] ref=%.6f gpu=%.6f diff=%.6f\n",
                           i, h, d, ref_val, gpu_val, diff);
                }
            }
        }
    }
    
    if (all_valid) {
        printf("✓ Full validation passed (checked %zu elements)\n", total_elements);
    } else {
        printf("✗ Validation failed with %zu/%zu total errors\n",
               total_errors, total_elements);
    }
    
    return all_valid;
}

// ─── CPU reference: Grouped-Query Attention (GQA) ──────────────────────────
//
// Q  layout: [B, N, H,    D]   (row-major, contiguous in D)
// K  layout: [B, N, H_KV, D]
// V  layout: [B, N, H_KV, D]
// O  layout: [B, N, H,    D]
//
// Standard scaled-dot-product attention with online softmax:
//   S[i,j]  = sum_d Q[b,i,h,d] * K[b,j,h_kv,d]   (h_kv = h / group_size)
//   P[i,:]  = softmax( S[i,:] / sqrt(D) )
//   O[i,d]  = sum_j P[i,j] * V[b,j,h_kv,d]
//
void pa_attention_ref(
    const bf16_t* Q,  // [N, H, D]
    const bf16_t* K,  // [N, H_KV, D]
    const bf16_t* V,  // [N, H_KV, D]
    bf16_t*       O,  // [N, H, D]
    int N, int H, int H_KV, int D)
{
    const int GROUP_SIZE = H / H_KV;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // Strides (row-major, last dim = D is contiguous)
    const int stride_q_n = H * D;
    const int stride_q_h = D;

    const int stride_kv_n = H_KV * D;
    const int stride_kv_h = D;

    #pragma omp parallel for collapse(2)
    for (int h = 0; h < H; h++) {
        for (int i = 0; i < N; i++) {
            const int h_kv = h / GROUP_SIZE;
            const bf16_t* q_row = Q + i * stride_q_n + h * stride_q_h;

            // ---- Compute attention scores S[j] = Q[i,h,:] . K[j,h_kv,:] ----
            const int max_j = N;
            std::vector<float> scores(max_j);
            for (int j = 0; j < max_j; j++) {
                const bf16_t* k_row = K + j * stride_kv_n + h_kv * stride_kv_h;
                float dot = 0.0f;
                for (int d = 0; d < D; d++) {
                    dot += static_cast<float>(q_row[d] * k_row[d]);
                }
                scores[j] = dot * scale;
            }

            // ---- Softmax ----
            float max_score = *std::max_element(scores.begin(), scores.end());
            float sum_exp = 0.0f;
            for (int j = 0; j < max_j; j++) {
                scores[j] = std::exp(scores[j] - max_score);
                sum_exp += scores[j];
            }
            for (int j = 0; j < max_j; j++) {
                scores[j] /= sum_exp;
            }
            std::vector<bf16_t> p_row(max_j);
            for (int j = 0; j < max_j; j++) {
                p_row[j] = static_cast<bf16_t>(scores[j]);
            }

            // ---- Output: O[i,h,d] = sum_j P[j] * V[j,h_kv,d] ----
            bf16_t* o_row = O + i * stride_q_n + h * stride_q_h;
            for (int d = 0; d < D; d++) {
                float acc = 0.0f;
                for (int j = 0; j < max_j; j++) {
                    const bf16_t* v_row = V + j * stride_kv_n + h_kv * stride_kv_h;
                    acc += static_cast<float>(p_row[j] * v_row[d]);
                }
                o_row[d] = static_cast<bf16_t>(acc);
            }
        }
    }
}

// ─── main ───────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    int H    = 128;   // query heads
    int H_KV = 1;     // key/value heads
    int N    = 1024;  // sequence length
    int D    = 512;   // head dimension

    // Parse command line arguments. Supports: -n 16384 and -n=16384.
    bool verify = false;
    auto parse_val = [](const char* arg, const char* flag) -> const char* {
        size_t len = std::strlen(flag);
        if (std::strncmp(arg, flag, len) == 0) {
            if (arg[len] == '=') return arg + len + 1;       // -flag=value
            if (arg[len] == '\0') return reinterpret_cast<const char*>(1); // -flag value (next arg)
        }
        return nullptr;
    };
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        const char* val;
        if (std::strcmp(arg, "--verify") == 0) { verify = true; continue; }
        auto try_parse = [&](int& target, const char* flag) {
            if ((val = parse_val(arg, flag))) {
                if (val == reinterpret_cast<const char*>(1)) { if (i + 1 < argc) target = std::atoi(argv[++i]); }
                else target = std::atoi(val);
                return true;
            }
            return false;
        };
        if (try_parse(H, "-h_q")) continue;
        if (try_parse(H_KV, "-h_kv")) continue;
        if (try_parse(N, "-n")) continue;
        if (try_parse(D, "-d")) continue;
    }

    if (H <= 0 || H_KV <= 0 || N <= 0 || D <= 0 || H % H_KV != 0) {
        std::cerr << "Invalid parameters. H_Q,H_KV,N,D must be positive and H_Q must be divisible by H_KV.\n";
        return 1;
    }

    const int GROUP_SIZE = H / H_KV;
    printf("PA Prefill Attention: H_Q=%d, H_KV=%d, GROUP_SIZE=%d, N=%d, D=%d\n",
           H, H_KV, GROUP_SIZE, N, D);

    // Allocate host memory
    const size_t q_size = (size_t)N * H * D;
    const size_t kv_size = (size_t)N * H_KV * D;
    auto host_q = std::make_unique<bf16_t[]>(q_size);
    auto host_k = std::make_unique<bf16_t[]>(kv_size);
    auto host_v = std::make_unique<bf16_t[]>(kv_size);
    auto host_o_ref = std::make_unique<bf16_t[]>(q_size);
    auto host_o_gpu = std::make_unique<bf16_t[]>(q_size);

    // Initialize with random data
    rand_vector(host_q.get(), q_size, -2.f, 2.f);
    rand_vector(host_k.get(), kv_size, -2.f, 2.f);
    rand_vector(host_v.get(), kv_size, -2.f, 2.f);

    // Allocate device memory
    bf16_t *dev_q, *dev_k, *dev_v, *dev_o;
    CHECK_HIP(hipMalloc(&dev_q, q_size * sizeof(bf16_t)));
    CHECK_HIP(hipMalloc(&dev_k, kv_size * sizeof(bf16_t)));
    CHECK_HIP(hipMalloc(&dev_v, kv_size * sizeof(bf16_t)));
    CHECK_HIP(hipMalloc(&dev_o, q_size * sizeof(bf16_t)));

    CHECK_HIP(hipMemcpy(dev_q, host_q.get(), q_size * sizeof(bf16_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_k, host_k.get(), kv_size * sizeof(bf16_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_v, host_v.get(), kv_size * sizeof(bf16_t), hipMemcpyHostToDevice));

    // Setup kernel arguments
    pa_kargs kargs{};
    kargs.ptr_q = dev_q;
    kargs.ptr_k = dev_k;
    kargs.ptr_v = dev_v;
    kargs.ptr_o = dev_o;
    kargs.N = N;
    kargs.H = H;
    kargs.H_KV = H_KV;
    kargs.D = D;
    kargs.stride_q_n = H * D;
    kargs.stride_q_h = D;
    kargs.stride_kv_n = H_KV * D;
    kargs.stride_kv_h = D;

    // Dispatch to kernel
    auto run = [&]<typename PATraits>(PATraits) {
        if (D != PATraits::D_TILE_SIZE) {
            std::cerr << "This kernel only supports head dimension D=" << PATraits::D_TILE_SIZE << ", got D=" << D << "\n";
            return 1;
        }
        if ((N % PATraits::KV_TILE_SIZE) != 0 || (N / PATraits::KV_TILE_SIZE) < 6) {
            std::cerr << "This attend-style pipeline requires N to be a multiple of "
                      << PATraits::KV_TILE_SIZE << " and span at least 6 KV tiles, got N=" << N << "\n";
            return 1;
        }
        if ((N % (PATraits::Q_TILE_SIZE * PATraits::NUM_WARPS)) != 0) {
            std::cerr << "This kernel requires N to be a multiple of "
                      << (PATraits::Q_TILE_SIZE * PATraits::NUM_WARPS)
                      << " so every warp maps to a valid Q tile, got N=" << N << "\n";
            return 1;
        }
        const int num_q_tiles = ceil_div(N, PATraits::Q_TILE_SIZE);
        const int num_q_blocks = ceil_div(num_q_tiles, PATraits::NUM_WARPS);
        dim3 grid(H, num_q_blocks, 1);
        dim3 block(PATraits::BLOCK_SIZE);

        printf("PA kernel launch config: grid=(%d,%d,%d), block=%d (NUM_WARPS=%d), smem=%zu bytes (K/V tiles)\n",
               grid.x, grid.y, grid.z, (int)block.x, PATraits::NUM_WARPS, PATraits::smem_size_bytes());

        pa_launch<PATraits>(kargs, grid, block);
        CHECK_HIP_KERNEL_LAUNCH();

        if (verify) {
            printf("\nValidating GPU results against CPU reference...\n");
            CHECK_HIP(hipMemcpy(host_o_gpu.get(), dev_o, q_size * sizeof(bf16_t), hipMemcpyDeviceToHost));
            pa_attention_ref(host_q.get(), host_k.get(), host_v.get(), host_o_ref.get(),
                              N, H, H_KV, D);

            bool all_valid = validate_pa_results(host_o_ref.get(), host_o_gpu.get(), N, H, D);
            printf("\n[Overall] %s\n", all_valid ? "✓ GPU KERNEL VALID" : "✗ GPU KERNEL FAILED");
            if (!all_valid) return 1;
        }

        printf("\n");
        benchmark_pa_kernel<PATraits>(kargs, grid, block);
        printf("\n");
        return 0;
    };

    int rc;
    if (D == 512) {
        rc = run(pa_traits<16, 32, 512, 8>{});
    } else {
        std::cerr << "-d must be 512, got " << D << "\n";
        return 1;
    }
    if (rc) return rc;

    // Cleanup
    CHECK_HIP(hipFree(dev_q));
    CHECK_HIP(hipFree(dev_k));
    CHECK_HIP(hipFree(dev_v));
    CHECK_HIP(hipFree(dev_o));

    return 0;
}
