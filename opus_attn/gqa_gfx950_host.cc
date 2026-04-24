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

#include "gqa_common.h"

// Declared in gqa_gfx950_kernel.cc (device TU)
template<class Traits>
__global__ void gqa_kernel(opus_gqa_kargs kargs);

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
void benchmark_gqa_kernel(const opus_gqa_kargs& kargs, dim3 grid, dim3 block,
                          int warmup = 100, int iterations = 50) {
    for (int i = 0; i < warmup; ++i) {
        gqa_kernel<Traits><<<grid, block>>>(kargs);
        CHECK_HIP_KERNEL_LAUNCH();
    }
    CHECK_HIP(hipDeviceSynchronize());

    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    CHECK_HIP(hipEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        gqa_kernel<Traits><<<grid, block>>>(kargs);
        CHECK_HIP_KERNEL_LAUNCH();
    }
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));

    float total_time = 0;
    CHECK_HIP(hipEventElapsedTime(&total_time, start, stop));

    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));

    const float avg_time = total_time / iterations;
    //   Q @ K^T -> 2 * B * H * N^2 * D
    //   P @ V   -> 2 * B * H * N^2 * D_V
    //   causal attention -> half of the full-attention work
    const double flops = (2.0 * kargs.B * kargs.H * kargs.N * kargs.N * (kargs.D + kargs.D_V))
                       / (Traits::CAUSAL ? 2.0 : 1.0);
    const double tflops = flops / (avg_time * 1e-3) / 1e12;

    printf("GQA %s Kernel Performance: avg_time=%.3f ms, %.2f TFlops\n",
           Traits::CAUSAL ? "Causal" : "Non-causal", avg_time, tflops);
}

// Validate GQA GPU results against CPU reference
bool validate_gqa_results(const bf16_t* ref, const bf16_t* gpu,
                          int B, int N, int H, int D_V, float threshold = 5e-2f) {
    bool all_valid = true;
    int total_errors = 0;

    // Sample-based validation (check a subset to avoid too much output)
    const int sample_heads = std::min(4, H);
    const int sample_queries = std::min(8, N);
    
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < sample_heads; h++) {
            for (int i = 0; i < sample_queries; i++) {
                int offset = b * N * H * D_V + i * H * D_V + h * D_V;
                
                // Check element-wise
                int local_errors = 0;
                float max_diff = 0.0f;
                for (int d = 0; d < D_V; d++) {
                    float ref_val = static_cast<float>(ref[offset + d]);
                    float gpu_val = static_cast<float>(gpu[offset + d]);
                    float diff = std::abs(ref_val - gpu_val);
                    max_diff = std::max(max_diff, diff);
                    if (diff > threshold) {
                        local_errors++;
                        total_errors++;
                    }
                }
                
                if (local_errors > 0) {
                    printf("  [b=%d,h=%d,n=%d] max_diff=%.6f, errors=%d/%d\n",
                           b, h, i, max_diff, local_errors, D_V);
                    all_valid = false;
                }
            }
        }
    }
    
    if (all_valid) {
        printf("✓ Sample validation passed (checked %d samples)\n", 
               B * sample_heads * sample_queries);
    } else {
        printf("✗ Validation failed with %d total errors\n", total_errors);
    }
    
    return all_valid;
}

// ─── CPU reference: Grouped-Query Attention (GQA) ──────────────────────────
//
// Q  layout: [B, N, H,    D]     (row-major, contiguous in D)
// K  layout: [B, N, H_KV, D]
// V  layout: [B, N, H_KV, D_V]
// O  layout: [B, N, H,    D_V]
//
// Standard scaled-dot-product attention with online softmax:
//   S[i,j]  = sum_d Q[b,i,h,d] * K[b,j,h_kv,d]   (h_kv = h / group_size)
//   P[i,:]  = softmax( S[i,:] / sqrt(D) )
//   O[i,d_v] = sum_j P[i,j] * V[b,j,h_kv,d_v]
//
void gqa_attention_ref(
    const bf16_t* Q,  // [B, N, H, D]
    const bf16_t* K,  // [B, N, H_KV, D]
    const bf16_t* V,  // [B, N, H_KV, D_V]
    bf16_t*       O,  // [B, N, H, D_V]
    int B, int N, int H, int H_KV, int D, int D_V, bool causal = false)
{
    const int GROUP_SIZE = H / H_KV;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // Strides (row-major, last dim is contiguous)
    const int stride_q_b = N * H * D;
    const int stride_q_n = H * D;
    const int stride_q_h = D;

    const int stride_k_b = N * H_KV * D;
    const int stride_k_n = H_KV * D;
    const int stride_k_h = D;

    const int stride_v_b = N * H_KV * D_V;
    const int stride_v_n = H_KV * D_V;
    const int stride_v_h = D_V;

    const int stride_o_b = N * H * D_V;
    const int stride_o_n = H * D_V;
    const int stride_o_h = D_V;

    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            for (int i = 0; i < N; i++) {
                const int h_kv = h / GROUP_SIZE;
                const bf16_t* q_row = Q + b * stride_q_b + i * stride_q_n + h * stride_q_h;

                // ---- Compute attention scores S[j] = Q[b,i,h,:] . K[b,j,h_kv,:] ----
                const int max_j = causal ? (i + 1) : N;
                std::vector<float> scores(max_j);
                for (int j = 0; j < max_j; j++) {
                    const bf16_t* k_row = K + b * stride_k_b + j * stride_k_n + h_kv * stride_k_h;
                    float dot = 0.0f;
                    for (int d = 0; d < D; d++) {
                        dot += static_cast<float>(q_row[d]) * static_cast<float>(k_row[d]);
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

                // ---- Output: O[b,i,h,d_v] = sum_j P[j] * V[b,j,h_kv,d_v] ----
                bf16_t* o_row = O + b * stride_o_b + i * stride_o_n + h * stride_o_h;
                for (int d = 0; d < D_V; d++) {
                    float acc = 0.0f;
                    for (int j = 0; j < max_j; j++) {
                        const bf16_t* v_row = V + b * stride_v_b + j * stride_v_n + h_kv * stride_v_h;
                        acc += scores[j] * static_cast<float>(v_row[d]);
                    }
                    o_row[d] = static_cast<bf16_t>(acc);
                }
            }
        }
    }
}

// ─── main ───────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    int B    = 16;    // batch size
    int H    = 64;    // query heads
    int H_KV = 8;     // key/value heads
    int N    = 1024;  // sequence length
    int D    = 128;   // Q/K head dimension
    int D_V  = D;     // V/O head dimension

    // Parse command line arguments. Supports: -n 16384, -n=16384, --seq=16384
    bool causal = true;
    int verify = 0;
    auto parse_val = [](const char* arg, const char* flag) -> const char* {
        size_t len = std::strlen(flag);
        if (std::strncmp(arg, flag, len) == 0) {
            if (arg[len] == '=') return arg + len + 1;       // --flag=value
            if (arg[len] == '\0') return reinterpret_cast<const char*>(1); // --flag value (next arg)
        }
        return nullptr;
    };
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        const char* val;
        if (std::strcmp(arg, "--causal") == 0) { causal = true; continue; }
        if (std::strcmp(arg, "--no-causal") == 0) { causal = false; continue; }
        auto try_parse = [&](int& target, const char* s, const char* l) {
            if ((val = parse_val(arg, s)) || (l && (val = parse_val(arg, l)))) {
                if (val == reinterpret_cast<const char*>(1)) { if (i + 1 < argc) target = std::atoi(argv[++i]); }
                else target = std::atoi(val);
                return true;
            }
            return false;
        };
        if (try_parse(B, "-b", "--batch")) continue;
        if (try_parse(H, "-h", "--heads")) continue;
        if (try_parse(H_KV, "--hkv", nullptr)) continue;
        if (try_parse(N, "-n", "--seq")) continue;
        if (try_parse(D, "-d", "--dim")) continue;
        if (try_parse(D_V, "--dim-v", "--vdim")) continue;
        if (try_parse(verify, "-v", "--verify")) continue;
    }

    if (B <= 0 || H <= 0 || H_KV <= 0 || N <= 0 || D <= 0 || D_V <= 0 || H % H_KV != 0) {
        std::cerr << "Invalid parameters. B,H,H_KV,N,D,D_V must be positive and H must be divisible by H_KV.\n";
        return 1;
    }

    const int GROUP_SIZE = H / H_KV;
    printf("GQA Attention: B=%d, H=%d, H_KV=%d, GROUP_SIZE=%d, N=%d, D=%d, D_V=%d, CAUSAL=%d\n",
           B, H, H_KV, GROUP_SIZE, N, D, D_V, causal ? 1 : 0);

    // Allocate host memory
    const size_t q_size = (size_t)B * N * H * D;
    const size_t k_size = (size_t)B * N * H_KV * D;
    const size_t v_size = (size_t)B * N * H_KV * D_V;
    const size_t o_size = (size_t)B * N * H * D_V;
    auto host_q = std::make_unique<bf16_t[]>(q_size);
    auto host_k = std::make_unique<bf16_t[]>(k_size);
    auto host_v = std::make_unique<bf16_t[]>(v_size);
    auto host_o_ref = std::make_unique<bf16_t[]>(o_size);
    auto host_o_gpu = std::make_unique<bf16_t[]>(o_size);

    // Initialize with random data
    rand_vector(host_q.get(), q_size, -2.f, 2.f);
    rand_vector(host_k.get(), k_size, -2.f, 2.f);
    rand_vector(host_v.get(), v_size, -2.f, 2.f);

    // Allocate device memory
    bf16_t *dev_q, *dev_k, *dev_v, *dev_o;
    CHECK_HIP(hipMalloc(&dev_q, q_size * sizeof(bf16_t)));
    CHECK_HIP(hipMalloc(&dev_k, k_size * sizeof(bf16_t)));
    CHECK_HIP(hipMalloc(&dev_v, v_size * sizeof(bf16_t)));
    CHECK_HIP(hipMalloc(&dev_o, o_size * sizeof(bf16_t)));

    CHECK_HIP(hipMemcpy(dev_q, host_q.get(), q_size * sizeof(bf16_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_k, host_k.get(), k_size * sizeof(bf16_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_v, host_v.get(), v_size * sizeof(bf16_t), hipMemcpyHostToDevice));

    // Setup kernel arguments
    opus_gqa_kargs kargs{};
    kargs.ptr_q = dev_q;
    kargs.ptr_k = dev_k;
    kargs.ptr_v = dev_v;
    kargs.ptr_o = dev_o;
    kargs.B = B;
    kargs.N = N;
    kargs.H = H;
    kargs.H_KV = H_KV;
    kargs.D = D;
    kargs.D_V = D_V;
    kargs.stride_q_b = N * H * D;
    kargs.stride_q_n = H * D;
    kargs.stride_q_h = D;
    kargs.stride_k_b = N * H_KV * D;
    kargs.stride_k_n = H_KV * D;
    kargs.stride_k_h = D;
    kargs.stride_v_b = N * H_KV * D_V;
    kargs.stride_v_n = H_KV * D_V;
    kargs.stride_v_h = D_V;
    kargs.stride_o_b = N * H * D_V;
    kargs.stride_o_n = H * D_V;
    kargs.stride_o_h = D_V;

    // Dispatch to causal or non-causal kernel
    auto run = [&]<typename GqaTraits>(GqaTraits) {
        if (D != GqaTraits::D_TILE_SIZE || D_V != GqaTraits::DV_TILE_SIZE) {
            std::cerr << "This kernel only supports Q/K head dim D=" << GqaTraits::D_TILE_SIZE
                      << " and V/O head dim D_V=" << GqaTraits::DV_TILE_SIZE
                      << ", got D=" << D << ", D_V=" << D_V << "\n";
            return 1;
        }
        if ((N % GqaTraits::KV_TILE_SIZE) != 0 || (N / GqaTraits::KV_TILE_SIZE) < 6) {
            std::cerr << "This attend-style pipeline requires N to be a multiple of "
                      << GqaTraits::KV_TILE_SIZE << " and span at least 6 KV tiles, got N=" << N << "\n";
            return 1;
        }
        if ((N % (GqaTraits::Q_TILE_SIZE * GqaTraits::NUM_WARPS)) != 0) {
            std::cerr << "This kernel requires N to be a multiple of "
                      << (GqaTraits::Q_TILE_SIZE * GqaTraits::NUM_WARPS)
                      << " so every warp maps to a valid Q tile, got N=" << N << "\n";
            return 1;
        }
        const int num_q_tiles = ceil_div(N, GqaTraits::Q_TILE_SIZE);
        const int num_q_blocks = ceil_div(num_q_tiles, GqaTraits::NUM_WARPS);
        dim3 grid(H, num_q_blocks, B);
        dim3 block(GqaTraits::BLOCK_SIZE);

        printf("GQA kernel launch config: grid=(%d,%d,%d), block=%d (NUM_WARPS=%d), smem=%zu bytes (K/V tiles)\n",
               grid.x, grid.y, grid.z, (int)block.x, GqaTraits::NUM_WARPS, GqaTraits::smem_size_bytes());

        gqa_kernel<GqaTraits><<<grid, block>>>(kargs);
        CHECK_HIP_KERNEL_LAUNCH();

        if (verify) {
            printf("\nValidating GPU results against CPU reference...\n");
            CHECK_HIP(hipMemcpy(host_o_gpu.get(), dev_o, o_size * sizeof(bf16_t), hipMemcpyDeviceToHost));
            gqa_attention_ref(host_q.get(), host_k.get(), host_v.get(), host_o_ref.get(),
                              B, N, H, H_KV, D, D_V, GqaTraits::CAUSAL);

            bool all_valid = validate_gqa_results(host_o_ref.get(), host_o_gpu.get(), B, N, H, D_V);
            printf("\n[Overall] %s\n", all_valid ? "✓ GPU KERNEL VALID" : "✗ GPU KERNEL FAILED");
            if (!all_valid) return 1;
        }

        printf("\n");
        benchmark_gqa_kernel<GqaTraits>(kargs, grid, block);
        printf("\n");
        return 0;
    };

    int rc;
    if (causal) {
        if (D == 128 && D_V == 128)
            rc = run(opus_gqa_traits<32, 64, 128, 128, 8, true>{});
        else if (D == 192 && D_V == 128)
            rc = run(opus_gqa_traits<32, 64, 192, 128, 8, true>{});
        else {
            std::cerr << "Unsupported causal kernel dims: D=" << D << ", D_V=" << D_V
                      << ". Supported pairs: (128,128), (192,128)\n";
            return 1;
        }
    } else {
        if (D == 128 && D_V == 128)
            rc = run(opus_gqa_traits<32, 64, 128, 128, 8, false>{});
        else if (D == 192 && D_V == 128)
            rc = run(opus_gqa_traits<32, 64, 192, 128, 8, false>{});
        else {
            std::cerr << "Unsupported non-causal kernel dims: D=" << D << ", D_V=" << D_V
                      << ". Supported pairs: (128,128), (192,128)\n";
            return 1;
        }
    }
    if (rc) return rc;

    // Cleanup
    CHECK_HIP(hipFree(dev_q));
    CHECK_HIP(hipFree(dev_k));
    CHECK_HIP(hipFree(dev_v));
    CHECK_HIP(hipFree(dev_o));

    return 0;
}
