// Host-only: benchmark harness, CPU reference, main()
#include <opus/hip_minimal.hpp>
#include <algorithm>
#include <random>
#include <iostream>
#include <limits>
#include <numeric>
#include <memory>
#include <vector>
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

void init_sparse_kv_indices(std::vector<int>& kv_indptr,
                            std::vector<int>& kv_indices,
                            int N,
                            int total_pages,
                            int kv_tile_size) {
    assert(N >= 0);
    assert(total_pages > 0);
    assert(kv_tile_size > 0);

    kv_indptr.assign(N + 1, 0);
    kv_indices.clear();

    std::mt19937 gen(1234);
    std::vector<int> pages(total_pages);
    std::iota(pages.begin(), pages.end(), 0);

    auto clamp_len = [&](int len) {
        return std::max(0, std::min(len, total_pages));
    };

    const std::vector<int> boundary_lengths = {
        0,
        1,
        kv_tile_size - 1,
        kv_tile_size,
        kv_tile_size + 1,
        2 * kv_tile_size,
        2 * kv_tile_size + 1,
        total_pages
    };
    std::uniform_int_distribution<int> random_len(0, total_pages);

    for (int q = 0; q < N; ++q) {
        int nnz = 0;
        if (q < static_cast<int>(boundary_lengths.size())) {
            nnz = clamp_len(boundary_lengths[q]);
        } else {
            nnz = random_len(gen);
        }

        std::shuffle(pages.begin(), pages.end(), gen);
        kv_indices.insert(kv_indices.end(), pages.begin(), pages.begin() + nnz);
        assert(kv_indices.size() <= static_cast<size_t>(std::numeric_limits<int>::max()));
        kv_indptr[q + 1] = static_cast<int>(kv_indices.size());
    }

    assert(kv_indptr.front() == 0);
    assert(kv_indptr.back() == static_cast<int>(kv_indices.size()));
    for (int q = 0; q < N; ++q) {
        assert(kv_indptr[q] <= kv_indptr[q + 1]);
        for (int p = kv_indptr[q]; p < kv_indptr[q + 1]; ++p) {
            assert(kv_indices[p] >= 0 && kv_indices[p] < total_pages);
        }
    }
}

void init_dense_kv_indices(std::vector<int>& kv_indptr,
                           std::vector<int>& kv_indices,
                           int N,
                           int total_pages) {
    assert(N >= 0);
    assert(total_pages > 0);
    const size_t total_indices = static_cast<size_t>(N) * total_pages;
    assert(total_indices <= static_cast<size_t>(std::numeric_limits<int>::max()));

    kv_indptr.resize(N + 1);
    kv_indices.resize(total_indices);

    for (int q = 0; q <= N; ++q) {
        kv_indptr[q] = static_cast<int>(static_cast<size_t>(q) * total_pages);
    }
    for (int q = 0; q < N; ++q) {
        const size_t row_begin = static_cast<size_t>(q) * total_pages;
        for (int page = 0; page < total_pages; ++page) {
            kv_indices[row_begin + page] = page;
        }
    }
}

// Benchmark PA kernel performance with warm-up and timing
template<class Traits>
void benchmark_pa_kernel(const pa_kargs& kargs, dim3 grid, dim3 block,
                          int indices_prefix_sum, int warmup = 100, int iterations = 50) {
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
    //   sparse attention -> 4 * H * nnz(indices) * D
    const double flops = (4.0 * kargs.H * indices_prefix_sum * kargs.D);
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

// ─── CPU reference: Paged Attention (PA) ──────────────────────────
//
// Sparse scaled-dot-product attention over CSR rows:
//   rows_i = kv_indices[kv_indptr[i] : kv_indptr[i+1]]
//   S[i,p] = sum_d Q[i,h,d] * K[rows_i[p],d]
//   O[i,h,d] = sum_p softmax(S[i,p] / sqrt(D)) * V[rows_i[p],d]
//
void pa_attention_ref(
    const bf16_t* Q,  // [N, H, D]
    const bf16_t* K,  // [total_pages, D]
    const bf16_t* V,  // [total_pages, D]
    bf16_t*       O,  // [N, H, D]
    const int* kv_indptr,
    const int* kv_indices,
    int N, int H, int D)
{
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // Strides (row-major, last dim = D is contiguous)
    const int stride_qo_n = H * D;
    const int stride_qo_h = D;
    const int stride_kv_page = D;

    #pragma omp parallel for collapse(2)
    for (int h = 0; h < H; h++) {
        for (int i = 0; i < N; i++) {
            const bf16_t* q_row = Q + i * stride_qo_n + h * stride_qo_h;
            const int row_begin = kv_indptr[i];
            const int row_end = kv_indptr[i + 1];
            const int num_rows = row_end - row_begin;

            if (num_rows <= 0) {
                bf16_t* o_row = O + i * stride_qo_n + h * stride_qo_h;
                for (int d = 0; d < D; d++) {
                    o_row[d] = static_cast<bf16_t>(0.0f);
                }
                continue;
            }

            // ---- Compute attention scores S[p] = Q[i,h,:] . K[kv_indices[p],:] ----
            std::vector<float> scores(num_rows);
            for (int p = 0; p < num_rows; p++) {
                const int kv_row = kv_indices[row_begin + p];
                const bf16_t* k_row = K + kv_row * stride_kv_page;
                float dot = 0.0f;
                for (int d = 0; d < D; d++) {
                    dot += static_cast<float>(q_row[d] * k_row[d]);
                }
                scores[p] = dot * scale;
            }

            // ---- Softmax ----
            float max_score = *std::max_element(scores.begin(), scores.end());
            float sum_exp = 0.0f;
            for (int p = 0; p < num_rows; p++) {
                scores[p] = std::exp(scores[p] - max_score);
                sum_exp += scores[p];
            }
            for (int p = 0; p < num_rows; p++) {
                scores[p] /= sum_exp;
            }
            std::vector<bf16_t> p_row(num_rows);
            for (int p = 0; p < num_rows; p++) {
                p_row[p] = static_cast<bf16_t>(scores[p]);
            }

            // ---- Output: O[i,h,d] = sum_p P[p] * V[kv_indices[p],d] ----
            bf16_t* o_row = O + i * stride_qo_n + h * stride_qo_h;
            for (int d = 0; d < D; d++) {
                float acc = 0.0f;
                for (int p = 0; p < num_rows; p++) {
                    const int kv_row = kv_indices[row_begin + p];
                    const bf16_t* v_row = V + kv_row * stride_kv_page;
                    acc += static_cast<float>(p_row[p] * v_row[d]);
                }
                o_row[d] = static_cast<bf16_t>(acc);
            }
        }
    }
}

// ─── main ───────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    int H = 128;   // query heads
    int N = 1024;  // sequence length
    int D = 512;   // head dimension
    int total_pages = N; // total number of pages for kv cache

    // Parse command line arguments. Supports: -n 16384 and -n=16384.
    bool verify = false;
    bool dense_kv = false;
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
        if (std::strcmp(arg, "--dense") == 0) { dense_kv = true; continue; }
        auto try_parse = [&](int& target, const char* flag) {
            if ((val = parse_val(arg, flag))) {
                if (val == reinterpret_cast<const char*>(1)) { if (i + 1 < argc) target = std::atoi(argv[++i]); }
                else target = std::atoi(val);
                return true;
            }
            return false;
        };
        if (try_parse(H, "-h_q")) continue;
        if (try_parse(N, "-n")) continue;
        if (try_parse(D, "-d")) continue;
        if (try_parse(total_pages, "-total_pages")) continue;
    }

    if (H <= 0 || N <= 0 || D <= 0 || total_pages <= 0) {
        std::cerr << "Invalid parameters. H_Q,N,D,total_pages must be positive.\n";
        return 1;
    }

    printf("PA Prefill Attention: H_Q=%d, N=%d, D=%d, total_pages=%d\n",
           H, N, D, total_pages);

    // Allocate host memory
    const size_t q_size = (size_t)N * H * D;
    const size_t kv_size = (size_t)total_pages * D;
    auto host_q = std::make_unique<bf16_t[]>(q_size);
    auto host_k = std::make_unique<bf16_t[]>(kv_size);
    auto host_v = std::make_unique<bf16_t[]>(kv_size);
    auto host_o_ref = std::make_unique<bf16_t[]>(q_size);
    auto host_o_gpu = std::make_unique<bf16_t[]>(q_size);
    std::vector<int> host_kv_indptr;
    std::vector<int> host_kv_indices;

    // Initialize with random data
    rand_vector(host_q.get(), q_size, -2.f, 2.f);
    rand_vector(host_k.get(), kv_size, -2.f, 2.f);
    rand_vector(host_v.get(), kv_size, -2.f, 2.f);
    if (dense_kv) {
        init_dense_kv_indices(host_kv_indptr, host_kv_indices, N, total_pages);
    } else {
        init_sparse_kv_indices(host_kv_indptr,
                               host_kv_indices,
                               N,
                               total_pages,
                               pa_traits<16, 32, 512, 8>::KV_TILE_SIZE);
    }
    const int indices_prefix_sum = static_cast<int>(host_kv_indices.size());

    // Allocate device memory
    bf16_t *dev_q, *dev_k, *dev_v, *dev_o;
    int *dev_kv_indptr, *dev_kv_indices;
    const size_t kv_indices_alloc_size = std::max<size_t>(host_kv_indices.size(), 1);
    CHECK_HIP(hipMalloc(&dev_q, q_size * sizeof(bf16_t)));
    CHECK_HIP(hipMalloc(&dev_k, kv_size * sizeof(bf16_t)));
    CHECK_HIP(hipMalloc(&dev_v, kv_size * sizeof(bf16_t)));
    CHECK_HIP(hipMalloc(&dev_o, q_size * sizeof(bf16_t)));
    CHECK_HIP(hipMalloc(&dev_kv_indptr, host_kv_indptr.size() * sizeof(int)));
    CHECK_HIP(hipMalloc(&dev_kv_indices, kv_indices_alloc_size * sizeof(int)));

    CHECK_HIP(hipMemcpy(dev_q, host_q.get(), q_size * sizeof(bf16_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_k, host_k.get(), kv_size * sizeof(bf16_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_v, host_v.get(), kv_size * sizeof(bf16_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_kv_indptr, host_kv_indptr.data(), host_kv_indptr.size() * sizeof(int), hipMemcpyHostToDevice));
    if (!host_kv_indices.empty()) {
        CHECK_HIP(hipMemcpy(dev_kv_indices, host_kv_indices.data(), host_kv_indices.size() * sizeof(int), hipMemcpyHostToDevice));
    }

    // Setup kernel arguments
    pa_kargs kargs{};
    kargs.ptr_q = dev_q;
    kargs.ptr_k = dev_k;
    kargs.ptr_v = dev_v;
    kargs.ptr_o = dev_o;
    kargs.kv_indptr = dev_kv_indptr;
    kargs.kv_indices = dev_kv_indices;
    kargs.N = N;
    kargs.H = H;
    kargs.D = D;
    kargs.total_pages = total_pages;
    kargs.stride_qo_n = H * D;
    kargs.stride_qo_h = D;
    kargs.stride_kv_page = D;

    // Dispatch to kernel
    auto run = [&]<typename PATraits>(PATraits) {
        if (D != PATraits::D_TILE_SIZE) {
            std::cerr << "This kernel only supports head dimension D=" << PATraits::D_TILE_SIZE << ", got D=" << D << "\n";
            return 1;
        }
        if ((H % (PATraits::Q_TILE_SIZE * PATraits::NUM_WARPS)) != 0) {
            std::cerr << "This kernel requires H to be a multiple of "
                      << (PATraits::Q_TILE_SIZE * PATraits::NUM_WARPS)
                      << " so every warp maps to a valid H tile, got H=" << H << "\n";
            return 1;
        }
        const int num_h_tiles = ceil_div(H, PATraits::Q_TILE_SIZE);
        const int num_h_blocks = ceil_div(num_h_tiles, PATraits::NUM_WARPS);
        dim3 grid(N, num_h_blocks, 1);
        dim3 block(PATraits::BLOCK_SIZE);

        printf("PA kernel launch config: grid=(%d,%d,%d), block=%d (NUM_WARPS=%d), smem=%zu bytes (K/V tiles)\n",
               grid.x, grid.y, grid.z, (int)block.x, PATraits::NUM_WARPS, PATraits::smem_size_bytes());

        pa_launch<PATraits>(kargs, grid, block);
        CHECK_HIP_KERNEL_LAUNCH();

        if (verify) {
            printf("\nValidating GPU results against CPU reference...\n");
            CHECK_HIP(hipMemcpy(host_o_gpu.get(), dev_o, q_size * sizeof(bf16_t), hipMemcpyDeviceToHost));
            pa_attention_ref(host_q.get(), host_k.get(), host_v.get(), host_o_ref.get(),
                              host_kv_indptr.data(), host_kv_indices.data(), N, H, D);

            bool all_valid = validate_pa_results(host_o_ref.get(), host_o_gpu.get(), N, H, D);
            printf("\n[Overall] %s\n", all_valid ? "✓ GPU KERNEL VALID" : "✗ GPU KERNEL FAILED");
            if (!all_valid) return 1;
        }

        printf("\n");
        benchmark_pa_kernel<PATraits>(kargs, grid, block, indices_prefix_sum);
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
    CHECK_HIP(hipFree(dev_kv_indptr));
    CHECK_HIP(hipFree(dev_kv_indices));

    return 0;
}
