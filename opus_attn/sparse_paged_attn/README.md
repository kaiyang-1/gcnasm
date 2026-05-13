# Sparse Paged Prefill Attention for DeepSeek-V4

Optimized sparse paged prefill attention using the [OPUS](https://github.com/ROCm/aiter) C++ template library for DeepSeek-V4 inference on AMD gfx950.

This directory targets the DeepSeek-V4 MQA prefill shape: `H_Q = 128` query heads, `D = 512` head dimension, and one shared K/V stream with layout `[total_pages, D]`. The production use case is model inference, where every prefill token attends to a sparse set of historical and current-chunk K/V rows.

## Features

- DeepSeek-V4 prefill attention shape: BF16 Q/K/V/O, `H_Q = 128`, `D = 512`.
- MQA layout: Q/O carry the query-head dimension, while K/V are shared across query heads and have no head dimension.
- Paged sparse attention through two CSR index ranges per query token:
  - `prefix`: rows from already materialized or persistent K/V state.
  - `extend`: rows from the current prefill chunk.
- Online softmax across both CSR ranges, with no materialized attention matrix.
- OPUS-based gfx950 kernel using BF16 MFMA, double-buffered K/V shared-memory tiles, and FP32 accumulation.
- Standalone host harness with random sparse/dense index generation and CPU reference validation.

## Files

```text
sparse_paged_attn/
|-- Makefile                         # Build rules for the standalone executable
|-- pa_defs.h                        # Kernel argument struct and compile-time traits
|-- pa_host.cc                       # Host launcher, test harness, and CPU reference
|-- pa_prefill_kernel.cc             # D=512 kernel instantiation
`-- pa_prefill_kernel_template.hpp   # OPUS/HIP kernel implementation
```

## Attention Model

For each query token `i`, the caller provides two CSR rows:

```text
prefix_rows = kv_indices_prefix[kv_indptr_prefix[i] : kv_indptr_prefix[i + 1]]
extend_rows = kv_indices_extend[kv_indptr_extend[i] : kv_indptr_extend[i + 1]]
```

The kernel computes scaled dot-product attention over `prefix_rows` followed by `extend_rows`. Both ranges share the same online-softmax state, so the output is equivalent to attending over their concatenation:

```text
rows_i = concat(prefix_rows, extend_rows)
O[i, h, :] = softmax(Q[i, h, :] @ K[rows_i, :].T / sqrt(D)) @ V[rows_i, :]
```

In the DeepSeek-V4 prefill path, these two logical ranges map naturally to:

- `prefix`: previously available state, such as the sliding-window tail and compressed cache pages.
- `extend`: K/V rows produced by the current prefill chunk.

The standalone C++ interface exposes one K pointer and one V pointer. Therefore, both CSR ranges index the same `[total_pages, D]` K/V address space. Integrations that keep prefix and extend in separate physical buffers should either pack them into one address space before launch or extend the kernel arguments with separate base pointers.

## Tensor Layout

All tensor data is BF16.

| Tensor | Shape | Notes |
| --- | --- | --- |
| `Q` | `[N, H_Q, D]` | Query tokens. DeepSeek-V4 uses `H_Q = 128`. |
| `K` | `[total_pages, D]` | Shared MQA K rows. |
| `V` | `[total_pages, D]` | Shared MQA V rows. |
| `O` | `[N, H_Q, D]` | Output tokens. |
| `kv_indptr_prefix` | `[N + 1]` | CSR row pointers for prefix rows. |
| `kv_indices_prefix` | `[nnz_prefix]` | K/V row indices for prefix rows. |
| `kv_indptr_extend` | `[N + 1]` | CSR row pointers for extend rows. |
| `kv_indices_extend` | `[nnz_extend]` | K/V row indices for extend rows. |

The kernel assumes row-major contiguous layout with `D` as the fastest-changing dimension.

## Kernel Configuration

The compiled instantiation is:

```cpp
pa_traits<16, 32, 512, 8>
```

| Parameter | Value | Meaning |
| --- | --- | --- |
| `Q_TILE_SIZE` | `16` | Query-head tile per wave. |
| `KV_TILE_SIZE` | `32` | K/V rows loaded per sparse tile. |
| `D_TILE_SIZE` | `512` | Head dimension. |
| `NUM_WARPS` | `8` | Waves per workgroup. |
| `BLOCK_SIZE` | `512` | AMD wavefront size `64` times `8` waves. |

One workgroup covers one query token and up to `Q_TILE_SIZE * NUM_WARPS = 128` query heads. This matches the DeepSeek-V4 MQA target where the number of query heads is fixed at 128.

## Build

Prerequisites:

- ROCm 7+ with `hipcc`.
- gfx950 GPU target.
- OPUS headers from `aiter`, exposed through `OPUS_INCLUDE_DIR`.
- OpenMP support for the host reference path.

```bash
cd opus_attn/sparse_paged_attn
export OPUS_INCLUDE_DIR=/path/to/aiter/csrc/include
make -j
```

The executable is written to:

```text
build/pa_prefill.exe
```

## Run and Validate

Run with the DeepSeek-V4 MQA shape:

```bash
./build/pa_prefill.exe -h_q 128 -d 512 -n 1024 -total_pages 1024 --verify
```

Useful options:

| Option | Default | Description |
| --- | --- | --- |
| `-h_q` | `128` | Number of query heads. Currently only `128` is supported. |
| `-d` | `512` | Head dimension. Only `512` is compiled in this directory. |
| `-n` | `1024` | Number of query tokens in the standalone harness. |
| `-total_pages` | `N` | Number of K/V rows available to the generated CSR indices. |
| `--dense` | off | Generate dense CSR rows instead of random sparse rows. |
| `--verify` | off | Compare GPU output against the CPU reference implementation. |

The harness initializes random BF16 tensors, generates prefix and extend CSR index ranges, launches the kernel, and optionally checks the result against `pa_attention_ref()` in `pa_host.cc`.

## Integration Notes

- The caller owns CSR construction. Causal, sliding-window, compressed-cache, or top-k semantics should already be reflected in `kv_indices_*` and `kv_indptr_*`.
- Empty CSR rows are allowed by the CPU reference. Production callers should avoid empty rows unless zero output is intended.
- All K/V indices must be in `[0, total_pages)`.
- The softmax scale is fixed to `1 / sqrt(D)` in both the GPU kernel and the CPU reference.
- This directory intentionally contains only the D=512 prefill variant used by the DeepSeek-V4 MQA inference path.
