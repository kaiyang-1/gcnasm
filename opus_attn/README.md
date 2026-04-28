# opus_attn — GQA Flash Attention Kernel for gfx950

Hand-written Grouped-Query Attention (GQA) kernel using the [OPUS](https://github.com/ROCm/aiter) template library and MFMA 32x32x16 bf16 instructions on AMD gfx950 (MI355).

## Features

- Flash Attention with online softmax (no materialized NxN attention matrix)
- Causal and non-causal variants (parallel compilation via `make -j`)
- Double-buffered K/V tiles in shared memory
- Software-pipelined global→shared→register data movement
- Fine-grained scheduling barriers for MFMA/VALU/EXP interleaving
- `__HIP_DEVICE_COMPILE__` guard for fast host pass (~580ms saved)
- CPU reference implementation for validation

## Files

```
opus_attn/
├── Makefile                              # Parallel build (make -j)
├── rebuild.sh                            # Build + benchmark both variants
├── gqa_defs.h                            # Shared types: bf16_t, opus_gqa_kargs, opus_gqa_traits
├── gqa_kernel_template.hpp               # Kernel implementation (included by variant .cc files)
├── gqa_kernel_causal.cc                  # Causal kernel instantiation
├── gqa_kernel_noncausal.cc               # Non-causal kernel instantiation
├── gqa_host.cc                           # Host launcher, benchmark, validation, main()
├── hip_minimal.h                         # Local HIP minimal header
└── monolithic/                           # Original single-file build (for reference)
    ├── gqa_gfx950.cc
    └── rebuild.sh
```

## Prerequisites

- ROCm with hipcc (tested with ROCm 7.1.1)
- gfx950 GPU target
- OPUS headers from [aiter](https://github.com/ROCm/aiter): set `OPUS_INCLUDE_DIR` to `<aiter_root>/csrc/include`
- OpenMP support (for CPU reference and random init)

## Build

```bash
cd opus_attn
export OPUS_INCLUDE_DIR=/path/to/aiter/csrc/include   # default: /home/carhuang/repo/aiter/csrc/include
make -j        # parallel build: causal + non-causal + host + link
```

Or use the convenience script (builds + runs benchmarks):

```bash
./rebuild.sh
```

### Monolithic build (for reference)

```bash
cd monolithic
./rebuild.sh
```

## Run

```bash
./build/gqa_attn.exe                          # causal, N=1024 (default)
./build/gqa_attn.exe --no-causal              # non-causal
./build/gqa_attn.exe -n=16384                 # causal, N=16384
./build/gqa_attn.exe --no-causal -n=16384     # non-causal, N=16384
```

### Command-line options

| Flag | Description | Default |
|------|-------------|---------|
| `-b` | Batch size | 16 |
| `-h_q` | Number of query heads | 64 |
| `-h_kv` | Number of KV heads | 8 |
| `-n` | Sequence length | 1024 |
| `-d` | Head dimension (must be 512) | 512 |
| `--causal` | Enable causal masking | (default) |
| `--no-causal` | Disable causal masking | |
| `--verify` | Enable CPU reference verification | off |
| `--no-verify` | Disable CPU reference verification | (default) |

Numeric flags support both `-n 16384` and `-n=16384` syntax.

## Kernel configuration

The kernel is parameterized via `opus_gqa_traits<Q_TILE, KV_TILE, D_TILE, NUM_WARPS, CAUSAL>`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Q_TILE_SIZE | 32 | Query tile size per warp |
| KV_TILE_SIZE | 64 | KV tile size in shared memory |
| D_TILE_SIZE | 512 | Head dimension (fixed) |
| NUM_WARPS | 8 | Warps per workgroup (512 threads) |
| CAUSAL | `true`/`false` | Causal masking (two separate kernel binaries) |
| MFMA | 32x32x16 bf16 | Matrix multiply instruction |
| Shared memory | 272384 bytes | Double-buffered K + V tiles |

## Compile time

Measured on MI355X with ROCm 7.1.1 and [optimized opus.hpp](https://github.com/ROCm/aiter/pull/2701):

| Build mode | Time | Notes |
|------------|------|-------|
| `make -j` (parallel) | **~1.35s** | Both kernel variants + host compiled in parallel |
| `make` (sequential) | ~4.2s | Causal + non-causal + host sequentially |
| Monolithic (`monolithic/rebuild.sh`) | ~2.9s | Single file, host+device |

### Compile-time techniques applied

- **`__HIP_DEVICE_COMPILE__` guard**: kernel .cc files skip the full kernel body on the host pass, providing only an empty stub for `__device_stub__` generation (~580ms saved per kernel file)
- **`-D__HIPCC_RTC__`**: applied to kernel .cc files to skip the implicit `__clang_hip_runtime_wrapper.h` on the host pass (~250ms saved per kernel file)
- **Parallel build**: causal, non-causal, and host compile simultaneously via `make -j`

### Per-file breakdown

| File | Time | VGPRs | SGPRs | Spill | Occ |
|------|:----:|:-----:|:-----:|:-----:|:---:|
| `gqa_kernel_causal.cc` | ~1.3s | 236 | 50 | 0 | 2 |
| `gqa_kernel_noncausal.cc` | ~1.3s | 232 | 44 | 0 | 2 |
| `gqa_host.cc` | ~0.9s | — | — | — | — |
| Link | ~0.03s | — | — | — | — |

## Performance

Historical D=128 results, measured on MI355X. Rerun benchmarks after the D=512 change before using these numbers:

| N | Causal TFlops | Causal Time | Non-causal TFlops | Non-causal Time |
|---:|---:|---:|---:|---:|
| 1024 | 721 | 0.38 ms | 1020 | 0.54 ms |
| 2048 | 930 | 1.18 ms | 1178 | 1.87 ms |
| 4096 | 1118 | 3.93 ms | 1227 | 7.17 ms |
| 8192 | 1207 | 14.57 ms | 1265 | 27.80 ms |
| 16384 | 1252 | 56.17 ms | 1275 | 110.37 ms |
