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
├── gqa_device_helpers.hpp                # Shared device helpers used by all kernel entries
├── gqa_d128_kernel_template.hpp          # D=128 kernel entry template
├── gqa_d192_dv128_kernel_template.hpp    # D=192, D_V=128 kernel entry template
├── gqa_d128_causal_kernel.cc             # D=128, D_V=128, causal TU
├── gqa_d128_noncausal_kernel.cc          # D=128, D_V=128, non-causal TU
├── gqa_d192_dv128_causal_kernel.cc       # D=192, D_V=128, causal TU
├── gqa_d192_dv128_noncausal_kernel.cc    # D=192, D_V=128, non-causal TU
├── gqa_host.cc                           # Host launcher, benchmark, validation, main()
├── hip_minimal.h                         # Local HIP minimal header
└── monolithic/                           # Original single-file build (for reference)
    ├── gqa.cc
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
make -j        # parallel build: four kernel TUs + host + link
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
./build/gqa_attn.exe -n 16384                 # causal, N=16384
./build/gqa_attn.exe --no-causal -n 16384     # non-causal, N=16384
./build/gqa_attn.exe -d 192 -dv 128           # dedicated 192/128 path
```

### Command-line options

| Flag | Description | Default |
|------|-------------|---------|
| `-b` | Batch size | 16 |
| `-h_q` | Number of query heads | 64 |
| `-h_kv` | Number of KV heads | 8 |
| `-n` | Sequence length | 1024 |
| `-d` | Q/K head dimension | 128 |
| `-dv` | V/O head dimension | same as `-d` |
| `--causal` | Enable causal masking | (default) |
| `--no-causal` | Disable causal masking | |
| `--verify` | Enable CPU reference verification | off |

All numeric flags support both `-n 16384` and `-n=16384` syntax.

## Kernel configuration

The kernel is parameterized via `opus_gqa_traits<Q_TILE, KV_TILE, D_TILE, DV_TILE, NUM_WARPS, CAUSAL>`. The current tree builds two root instantiations: `(D,DV)=(128,128)` and `(192,128)`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Q_TILE_SIZE | 32 | Query tile size per warp |
| KV_TILE_SIZE | 64 | KV tile size in shared memory |
| D_TILE_SIZE | 128 or 192 | Q/K head dimension |
| DV_TILE_SIZE | 128 | V/O head dimension |
| NUM_WARPS | 8 | Warps per workgroup (512 threads) |
| CAUSAL | `true`/`false` | Causal masking (two separate kernel binaries) |
| MFMA | 32x32x16 bf16 | Matrix multiply instruction |

## Compile time

Measured on MI355X with ROCm 7.1.1 and [optimized opus.hpp](https://github.com/ROCm/aiter/pull/2701):

| Build mode | Time | Notes |
|------------|------|-------|
| `make -j` (parallel) | **~1.35s** | Kernel TUs + host compiled in parallel |
| `make` (sequential) | ~4.2s | Kernel TUs + host sequentially |
| Monolithic (`monolithic/rebuild.sh`) | ~2.9s | Single file, host+device |

### Compile-time techniques applied

- **`__HIP_DEVICE_COMPILE__` guard**: kernel .cc files skip the full kernel body on the host pass, providing only an empty stub for `__device_stub__` generation (~580ms saved per kernel file)
- **`-D__HIPCC_RTC__`**: applied to kernel .cc files to skip the implicit `__clang_hip_runtime_wrapper.h` on the host pass (~250ms saved per kernel file)
- **Parallel build**: four kernel TUs and host compile simultaneously via `make -j`

### Kernel translation units

- `gqa_d128_causal_kernel.cc`
- `gqa_d128_noncausal_kernel.cc`
- `gqa_d192_dv128_causal_kernel.cc`
- `gqa_d192_dv128_noncausal_kernel.cc`

Splitting by shape and causality reduces incremental rebuild scope when tuning only one kernel path.

## Performance

B=16, H=64, H_KV=8, D=128, measured on MI355X:

| N | Causal TFlops | Causal Time | Non-causal TFlops | Non-causal Time |
|---:|---:|---:|---:|---:|
| 1024 | 721 | 0.38 ms | 1020 | 0.54 ms |
| 2048 | 930 | 1.18 ms | 1178 | 1.87 ms |
| 4096 | 1118 | 3.93 ms | 1227 | 7.17 ms |
| 8192 | 1207 | 14.57 ms | 1265 | 27.80 ms |
| 16384 | 1252 | 56.17 ms | 1275 | 110.37 ms |
