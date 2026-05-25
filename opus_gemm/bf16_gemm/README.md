# BF16 GEMM (gfx950)

Two hand-tuned BF16 GEMM kernels built on top of [`opus`](https://github.com/ROCm/aiter)
layouts, sharing one host driver. Both kernels compute `C = A · Bᵀ` (A row-major,
B col-major) in batched form on AMD CDNA / gfx950, using `mfma_16x16x32_bf16`.

The two kernels exist to compare **two different ways of decomposing one block
tile into MMAs**, not two different tile sizes — both are templates and can in
principle be instantiated at other shapes, subject to their respective
divisibility constraints.

## Files

| File | Role |
|---|---|
| [gemm_defs.h](gemm_defs.h) | Shared `opus_gemm_kargs` + user-facing `opus_gemm_traits` (block tile, dtype, vector widths). No `opus` dependency. |
| [gemm_a16w16_mono_tile_kernel_template.hpp](gemm_a16w16_mono_tile_kernel_template.hpp) | Mono-tile kernel template (device-only). |
| [gemm_a16w16_mono_tile_kernel.cc](gemm_a16w16_mono_tile_kernel.cc) | Mono-tile stub TU: host-pass empty body + device-pass explicit instantiation. |
| [gemm_a16w16_quad_subtile_kernel_template.hpp](gemm_a16w16_quad_subtile_kernel_template.hpp) | Quad-subtile kernel template (device-only). |
| [gemm_a16w16_quad_subtile_kernel.cc](gemm_a16w16_quad_subtile_kernel.cc) | Quad-subtile stub TU. |
| [gemm_host.cc](gemm_host.cc) | Host driver: random init, CPU reference, validate + benchmark both kernels. |
| [Makefile](Makefile) | Build (kernel TUs use `-D__HIPCC_RTC__` on the host pass to skip the HIP runtime wrapper). |

## Mono tile vs Quad subtile

The two kernels differ in the **shape of the per-K-iteration compute graph**:

|  | Mono tile | Quad subtile |
|---|---|---|
| Per-K MMAs | 1 (covers full `B_M × B_N`) | 4 (2×2 grid of half-tiles `HALF_B_M × HALF_B_N`) |
| Register accumulators | `v_c` | `v_c[2][2]` |
| SMEM buffers (A / B) | 2-slot double-buffer / 3-slot ring-buffer | `[2][2]` double-buffer |
| Layout dimension | `B_M`, `B_N` directly | `HALF_B_M = B_M/2`, `HALF_B_N = B_N/2` |
| Divisibility requirement | `B_M % (W_M·T_M) == 0`, `B_N % (W_N·T_N) == 0` | additionally requires `HALF_*` to satisfy the same |
| Output stores | 1 `store` per workgroup | 4 `store` (one per `v_c[i][j]`) |
| Pipeline grain | Tile-level — single big MMA per K iter; B prefetch hidden via 3-slot ring rotation | Subtile-level — 4 sub-MMAs interleaved with ds_read / async_load and `s_barrier`s for finer latency hiding |

`gemm_host.cc` launches both kernels back-to-back against the same A/B input
and reports each kernel's TFlops, so the two pipelines can be compared
directly. Each call site explicitly pairs a `Traits` with a kernel template
instantiation — no implicit dispatch — so any `(Traits, kernel)` combination
can be benchmarked. The default benchmark instantiates quad subtile at
`B_M × B_N = 256 × 256` and mono tile at `192 × 256`.

## Build & run

Set `OPUS_INCLUDE_DIR` to your `aiter/csrc/include` checkout (the Makefile's
default is a placeholder):

```sh
export OPUS_INCLUDE_DIR=/path/to/aiter/csrc/include
make -j
./build/gemm_a16w16.exe
./build/gemm_a16w16.exe -m 4096 -n 4096 -k 4096 -b 1
```

Flags: `-m`, `-n`, `-k`, `-b` (each takes the next argv). Override `ARCH`
(default `gfx950`) or `HIPCC` (default `/opt/rocm/bin/hipcc`) on the make line
if needed. `make clean` removes `build/`.

The driver runs both kernels against the same A/B, validates each against a
CPU reference (OpenMP), and prints per-kernel TFlops and a pass/fail summary.

## Benchmark

Square GEMM (M = N = K), `batch = 1`, on AMD MI355X.
Each row launches both kernels against the same input matrices; the higher
TFlops in each row is **bold**.

Command:

```sh
./build/gemm_a16w16.exe -b 1 -m <S> -n <S> -k <S>
```

| M = N = K | Quad subtile 256×256 (TFlops) | Mono tile 192×256 (TFlops) |
|:---------:|:----------------:|:----------------:|
|  4096 | **1248.82** | 1085.72 |
|  5120 | **1217.99** | 1121.96 |
|  6144 |  1294.43 | **1505.92** |
|  7168 |  1293.82 | **1309.22** |
|  8192 | **1572.70** | 1342.35 |
|  9216 |  1401.00 | **1511.43** |
| 10240 | **1442.25** | 1358.88 |
