// GQA flash attention kernel — D=192, D_V=128, non-causal instantiation
// Host pass: empty stub for __device_stub__ generation
// Device pass: includes the D=192, D_V=128 kernel template
#include <opus/hip_minimal.hpp>
#include "gqa_defs.h"
#ifndef __HIP_DEVICE_COMPILE__
template<typename Traits> __global__ void gqa_d192_dv128_kernel(opus_gqa_kargs kargs) {}
template __global__ void gqa_d192_dv128_kernel<opus_gqa_traits<32, 64, 192, 128, 8, false>>(opus_gqa_kargs);
#else
#include "gqa_d192_dv128_kernel_template.hpp"
template __global__ void gqa_d192_dv128_kernel<opus_gqa_traits<32, 64, 192, 128, 8, false>>(opus_gqa_kargs);
#endif
