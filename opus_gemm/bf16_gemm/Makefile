OPUS_INCLUDE_DIR ?= /path/to/aiter/csrc/include
HIPCC           ?= /opt/rocm/bin/hipcc
ARCH            ?= gfx950

COMMON_FLAGS = -I. -I$(OPUS_INCLUDE_DIR) -std=c++20 -fopenmp -O3 -Wall --offload-arch=$(ARCH) -ffast-math

BUILD = build
OUT   = $(BUILD)/gemm_a16w16.exe

KERNEL_SRCS = gemm_a16w16_quad_subtile_kernel.cc gemm_a16w16_mono_tile_kernel.cc
HOST_SRC    = gemm_host.cc
KERNEL_OBJS = $(patsubst %.cc,$(BUILD)/%.o,$(KERNEL_SRCS))
HOST_OBJ    = $(BUILD)/gemm_host.o

.PHONY: all clean benchmark

all: $(OUT)

$(BUILD):
	mkdir -p $(BUILD)

# Kernel objects: heavy device-pass template compilation. -D__HIPCC_RTC__ skips
# the implicit __clang_hip_runtime_wrapper.h on the host pass (~250 ms saved).
$(BUILD)/%.o: %.cc gemm_a16w16_quad_subtile_kernel_template.hpp gemm_a16w16_mono_tile_kernel_template.hpp gemm_defs.h | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) -D__HIPCC_RTC__ -Rpass-analysis=kernel-resource-usage -save-temps=obj -c -o $@

# Host object: needs full HIP runtime for <<<>>> launch syntax (no -D__HIPCC_RTC__).
$(HOST_OBJ): $(HOST_SRC) gemm_defs.h | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) -c -o $@

$(OUT): $(KERNEL_OBJS) $(HOST_OBJ)
	$(HIPCC) $^ --offload-arch=$(ARCH) -fopenmp -o $@

benchmark: $(OUT)
	@$(OUT)
	@$(OUT) -m 4096 -n 4096 -k 4096 -b 1

clean:
	rm -rf $(BUILD)
