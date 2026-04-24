OPUS_INCLUDE_DIR ?= /home/carhuang/repo/aiter/csrc/include
HIPCC           ?= /opt/rocm/bin/hipcc
ARCH            ?= gfx950

COMMON_FLAGS = -I. -I$(OPUS_INCLUDE_DIR) -std=c++20 -fopenmp -O3 -Wall --offload-arch=$(ARCH) -ffast-math

BUILD = build
OUT   = $(BUILD)/gqa_attn.exe

D128_KERNEL_SRCS = \
	gqa_d128_causal_kernel.cc \
	gqa_d128_noncausal_kernel.cc

D192_DV128_KERNEL_SRCS = \
	gqa_d192_dv128_causal_kernel.cc \
	gqa_d192_dv128_noncausal_kernel.cc

KERNEL_SRCS = $(D128_KERNEL_SRCS) $(D192_DV128_KERNEL_SRCS)
HOST_SRC    = gqa_host.cc
KERNEL_OBJS = $(patsubst %.cc,$(BUILD)/%.o,$(KERNEL_SRCS))
HOST_OBJ    = $(BUILD)/gqa_host.o
KERNEL_COMMON_DEPS = gqa_device_helpers.hpp gqa_defs.h
KERNEL_COMPILE_CMD = $(HIPCC) $< $(COMMON_FLAGS) -D__HIPCC_RTC__ -Rpass-analysis=kernel-resource-usage -c -o $@

.PHONY: all clean benchmark

all: $(OUT)

$(BUILD):
	mkdir -p $(BUILD)

# Kernel objects (heavy — device+host passes, parallelizable via make -j)
# -D__HIPCC_RTC__ skips implicit __clang_hip_runtime_wrapper.h on host pass (~250ms saved)
$(patsubst %.cc,$(BUILD)/%.o,$(D128_KERNEL_SRCS)): $(BUILD)/%.o: %.cc gqa_d128_kernel_template.hpp $(KERNEL_COMMON_DEPS) | $(BUILD)
	$(KERNEL_COMPILE_CMD)

$(patsubst %.cc,$(BUILD)/%.o,$(D192_DV128_KERNEL_SRCS)): $(BUILD)/%.o: %.cc gqa_d192_dv128_kernel_template.hpp $(KERNEL_COMMON_DEPS) | $(BUILD)
	$(KERNEL_COMPILE_CMD)

# Host object (no -D__HIPCC_RTC__ — needs full HIP runtime for <<<>>> syntax)
$(HOST_OBJ): $(HOST_SRC) gqa_defs.h | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) -c -o $@

# Link
$(OUT): $(KERNEL_OBJS) $(HOST_OBJ)
	$(HIPCC) $^ --offload-arch=$(ARCH) -fopenmp -o $@

benchmark: $(OUT)
	@echo "=== Causal ===" && $(OUT)
	@echo "=== Non-causal ===" && $(OUT) --no-causal

clean:
	rm -rf $(BUILD)
