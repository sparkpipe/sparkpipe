ifndef MODULE_FAMILY
$(error MODULE_FAMILY must be defined before including resident_decode_stage_rules.mk)
endif
ifndef MODULE_IDENTIFIER
$(error MODULE_IDENTIFIER must be defined before including resident_decode_stage_rules.mk)
endif
ifndef MODULE_TARGET
$(error MODULE_TARGET must be defined before including resident_decode_stage_rules.mk)
endif
ifndef MODEL_HEADER
$(error MODEL_HEADER must be defined before including resident_decode_stage_rules.mk)
endif
ifndef MODULE_ENTRY_PREFIX
$(error MODULE_ENTRY_PREFIX must be defined before including resident_decode_stage_rules.mk)
endif
ifndef MODULE_HOST_SOURCE
$(error MODULE_HOST_SOURCE must be defined before including resident_decode_stage_rules.mk)
endif
ifndef MODULE_CUDA_SOURCE
$(error MODULE_CUDA_SOURCE must be defined before including resident_decode_stage_rules.mk)
endif
ifndef STAGE_PACK_PATH
$(error STAGE_PACK_PATH must be defined before including resident_decode_stage_rules.mk)
endif
ifndef RUNTIME_CONFIGURATION
$(error RUNTIME_CONFIGURATION must be defined before including resident_decode_stage_rules.mk)
endif
ifndef MODULE_COMPILE_FLAGS
$(error MODULE_COMPILE_FLAGS must be defined before including resident_decode_stage_rules.mk)
endif

REPOSITORY_ROOT ?= $(abspath ../..)
CC ?= cc
AR ?= ar
NVCC ?= nvcc
CUDA_HOME ?= /usr/local/cuda
CUDA_ARCH ?= sm_121a
CUDA_COMPUTE_ARCH ?= $(subst sm_,compute_,$(CUDA_ARCH))
CFLAGS ?= -std=c11 -Wall -Wextra -Werror -O3
MODULE_POSIX_FLAGS := -D_POSIX_C_SOURCE=200809L -D_FILE_OFFSET_BITS=64
NVCCFLAGS ?= -std=c++17 -O3 --expt-relaxed-constexpr -lineinfo -gencode arch=$(CUDA_COMPUTE_ARCH),code=$(CUDA_ARCH)
MODULE_LIBRARY_ROOT ?= $(REPOSITORY_ROOT)/build/module_library
GPU_VALIDATOR ?=
GPU_VALIDATOR_ARGUMENTS ?=

BUILD_DIRECTORY ?= $(REPOSITORY_ROOT)/build/modules/$(MODULE_FAMILY)_resident_decode_stage
MODULE_ARCHIVE ?= $(BUILD_DIRECTORY)/lib$(MODULE_FAMILY)_resident_decode_stage.a
MODULE_COMMON_HOST_SOURCES ?= \
	$(REPOSITORY_ROOT)/runtime/stage_module_common.c
MODULE_HOST_SOURCES := \
	$(MODULE_HOST_SOURCE) \
	$(MODULE_COMMON_HOST_SOURCES) \
	$(MODULE_ADDITIONAL_HOST_SOURCES)
MODULE_INCLUDE_FLAGS ?= \
	-I"$(REPOSITORY_ROOT)/include" \
	-I"$(REPOSITORY_ROOT)/model-families/common/include" \
	-I"$(REPOSITORY_ROOT)/model-families/$(MODULE_FAMILY)/include" \
	-I"$(REPOSITORY_ROOT)/modules/$(MODULE_FAMILY)_resident_decode_stage/include" \
	-I"$(REPOSITORY_ROOT)/modules/$(MODULE_FAMILY)_resident_decode_stage/source"
MODULE_HOST_OBJECTS := $(foreach source,$(MODULE_HOST_SOURCES),$(BUILD_DIRECTORY)/$(subst /,_,$(basename $(source))).o)
MODULE_CUDA_OBJECT := $(BUILD_DIRECTORY)/$(subst /,_,$(basename $(MODULE_CUDA_SOURCE))).o
VALIDATION_CONFIGURATION_SHA256 := $(shell printf '%s\n' '$(RUNTIME_CONFIGURATION)' | sha256sum | awk '{print $$1}')
VALIDATION_RECIPE ?= $(MODULE_FAMILY).resident_decode_stage.$(CUDA_ARCH).gpu.config_$(VALIDATION_CONFIGURATION_SHA256).v1

.PHONY: all archive contract validate publish clean require_cuda_target require_gpu_validator require_stage_pack

all: contract

$(BUILD_DIRECTORY):
	mkdir -p "$@"

define SPARK_STAGE_HOST_SOURCE_RULE
$(BUILD_DIRECTORY)/$(subst /,_,$(basename $(1))).o: $(1) | $(BUILD_DIRECTORY)
	$(CC) $(CFLAGS) $(MODULE_POSIX_FLAGS) -fPIC -fvisibility=hidden $(MODULE_INCLUDE_FLAGS) \
		$(MODULE_COMPILE_FLAGS) -I"$(CUDA_HOME)/include" -include $(MODEL_HEADER) -MMD -MP -c "$$<" -o "$$@"
endef
$(foreach source,$(MODULE_HOST_SOURCES),$(eval $(call SPARK_STAGE_HOST_SOURCE_RULE,$(source))))

$(MODULE_CUDA_OBJECT): $(MODULE_CUDA_SOURCE) | $(BUILD_DIRECTORY)
	$(NVCC) $(NVCCFLAGS) $(MODULE_INCLUDE_FLAGS) $(MODULE_COMPILE_FLAGS) -include $(MODEL_HEADER) \
		-Xcompiler -Wall,-Wextra,-fPIC,-fvisibility=hidden -MMD -MP -c "$<" -o "$@"

$(MODULE_ARCHIVE): $(MODULE_HOST_OBJECTS) $(MODULE_CUDA_OBJECT)
	$(AR) rcs "$@" $^

require_cuda_target:
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "$(MODULE_FAMILY): nvcc is required for a CUDA archive" >&2; exit 1; }
	@test "$(CUDA_ARCH)" = "sm_121a" || { echo "$(MODULE_FAMILY): CUDA_ARCH must be sm_121a, got $(CUDA_ARCH)" >&2; exit 1; }

require_stage_pack:
	@test -r "$(STAGE_PACK_PATH)" || { echo "$(MODULE_FAMILY): readable STAGE_PACK_PATH is required: $(STAGE_PACK_PATH)" >&2; exit 1; }

require_gpu_validator:
	@test -n "$(GPU_VALIDATOR)" || { echo "$(MODULE_FAMILY): GPU_VALIDATOR must name an executable retained-receipt validator" >&2; exit 1; }
	@test -x "$(GPU_VALIDATOR)" || { echo "$(MODULE_FAMILY): GPU_VALIDATOR is not executable: $(GPU_VALIDATOR)" >&2; exit 1; }

contract:
	@set -e; \
	for source in $(MODULE_HOST_SOURCES); do \
		$(CC) $(CFLAGS) $(MODULE_POSIX_FLAGS) -fsyntax-only -I"$(REPOSITORY_ROOT)/tests/cuda_stub" \
			$(MODULE_INCLUDE_FLAGS) $(MODULE_COMPILE_FLAGS) -include $(MODEL_HEADER) $$source; \
	done

archive: require_cuda_target $(MODULE_ARCHIVE)

validate: require_cuda_target require_stage_pack require_gpu_validator $(MODULE_ARCHIVE)
	$(RUNTIME_CONFIGURATION) \
		$(GPU_VALIDATOR) \
		$(VALIDATION_CONFIGURATION_SHA256) \
		$(MODULE_ARCHIVE)

publish: require_cuda_target require_stage_pack require_gpu_validator $(MODULE_ARCHIVE)
	$(MAKE) -C $(REPOSITORY_ROOT) build/sparkpipe_module_publish
	$(RUNTIME_CONFIGURATION) \
		$(REPOSITORY_ROOT)/build/sparkpipe_module_publish \
		--library $(MODULE_LIBRARY_ROOT) \
		--module $(MODULE_IDENTIFIER) \
		--target $(MODULE_TARGET) \
		--link-unit $(MODULE_ARCHIVE) \
		--recipe $(VALIDATION_RECIPE) \
		--initialize $(MODULE_ENTRY_PREFIX)Initialize \
		--execute $(MODULE_ENTRY_PREFIX)Execute \
		--admit $(MODULE_ENTRY_PREFIX)Admit \
		--snapshot $(MODULE_ENTRY_PREFIX)Snapshot \
		--destroy $(MODULE_ENTRY_PREFIX)Destroy \
		--validator $(GPU_VALIDATOR) \
		--validator-arg $(VALIDATION_CONFIGURATION_SHA256) \
		$(GPU_VALIDATOR_ARGUMENTS)

clean:
	rm -rf $(BUILD_DIRECTORY)

-include $(MODULE_HOST_OBJECTS:.o=.d) $(MODULE_CUDA_OBJECT:.o=.d)
