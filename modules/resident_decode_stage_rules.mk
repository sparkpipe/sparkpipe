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
MODULE_POSIX_FLAGS := -D_GNU_SOURCE -D_POSIX_C_SOURCE=200809L -D_FILE_OFFSET_BITS=64
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

.PHONY: all archive contract validate publish clean require_cuda_target require_gpu_validator require_stage_pack variants cold_variants publish_variants

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
	$(AR) rcs "$@.$$$$.tmp" $^ && mv "$@.$$$$.tmp" "$@"

# BATCH VARIANTS, opt-in per family. A family Makefile sets
# MODULE_BATCH_VARIANT_BUCKETS to the bucket ladder (every power of two from
# 1 to 1024) and splits MODULE_IDENTIFIER into MODULE_IDENTIFIER_PREFIX and
# MODULE_IDENTIFIER_SUFFIX; this file then emits one archive per bucket from
# the same translation units with -DSPARK_BATCH_BUCKET=<n> the ONLY
# difference, and every per-bucket constant that flag selects lives in the
# family's spark_<family>_batch_tuning.h. DRY by construction: one recipe
# template and one bucket list per family, so a fix to a recipe reaches
# every variant and a source edit cannot land in one variant only.
#
# A bucket is a capacity ceiling - the b64 archive serves 1-64 rows - and
# runtime selection takes the tightest ceiling at or above the microbatch,
# so a live batch pads to at most twice itself and the pools the bucket
# sizes stay within that factor of the truth. The unflagged archive IS the
# b1024 module and keeps the family's published unbucketed identifier; the
# tighter variants publish under <prefix>.b<n>.<suffix> (the contract
# generator adopts the bucketed form for b1024 when it variants). Trim the
# set for local iteration through the family's bucket variable, never by
# editing a variant.
MODULE_BATCH_VARIANT_BUCKETS ?=

ifneq ($(MODULE_BATCH_VARIANT_BUCKETS),)
ifndef MODULE_IDENTIFIER_PREFIX
$(error MODULE_IDENTIFIER_PREFIX must be defined when MODULE_BATCH_VARIANT_BUCKETS is set)
endif
ifndef MODULE_IDENTIFIER_SUFFIX
$(error MODULE_IDENTIFIER_SUFFIX must be defined when MODULE_BATCH_VARIANT_BUCKETS is set)
endif

# $(1) is the bucket everywhere below: a literal bucket anywhere in this
# file - a flag, an object name, an archive name - would be a hand-written
# variant rule, i.e. the fork this system exists to prevent.
MODULE_BATCH_VARIANT_ARCHIVE = $(MODULE_ARCHIVE:.a=_b$(1).a)
MODULE_BATCH_VARIANT_ARCHIVES := $(foreach bucket,$(MODULE_BATCH_VARIANT_BUCKETS),$(call MODULE_BATCH_VARIANT_ARCHIVE,$(bucket)))

# The variant recipes mirror the default-archive rules flag for flag; the
# b-suffixed object names keep a variant build from ever linking a
# default-tuned object or vice versa.
define SPARK_STAGE_VARIANT_HOST_SOURCE_RULE
$(BUILD_DIRECTORY)/$(subst /,_,$(basename $(1)))_b$(2).o: $(1) | $(BUILD_DIRECTORY)
	$(CC) $(CFLAGS) $(MODULE_POSIX_FLAGS) -fPIC -fvisibility=hidden $(MODULE_INCLUDE_FLAGS) \
		$(MODULE_COMPILE_FLAGS) -USPARK_BATCH_BUCKET -DSPARK_BATCH_BUCKET=$(2) -I"$(CUDA_HOME)/include" -include $(MODEL_HEADER) -MMD -MP -c "$$<" -o "$$@"
endef

define SPARK_MODULE_BATCH_VARIANT_RULES
$(foreach source,$(MODULE_HOST_SOURCES),$(eval $(call SPARK_STAGE_VARIANT_HOST_SOURCE_RULE,$(source),$(1))))
$(BUILD_DIRECTORY)/$(subst /,_,$(basename $(MODULE_CUDA_SOURCE)))_b$(1).o: $(MODULE_CUDA_SOURCE) | $(BUILD_DIRECTORY)
	$(NVCC) $(NVCCFLAGS) $(MODULE_INCLUDE_FLAGS) $(MODULE_COMPILE_FLAGS) -USPARK_BATCH_BUCKET -DSPARK_BATCH_BUCKET=$(1) -include $(MODEL_HEADER) \
		-Xcompiler -Wall,-Wextra,-fPIC,-fvisibility=hidden -MMD -MP -c "$$<" -o "$$@"

$(call MODULE_BATCH_VARIANT_ARCHIVE,$(1)): $(foreach source,$(MODULE_HOST_SOURCES),$(BUILD_DIRECTORY)/$(subst /,_,$(basename $(source)))_b$(1).o) $(BUILD_DIRECTORY)/$(subst /,_,$(basename $(MODULE_CUDA_SOURCE)))_b$(1).o
	$(AR) rcs "$$@.$$$$$.tmp" $$^ && mv "$$@.$$$$$.tmp" "$$@"
endef
$(foreach bucket,$(MODULE_BATCH_VARIANT_BUCKETS),$(eval $(call SPARK_MODULE_BATCH_VARIANT_RULES,$(bucket))))

MODULE_BATCH_VARIANT_DEPENDENCIES := $(foreach object,$(foreach bucket,$(MODULE_BATCH_VARIANT_BUCKETS),$(foreach source,$(MODULE_HOST_SOURCES) $(MODULE_CUDA_SOURCE),$(BUILD_DIRECTORY)/$(subst /,_,$(basename $(source)))_b$(bucket).o)),$(object:.o=.d))

# variants is the compile-only half of the variant contract: every bucket's
# archive from the one template above. cold_variants mirrors a cold archive.
variants: require_cuda_target $(MODULE_BATCH_VARIANT_ARCHIVES)

cold_variants:
	$(MAKE) clean
	$(MAKE) variants

# publish_variants is the publish half: each tighter bucket validated and
# published under its own module ID, so the module library holds
# content-addressed identities the loader resolves between. b1024 is absent
# by design: the unflagged archive IS the b1024 build and already publishes
# under the unbucketed MODULE_IDENTIFIER via the publish target - a second
# identity for the same bits is noise.
publish_variants: require_cuda_target require_stage_pack require_gpu_validator $(MODULE_BATCH_VARIANT_ARCHIVES)
	$(MAKE) -C $(REPOSITORY_ROOT) build/sparkpipe_module_publish
	@set -e; \
		for bucket in $(filter-out 1024,$(MODULE_BATCH_VARIANT_BUCKETS)); do \
			SPARK_MODULE_BATCH_BUCKET=$$bucket $(RUNTIME_CONFIGURATION) \
				$(REPOSITORY_ROOT)/build/sparkpipe_module_publish \
				--library $(MODULE_LIBRARY_ROOT) \
				--module $(MODULE_IDENTIFIER_PREFIX).b$$bucket.$(MODULE_IDENTIFIER_SUFFIX) \
				--target $(MODULE_TARGET) \
				--link-unit $(call MODULE_BATCH_VARIANT_ARCHIVE,$$bucket) \
				--recipe $(VALIDATION_RECIPE) \
				--initialize $(MODULE_ENTRY_PREFIX)Initialize \
				--execute $(MODULE_ENTRY_PREFIX)Execute \
				--admit $(MODULE_ENTRY_PREFIX)Admit \
				--snapshot $(MODULE_ENTRY_PREFIX)Snapshot \
				--destroy $(MODULE_ENTRY_PREFIX)Destroy \
				--validator $(GPU_VALIDATOR) \
				--validator-arg $(VALIDATION_CONFIGURATION_SHA256) \
				$(GPU_VALIDATOR_ARGUMENTS); \
		done
endif

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
	SPARK_MODULE_BATCH_BUCKET=$(if $(MODULE_BATCH_VARIANT_BUCKETS),1024,0) $(RUNTIME_CONFIGURATION) \
		$(GPU_VALIDATOR) \
		$(VALIDATION_CONFIGURATION_SHA256) \
		$(MODULE_ARCHIVE)

publish: require_cuda_target require_stage_pack require_gpu_validator $(MODULE_ARCHIVE)
	$(MAKE) -C $(REPOSITORY_ROOT) build/sparkpipe_module_publish
	SPARK_MODULE_BATCH_BUCKET=$(if $(MODULE_BATCH_VARIANT_BUCKETS),1024,0) $(RUNTIME_CONFIGURATION) \
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

-include $(MODULE_HOST_OBJECTS:.o=.d) $(MODULE_CUDA_OBJECT:.o=.d) \
	$(MODULE_BATCH_VARIANT_DEPENDENCIES)
