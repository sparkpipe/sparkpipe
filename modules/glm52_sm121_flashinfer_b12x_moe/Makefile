REPOSITORY_ROOT := $(abspath ../..)
CC ?= cc
AR ?= ar
CFLAGS ?= -std=c11 -Wall -Wextra -Werror -O3
BUILD_DIRECTORY := $(REPOSITORY_ROOT)/build/modules/glm52_sm121_flashinfer_b12x_moe
ADAPTER_OBJECT := $(BUILD_DIRECTORY)/spark_glm52_sm121_flashinfer_b12x_moe_link_required_adapter.o
ADAPTER_ARCHIVE := $(BUILD_DIRECTORY)/libglm52_sm121_flashinfer_b12x_moe_adapter.a
DEPENDENCY := $(ADAPTER_OBJECT:.o=.d)

.PHONY: archive clean

archive: $(ADAPTER_ARCHIVE)

$(BUILD_DIRECTORY):
	mkdir -p $(BUILD_DIRECTORY)

$(ADAPTER_OBJECT): source/spark_glm52_sm121_flashinfer_b12x_moe_link_required_adapter.c include/sparkpipe/spark_glm52_sm121_flashinfer_b12x_moe.h | $(BUILD_DIRECTORY)
	$(CC) -I$(REPOSITORY_ROOT)/include -Iinclude $(CFLAGS) -fPIC -fvisibility=hidden -MMD -MP -c $< -o $@

$(ADAPTER_ARCHIVE): $(ADAPTER_OBJECT)
	rm -f $@
	$(AR) rcs $@ $^

clean:
	rm -rf $(BUILD_DIRECTORY)

-include $(DEPENDENCY)
