CXX ?= c++
MOONCAKE_ROOT ?=
MOONCAKE_INCLUDE ?= $(MOONCAKE_ROOT)/mooncake-store/include
MOONCAKE_DEP_INCLUDE ?= $(MOONCAKE_ROOT)/local/include
MOONCAKE_COMMON_INCLUDE ?= $(MOONCAKE_ROOT)/mooncake-common/include
MOONCAKE_TRANSFER_INCLUDE ?= $(MOONCAKE_ROOT)/mooncake-transfer-engine/include
MOONCAKE_CACHELIB_INCLUDE ?= $(MOONCAKE_ROOT)/mooncake-store/include/cachelib_memory_allocator
MOONCAKE_GENERATED_INCLUDE ?= $(MOONCAKE_ROOT)/build/mooncake-store/include
MOONCAKE_LIB ?= $(MOONCAKE_ROOT)/build/mooncake-store/src
CXXFLAGS ?= -std=c++20 -Wall -Wextra -Werror -O3 -g -pthread -fPIC
LDFLAGS ?= -shared
TARGET ?= ../../build/libkv_mooncake.so

.PHONY: all clean

all: $(TARGET)

$(TARGET): spark_kv_mooncake.cpp ../../model-families/common/src/spark_kv_store.c ../../model-families/common/include/sparkpipe/spark_kv_store.h
	@test -n "$(MOONCAKE_ROOT)" || { echo "MOONCAKE_ROOT is required" >&2; exit 2; }
	@test -f "$(MOONCAKE_INCLUDE)/dummy_client.h" || { echo "missing Mooncake DummyClient headers" >&2; exit 2; }
	@test -f "$(MOONCAKE_DEP_INCLUDE)/ylt/coro_rpc/coro_rpc_client.hpp" || { echo "missing Mooncake yalantinglibs headers; set MOONCAKE_DEP_INCLUDE" >&2; exit 2; }
	@test -f "$(MOONCAKE_CACHELIB_INCLUDE)/Slab.h" || { echo "missing Mooncake CacheLib headers" >&2; exit 2; }
	@test -f "$(MOONCAKE_LIB)/libmooncake_store.so" || test -f "$(MOONCAKE_LIB)/libmooncake_store.a" || { echo "missing Mooncake Store library" >&2; exit 2; }
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -I../../include -I../../model-families/common/include -isystem $(MOONCAKE_INCLUDE) -isystem $(MOONCAKE_DEP_INCLUDE) -isystem $(MOONCAKE_COMMON_INCLUDE) -isystem $(MOONCAKE_TRANSFER_INCLUDE) -isystem $(MOONCAKE_CACHELIB_INCLUDE) -isystem $(MOONCAKE_CACHELIB_INCLUDE)/include -isystem $(MOONCAKE_CACHELIB_INCLUDE)/fake_include -isystem $(MOONCAKE_GENERATED_INCLUDE) spark_kv_mooncake.cpp ../../model-families/common/src/spark_kv_store.c -L$(MOONCAKE_LIB) -lmooncake_store -pthread -ldl $(LDFLAGS) -o $@

clean:
	rm -f $(TARGET)
