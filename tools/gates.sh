#!/bin/sh
# Every gate, each checked on its exit code.
#
# This exists because a previous run reported nine passing static_asserts from a
# translation unit that had failed to compile: the echo was not gated on the
# status. A gate that cannot fail is not a gate.
ok=0; bad=0
run() {
	if eval "$2" >/dev/null 2>&1
	then printf "  %-26s PASS\n" "$1"; ok=$((ok+1))
	else printf "  %-26s FAIL\n" "$1"; bad=$((bad+1))
	fi
}
run "ptx capability gate"  "python3 tests/test_ptx_capability_gate.py"
run "mma fragment mapping" "gcc -O2 -Wall -Wextra -o /tmp/g_f tests/test_mma_fragment_mapping.c && /tmp/g_f"
run "model constants"      "gcc -O2 -Wall -Wextra -I. -o /tmp/g_c tests/test_model_constants.c && /tmp/g_c"
run "sub-byte packing"     "gcc -O2 -Wall -Wextra -o /tmp/g_p tests/test_pack.c && /tmp/g_p"
run "free dequant"         "gcc -O2 -Wall -Wextra -o /tmp/g_d tests/test_dequant.c && /tmp/g_d"
run "reference oracle"     "gcc -O2 -Wall -Wextra -Itests -o /tmp/g_r tests/test_reference.c -lm && /tmp/g_r"
run "weight binding"       "gcc -O2 -Wall -Wextra -I. -Imodules/glm52_resident_decode_stage/include -Iinclude -Ideployment/include -Imodel-families/glm52/include -o /tmp/g_b tests/test_pack_bind.c && /tmp/g_b"
run "sidebands"            "gcc -O2 -Wall -Wextra -I. -o /tmp/g_s tests/test_sideband.c && /tmp/g_s"
run "kv cache"             "gcc -O2 -Wall -Wextra -I. -o /tmp/g_kv tests/test_cache.c && /tmp/g_kv"
run "kv geometry"          "g++ -std=c++17 -fsyntax-only -Wall -Wextra -I. tests/test_kv_geometry.cc"
run "workspace layout"     "gcc -O2 -Wall -Wextra -I. -o /tmp/g_w tests/test_group_gemm_workspace.c && /tmp/g_w"
run "tensor map geometry"  "gcc -O2 -Wall -Wextra -I. -o /tmp/g_t tests/test_tensor_map_geometry.c && /tmp/g_t"
run "tensor map encode"    "gcc -O2 -Wall -Wextra -I. -Itests/cuda_driver_stub -o /tmp/g_e tests/test_tensor_map_encode.c tests/cuda_driver_stub/stub.c && /tmp/g_e"
# The real compiler, for the real target. This replaced a keyword-shim gate that
# approximated nvcc by defining the CUDA keywords away. That shim could not see a
# missing include, could not see the 48 KB static shared limit, could not see
# -arch=sm_121a dropping its own suffix, and broke outright on extern __shared__.
# A proxy for the compiler is worth having only while the compiler is
# unavailable, and tools/get_cuda.sh means it is not.
run "launch planning"      "g++ -std=c++17 -O2 -Wall -Wextra -I. -D__host__= -D__device__= -o /tmp/g_l tests/test_launch.c && /tmp/g_l"
run "config coverage"      "python3 tests/test_config_coverage.py"
run "kernel algorithms"    "python3 tests/test_kernel_algorithms.py"
run "model contracts"      "python3 tests/test_model_driver_contracts.py"
run "nvcc: sm_121a build"  "sh tools/build.sh"
# The Makefile, which no gate covered. It did not parse: the reorganisation moved
# twelve sources and $(patsubst src/%.c,...) returned the non-matching paths
# UNCHANGED, so runtime/filesystem.c reached -include and make read a C file as a
# makefile. Every other gate was green throughout. .updaterepo-policy names four
# make targets as its validation and none of them could run.
run "makefile parses"      "make -n all"
run "makefile: test"       "make -n test"
run "makefile: tools"      "make -n tools"
run "makefile: backend"    "make -n glm52_ring_service_backend"
run "every source exists"  "python3 tests/test_sources_exist.py"
printf "  ---- %d pass, %d fail\n" "$ok" "$bad"
[ "$bad" -eq 0 ]
