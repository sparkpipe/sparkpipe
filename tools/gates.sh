#!/bin/sh
# Every gate, each checked on its exit code.
#
# This exists because a previous run reported nine passing static_asserts from a
# translation unit that had failed to compile: the echo was not gated on the
# status. A gate that cannot fail is not a gate.
ok=0; bad=0; skipped=0
run() {
	if eval "$2" >/dev/null 2>&1
	then printf "  %-26s PASS\n" "$1"; ok=$((ok+1))
	else printf "  %-26s FAIL\n" "$1"; bad=$((bad+1))
	fi
}
run_cuda() { if [ -x "${CUDA_HOME:-/opt/cuda}/bin/nvcc" ]; then run "$1" "$2"; else printf "  %-26s SKIP (nvcc unavailable)\n" "$1"; skipped=$((skipped+1)); fi; }
run_cuda "ptx capability gate" "python3 tests/test_ptx_capability_gate.py"
run "complete host inventory" "make -s test"
run "mma fragment mapping" "gcc -O2 -Wall -Wextra -o /tmp/g_f tests/test_mma_fragment_mapping.c && /tmp/g_f"
run "model constants"      "gcc -O2 -Wall -Wextra -I. -Imodel-families/glm52/include -o /tmp/g_c tests/test_model_constants.c && /tmp/g_c"
run "sub-byte packing"     "gcc -O2 -Wall -Wextra -o /tmp/g_p tests/test_pack.c && /tmp/g_p"
run "free dequant"         "gcc -O2 -Wall -Wextra -o /tmp/g_d tests/test_dequant.c && /tmp/g_d"
run "reference oracle"     "gcc -O2 -Wall -Wextra -Itests -o /tmp/g_r tests/test_reference.c -lm && /tmp/g_r"
run "weight binding"       "gcc -O2 -Wall -Wextra -I. -Imodules/glm52_resident_decode_stage/include -Iinclude -Ideployment/include -Imodel-families/glm52/include -o /tmp/g_b tests/test_pack_bind.c && /tmp/g_b"
run "sidebands"            "gcc -O2 -Wall -Wextra -I. -o /tmp/g_s tests/test_sideband.c && /tmp/g_s"
run "kv cache"             "gcc -O2 -Wall -Wextra -I. -o /tmp/g_kv tests/test_cache.c && /tmp/g_kv"
run "kv geometry"          "g++ -std=c++17 -fsyntax-only -Wall -Wextra -I. -Imodel-families/glm52/include tests/test_kv_geometry.cc"
run "workspace layout"     "gcc -O2 -Wall -Wextra -I. -o /tmp/g_w tests/test_group_gemm_workspace.c && /tmp/g_w"
run "tensor map geometry"  "gcc -O2 -Wall -Wextra -I. -o /tmp/g_t tests/test_tensor_map_geometry.c && /tmp/g_t"
run "tensor map encode"    "gcc -O2 -Wall -Wextra -I. -Itests/cuda_driver_stub -o /tmp/g_e tests/test_tensor_map_encode.c tests/cuda_driver_stub/stub.c && /tmp/g_e"
run "launch planning"      "g++ -std=c++17 -O2 -Wall -Wextra -I. -D__host__= -D__device__= -o /tmp/g_l tests/test_launch.c && /tmp/g_l"
run "recipes current"      "python3 tools/generate_recipe.py --check"
run "k3 stage doorway"     "gcc -Iinclude -Imodel-families/k3/include -Imodel-families/glm52/include -Wall -Werror -DNDEBUG -c modules/k3_resident_decode_stage/source/spark_k3_resident_decode_stage_validation.c -o /tmp/g_k3v.o"
run "mimo25 stage doorway"  "gcc -Iinclude -Wall -Werror -DNDEBUG -c modules/mimo25_resident_decode_stage/source/spark_mimo25_resident_decode_stage_validation.c -o /tmp/g_mimo25v.o"
run "qwen36 stage doorway"  "gcc -Iinclude -Wall -Werror -DNDEBUG -c modules/qwen36_resident_decode_stage/source/spark_qwen36_resident_decode_stage_validation.c -o /tmp/g_qwen36v.o"
run "dsv4 stage doorway"  "gcc -Iinclude -Wall -Werror -DNDEBUG -c modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_validation.c -o /tmp/g_dsv4v.o"
run "stage dispatch host compile" "g++ -x c++ -std=c++17 -fsyntax-only -Wall -Wextra -Werror -I. -Iinclude -Imodules/glm52_resident_decode_stage/include -Imodules/glm52_resident_decode_stage/source -Imodel-families/glm52/include -Itests/cuda_stub inference/stage/dispatch.cu"
run "hybrid kv arithmetic"   "make -s build/test_hybrid_kv_arithmetic && ./build/test_hybrid_kv_arithmetic"
run "uniform-profile admit"  "make -s build/test_uniform_profile_admit && ./build/test_uniform_profile_admit"
run "null seam link+run"   "make -s build/test_null_seam_link && ./build/test_null_seam_link"
run "seam symbol parity"   "sh tools/seam_parity.sh"
run "stage module + model"  "gcc -Iinclude -Imodules/glm52_resident_decode_stage/include -Imodules/glm52_resident_decode_stage/source -Imodel-families/glm52/include -Wall -Werror -DNDEBUG -c inference/stage/module.c -o /tmp/g_mod.o && gcc -Iinclude -Imodules/glm52_resident_decode_stage/include -Imodules/glm52_resident_decode_stage/source -Imodel-families/glm52/include -Wall -Werror -DNDEBUG -c modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_validation.c -o /tmp/g_val.o"
run "k3 kv seam"          "gcc -O2 -Wall -Wextra -Iinclude -Imodel-families/glm52/include -Imodel-families/k3/include -o /tmp/g_k3kv tests/test_k3_kv_cache.c cache/kv_cache.c && /tmp/g_k3kv"
run "state pool"          "gcc -O2 -Wall -Wextra -I. -o /tmp/g_sp tests/test_state_pool.c && /tmp/g_sp"
run "comms arena"         "gcc -O2 -Wall -Wextra -Werror -I. -o /tmp/g_ar tests/test_arena.c && /tmp/g_ar"
run "node daemons compile"  "make -s build/sparkpipe_glm52_cuda_residentd build/sparkpipe_glm52_ring_rank_daemon"
run_cuda "grouped topk builds"  "sh tools/build_grouped_topk.sh"
run_cuda "replay fold builds"   "sh tools/build_replay_fold.sh"
run_cuda "head topk builds"     "sh tools/build_head_topk.sh"
run "head topk on host"    "python3 tests/test_head_host.py"
run_cuda "nvcc: sm_121a build"  "sh tools/build.sh"
run "makefile parses"      "make -n all"
run "makefile: tools"      "make -n tools"
run "makefile: backend"    "make -n glm52_ring_service_backend"
run "makefile: variants"   "make -n -C modules/glm52_resident_decode_stage variants"
run "makefile: variant publish" "make -n -C modules/glm52_resident_decode_stage publish_variants"
run "core boundaries"      "make -s audit-boundaries"
run "no python in production" "python3 tests/test_no_python_in_production.py"
run "model serving architecture" "python3 tests/test_model_serving_architecture.py"
run "hardware handoff"     "make -s hardware_handoff"
run "package manifest"     "python3 tools/verify_package_manifest.py"
printf "  ---- %d pass, %d skip, %d fail\n" "$ok" "$skipped" "$bad"
[ "$bad" -eq 0 ]
