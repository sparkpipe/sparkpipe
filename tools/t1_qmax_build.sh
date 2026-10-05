#!/bin/sh
set -eu
: "${T1_STAGE:=/tmp/t1qmax_stage}"
: "${T1_RUNTIME:=/tmp/t1qmax}"
cd "$T1_STAGE/tree"
REPOSITORY_ROOT="$T1_STAGE/tree" make -C modules/qwen38_max_resident_decode_stage archive > "$T1_STAGE/make_q38.log" 2>&1
REPOSITORY_ROOT="$T1_STAGE/tree" EXPERT_CODEC=nvfp4 \
	MODEL_REVISION=84c6a6aa9497188e15a635ba793b0f95a79b1033 \
	CONTRACT_SHA256=$(sha256sum model_contracts/glm53_flash_authoritative.json | cut -d' ' -f1) \
	make -C modules/glm5_next_resident_decode_stage archive > "$T1_STAGE/make_g5n.log" 2>&1
make build/libsparkpipe_core.a build/sparkpipe_weightd build/libhidden_transport_spark_host_rdma_verbs.so > "$T1_STAGE/make_rest.log" 2>&1
ARCHIVE=$(find build/modules/qwen38_max_resident_decode_stage -name "libqwen38_max_resident_decode_stage_*.a" | head -1)
[ -n "$ARCHIVE" ] || { echo "module archive missing" >&2; exit 1; }
require_passed_record() {
	python3 - "$1" build/module_library/active <<'PY' || { echo "no passed module library record for $1" >&2; exit 1; }
import json, pathlib, sys
sha, root = sys.argv[1], pathlib.Path(sys.argv[2])
records = [json.loads(path.read_text()) for path in sorted(root.glob("*.json"))]
sys.exit(0 if any(r.get("artifact_sha256") == sha and r.get("validation_state") == "passed" for r in records) else 1)
PY
}
ARCHIVE_SHA256=$(sha256sum "$ARCHIVE" | cut -d' ' -f1)
require_passed_record "$ARCHIVE_SHA256"
nvcc -std=c++17 -O3 -arch=sm_121a \
	-I. -Iinclude -Imodel-families/common/include -Imodel-families/qwen38_max/include \
	-Imodules/qwen38_max_resident_decode_stage/include \
	-Imodules/qwen38_max_resident_decode_stage/source \
	-DSPARK_QWEN38_MAX_MODULE_BUILD=1 \
	-DSPARK_LLM_MTP_LAYER_COUNT=0u \
	-DQWEN38_MODEL_REVISION="d2dc35658bcf77e66643428cb52e774cc3b5bd29" \
	-DT1_VALIDATED_ARTIFACT_SHA256="\"$ARCHIVE_SHA256\"" \
	tools/t1_qmax_harness.c \
	"$ARCHIVE" \
	build/modules/glm5_next_resident_decode_stage/nvfp4/libglm5_next_resident_decode_stage_nvfp4.a \
	build/libsparkpipe_core.a \
	-L/usr/local/cuda/lib64 -lcudart -lcuda -o "$T1_RUNTIME/t1_qmax_harness"
cp build/sparkpipe_weightd build/libhidden_transport_spark_host_rdma_verbs.so "$T1_RUNTIME/"
cc -std=c11 -O2 -D_POSIX_C_SOURCE=200809L -I. -Iinclude \
	tools/qwen38max_parity_probe.c \
	runtime/spark_weightd_manifest.c runtime/spark_weightd_lease.c \
	-o "$T1_RUNTIME/qwen38max_parity_probe"
cp "$T1_RUNTIME/t1_qmax_harness" build/sparkpipe_weightd build/libhidden_transport_spark_host_rdma_verbs.so "$T1_RUNTIME/qwen38max_parity_probe" "$T1_STAGE/"
chmod +x "$T1_RUNTIME/t1_qmax_harness" "$T1_RUNTIME/sparkpipe_weightd" "$T1_RUNTIME/qwen38max_parity_probe"
ls -la "$T1_STAGE/t1_qmax_harness" "$T1_STAGE/libhidden_transport_spark_host_rdma_verbs.so" "$T1_STAGE/sparkpipe_weightd" "$T1_STAGE/qwen38max_parity_probe"
