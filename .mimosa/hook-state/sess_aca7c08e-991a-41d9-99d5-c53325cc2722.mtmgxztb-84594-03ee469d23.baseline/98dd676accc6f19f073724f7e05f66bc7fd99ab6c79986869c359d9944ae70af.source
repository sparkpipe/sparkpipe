#!/usr/bin/env bash
# k3_serial_tp_replay.sh SLICE_PACK OUT_PREFIX [TOKEN_ID] — slice a K3 tile_k=32
# slice pack 16 ways, build the serial-TP replay binary (shared harness + K3
# runner), run the FULL slice (golden) then ranks 0..15 at tp_degree 1, and
# print the sum-vs-golden comparison. Correctness only; one shard resident at a
# time under the 108 GiB budget (docs/serial_tp_replay.md).
set -euo pipefail
PACK="${1:?usage: k3_serial_tp_replay.sh SLICE_PACK OUT_PREFIX [TOKEN_ID]}"
PREFIX="${2:?usage: k3_serial_tp_replay.sh SLICE_PACK OUT_PREFIX [TOKEN_ID]}"
TOKEN="${3:-1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 1. shard 16 ways (the sharder admits degree 16 only on 32-element-tile packs)
PYTHONDONTWRITEBYTECODE=1 python3 tools/k3_shard.py "$PACK" "$PREFIX" 16

# 2. build the replay binary (sm_121a). Sources mirror k3_single_spark_step.sh
#    plus the shared harness (tests/serial_tp_replay.c) and this test TU.
INC="-I. -Iinclude -Isrc -Imodules/k3_resident_decode_stage/include -Imodel-families/common/include -Imodel-families/k3/include -Itests"
nvcc -std=c++17 -O3 --expt-relaxed-constexpr -lineinfo \
    -gencode arch=compute_121a,code=sm_121a $INC -Xcompiler -fPIC \
    tests/test_k3_serial_tp.cu \
    tests/serial_tp_replay.c \
    modules/k3_resident_decode_stage/source/spark_k3_resident_decode_stage_runner.cu \
    modules/k3_resident_decode_stage/source/spark_k3_resident_decode_stage_cuda.cu \
    inference/llms/kimi_k3/bind.cu \
    inference/llms/kimi_k3/unity.cu \
    modules/k3_resident_decode_stage/source/spark_k3_pack_load.c \
    modules/k3_resident_decode_stage/source/spark_k3_bind.c \
    modules/k3_resident_decode_stage/source/spark_k3_resident_decode_stage_module.c \
    runtime/json.c runtime/filesystem.c src/spark_status.c \
    build/libsparkpipe_model_common.a \
    -Xcompiler -pthread -ldl -lcuda -lcudart -o /tmp/k3_serial_tp

# 3. run: full slice (golden) + the 16 rank shards, then sum + compare
/tmp/k3_serial_tp "$PACK" \
    "$PREFIX.rank00.pack" "$PREFIX.rank01.pack" "$PREFIX.rank02.pack" \
    "$PREFIX.rank03.pack" "$PREFIX.rank04.pack" "$PREFIX.rank05.pack" \
    "$PREFIX.rank06.pack" "$PREFIX.rank07.pack" "$PREFIX.rank08.pack" \
    "$PREFIX.rank09.pack" "$PREFIX.rank10.pack" "$PREFIX.rank11.pack" \
    "$PREFIX.rank12.pack" "$PREFIX.rank13.pack" "$PREFIX.rank14.pack" \
    "$PREFIX.rank15.pack" --token "$TOKEN"
