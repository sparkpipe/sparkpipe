#!/bin/bash
set -u
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
RR="$HOME/sparkdata/glm5_next.tp16"
SRC="$HOME/g5mesh-src"
LOG=/tmp/mesh_task13.log
exec >> "$LOG" 2>&1
echo "=== START13 $(date -u +%H:%M:%S) ==="
rm -rf "$SRC"
git clone -q -b lane/glm5next-mtp-accept-takeover https://github.com/sparkpipe/sparkpipe.git "$SRC" || { echo CLONE-FAIL; exit 1; }
cd "$SRC"
git log --oneline -1
export PATH=/usr/local/cuda/bin:$PATH

make -j8 build/sparkpipe_model_compile || { echo HOST-BUILD-FAIL; exit 1; }

REV=84c6a6aa9497188e15a635ba793b0f95a79b1033
SHA=a40e9ec5fbfb0c1a180162c9d82915c887e8549fbd779c9f5dacb780a1498db4
sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' || true
make -C modules/glm5_next_resident_decode_stage publish \
  EXPERT_CODEC=fp8 MODEL_REVISION=$REV CONTRACT_SHA256=$SHA \
  NVCC=/usr/local/cuda/bin/nvcc CUDA_ARCH=sm_121a \
  STAGE_PACK_PATH="$RR/packs/glm5_next_stage.tp16.rank0.g5nsp" \
  || { echo PUBLISH-FAIL; exit 1; }

build/sparkpipe_model_compile \
  --model examples/model_descriptions/glm5_next_resident_decode_stage_fp8_firmware.json \
  --library build/module_library --output "$RR" \
  --cc /usr/bin/cc --include include \
  --cc-arg -L/usr/local/cuda/targets/sbsa-linux/lib --cc-arg -lcuda \
  --cc-arg -lcudart --cc-arg -lstdc++ --cc-arg -lm --cc-arg -ldl \
  --cc-arg -pthread || { echo COMPILE-FAIL; exit 1; }

n_new=$(strings "$RR/stages/stage_000/model_driver.so" | grep -c 'tp-nccl rank0 listening')
n_diag=$(strings "$RR/stages/stage_000/model_driver.so" | grep -c 'glm5next_pack_header_mismatch')
echo "strings: listening=$n_new diag=$n_diag"
{ [ "$n_new" -ge 1 ] && [ "$n_diag" -ge 1 ]; } || { echo STRINGS-FAIL; exit 1; }

cp "$RR/stages/stage_000/model_driver.so" /tmp/g5m_driver.so
cp build/module_library/active/e919a35b8c63bf20093b1f4e3e72d1f6e2af419cd5e76a3b97a641323da62122.json /tmp/g5m_rec.json
UNIT=$(python3 -c "import json;print(json.load(open('/tmp/g5m_rec.json'))['artifact_sha256'])")
cp "build/module_library/link_units/$UNIT.a" /tmp/g5m_unit.a || { echo UNIT-COPY-FAIL "$UNIT"; exit 1; }
cp "$RR/model_package.json" /tmp/g5m_pkg.json
echo "UNIT=$UNIT" | tee /tmp/g5m_unit_name
echo STAGE13-DONE
