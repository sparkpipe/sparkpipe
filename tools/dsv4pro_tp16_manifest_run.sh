#!/bin/bash
set -euo pipefail
cd "$HOME/dsv4pro_manifest_stage"
echo "== node=$(hostname) pack=$PACK"
echo "== dd-probe (cold-read stall guard)"
dd if="$PACK" of=/dev/null bs=1M count=64 iflag=direct 2>/dev/null || dd if="$PACK" of=/dev/null bs=1M count=64
echo "== strict compile of the routed-expert producer"
mkdir -p build
cc -std=c11 -Wall -Wextra -Werror -O2 -I. -Iinclude -Imodel-families/dsv4/include \
  -DSPARK_DSV4_PRO_BUILD=1 tools/dsv4_experts_manifest.c \
  runtime/spark_weightd_manifest.c src/spark_status.c src/spark_ck128.c \
  -o build/dsv4_experts_manifest
echo "== producer run"
./build/dsv4_experts_manifest "$PACK"
python3 - "$PACK" <<'PYEOF'
import os, struct, sys
pack_path = sys.argv[1]
manifest_path = pack_path + ".experts"
raw = open(manifest_path, "rb").read()
magic, version, range_count, zero = struct.unpack_from("<IIII", raw, 0)
assert len(raw) == 16 + 48 * range_count, (len(raw), range_count)
expert_bytes = 0
for i in range(range_count):
    record = struct.unpack_from("<IIIIQQ", raw, 16 + 48 * i)
    expert_bytes += record[5]
pack_bytes = os.path.getsize(pack_path)
manifest_bytes = os.path.getsize(manifest_path)
print("TP16-MANIFEST-RECEIPT node=%s pack=%s pack_bytes=%d manifest_bytes=%d range_count=%d expert_bytes=%d spine_bytes=%d magic=%08x version=%u" % (
    os.uname().nodename, pack_path, pack_bytes, manifest_bytes, range_count,
    expert_bytes, pack_bytes - expert_bytes, magic, version))
PYEOF
echo "MANIFEST-RUN-RC=0"
