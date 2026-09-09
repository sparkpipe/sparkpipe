#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/dsv4pro_checkout"
echo "== dd-probe (cold-read stall guard)"
dd if="$PACK" of=/dev/null bs=1M count=64 iflag=direct 2>/dev/null || dd if="$PACK" of=/dev/null bs=1M count=64
echo "== strict compile of the routed-expert producer"
cc -std=c11 -Wall -Wextra -Werror -O2 -I. -Iinclude -Imodel-families/dsv4/include \
  -DSPARK_DSV4_PRO_BUILD=1 tools/dsv4_experts_manifest.c \
  runtime/spark_weightd_manifest.c src/spark_status.c src/spark_ck128.c \
  -o build/dsv4_experts_manifest
echo "== run against the accepted rank00 pack"
/usr/bin/time -v ./build/dsv4_experts_manifest "$PACK" 2>&1 | grep -E 'published|Maximum resident|Elapsed' || ./build/dsv4_experts_manifest "$PACK"
ls -la "$PACK.experts"
echo "EXPECTED-BYTES=$((16 + 48*16*384*6))"
echo "MANIFEST-RUN-RC=0"
