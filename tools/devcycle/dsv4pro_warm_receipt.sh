#!/bin/bash
# dsv4_pro timed warm receipt (lane 5 M3): the three cold-launch numbers on
# the shared weightd plus BOTH working-set byte bases (fleet standard after
# the 2026-09-22 chunk-basis correction: lane 0 measured GLM at 5.17x, this
# family at 2.15x - the factor is family-specific, so receipts carry the
# numbers on both bases instead of a factor):
#
#   1. daemon cold-warm    - first warm after a daemon restart: lazy-arena
#                            create + pool-chunk materialization dominate
#                            (the once-per-daemon cost)
#   2. warm-daemon preload - a cold-launching instance's batch preload when
#                            the daemon already holds the set resident:
#                            WSET-WARM elapsed_ms. THE < 5 s claim.
#   3. steady-state        - repeated preloads against the warm daemon
#
# Bases: smoke_set_raw_bytes_per_node (exact expert range bytes) and
# smoke_set_chunked_bytes_per_node (ceil-per-span at the pool chunk size,
# the fleet receipt basis), with the exact mapped-union figure alongside.
# Only the lazy/partial pool tier pays this chunking; the pin-all
# whole-arena tier does not (runtime/spark_weightd.c: pool chunk =
# max(gpu allocation granularity, 2 MiB); 2 MiB measured on sm_121a).
#
# Queue invocation (bare repo-resident path - the dispatcher pre-expands $
# patterns in inline command text; probe receipt dsv4pro-dollar-probe):
#   --cmd 'bash tools/devcycle/dsv4pro_warm_receipt.sh'
#
# Env:
#   SPARK_WEIGHTD_SOCKET     the shared daemon socket (required)
#   FAMILY_WSET              .wset path (default: the committed smoke set)
#   FAMILY_POOL_BYTES        expert pool bound (default 4 GiB: raw 1497.7 MiB
#                            vs ~3.2 GiB chunked for the 179-key set)
#   FAMILY_WARM_RUNS         timed runs (default 5: 1 cold + 1 claim + 3 steady)
#   FAMILY_POOL_CHUNK_BYTES  chunk basis (default 2097152; measured sm_121a)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOCKET="${SPARK_WEIGHTD_SOCKET:?shared weightd socket required}"
WSET="${FAMILY_WSET:-$REPO/model-families/dsv4/smoke-standard-v1.wset}"
POOL="${FAMILY_POOL_BYTES:-4294967296}"
RUNS="${FAMILY_WARM_RUNS:-5}"
CHUNK="${FAMILY_POOL_CHUNK_BYTES:-2097152}"

HOST="$(hostname)"
case "$HOST" in
spark*) RANK=$((16#${HOST#spark})) ;;
*) echo "unexpected host: $HOST" >&2; exit 2 ;;
esac
ROOT="/home/$HOST/sparkdata/dsv4_pro.tp4pp4"
PACK="$ROOT/packs/dsv4_pro_tp4_pp4_stage.spstage"
SHA="$(cut -d' ' -f1 "$PACK.sha256")"
MANIFEST="$PACK.experts"
for required in "$PACK" "$PACK.sha256" "$MANIFEST" "$WSET"; do
  [ -f "$required" ] || { echo "missing: $required" >&2; exit 2; }
done

cd "$REPO"
make -s build/weightd_warm

echo "== warm receipt (rank $RANK, $PACK)"
echo "== pool=$POOL chunk=$CHUNK runs=$RUNS"
build/weightd_warm "$SOCKET" "$PACK" "$SHA" x 16 \
  --family dsv4_pro --world-rank "$RANK" --identity-print

python3 - "$MANIFEST" "$WSET" "$CHUNK" <<'PYEOF'
import struct, sys

manifest_path, wset_path, chunk = sys.argv[1], sys.argv[2], int(sys.argv[3])
if chunk <= 0:
    raise SystemExit("chunk basis must be positive")
data = open(manifest_path, "rb").read()
if len(data) < 16:
    raise SystemExit("manifest shorter than header")
magic, version, count, zero = struct.unpack_from("=IIII", data, 0)
if zero != 0 or count == 0 or len(data) != 16 + 48 * count:
    raise SystemExit(f"malformed manifest: magic=0x{magic:08x} version={version} "
                     f"count={count} size={len(data)}")
ranges = {}
offset = 16
for _ in range(count):
    layer, expert, kind, reserved, range_offset, range_bytes = \
        struct.unpack_from("=IIIIQQ", data, offset)
    offset += 48
    if reserved != 0:
        raise SystemExit("nonzero reserved field in range record")
    ranges.setdefault((layer, expert), []).append((range_offset, range_bytes))
keys = set()
wset = open(wset_path, "rb").read()
if len(wset) == 0 or len(wset) % 8 != 0:
    raise SystemExit("wset must be nonempty complete key pairs")
for index in range(0, len(wset), 8):
    keys.add(struct.unpack_from("=II", wset, index))
missing = sorted(key for key in keys if key not in ranges)
if missing:
    raise SystemExit(f"wset keys absent from manifest: {missing[:4]}")
raw = ceil_span = 0
chunks = set()
for key in keys:
    for range_offset, range_bytes in ranges[key]:
        raw += range_bytes
        ceil_span += -(-range_bytes // chunk) * chunk
        chunks.update(range(range_offset // chunk,
                            (range_offset + range_bytes - 1) // chunk + 1))
union = len(chunks) * chunk
mib = 1024 * 1024
print(f"WARM-BASES keys={len(keys)} chunk_bytes={chunk}")
print(f"smoke_set_raw_bytes_per_node={raw} ({raw / mib:.1f} MiB)")
print(f"smoke_set_chunked_bytes_per_node={ceil_span} "
      f"({ceil_span / mib:.1f} MiB, ceil-per-span basis)")
print(f"smoke_set_chunked_union_bytes_per_node={union} "
      f"({union / mib:.1f} MiB, exact mapped union)")
print(f"chunk_inflation={ceil_span / raw:.2f}x "
      "(lazy/partial pool tier only; pin-all whole-arena pays no chunking)")
PYEOF

echo "== timed warm (run 1 = daemon-cold-warm, run 2 = warm-daemon preload"
echo "   [the < 5 s claim], runs 3+ = steady-state)"
echo "   Each run prints RUN-WALL (whole invocation: connect + attach/"
echo "   arena-create + acquire + release + close, shell monotonic) and the"
echo "   tool's internal WSET-WARM elapsed_ms (acquire+release of the"
echo "   resident keys). The attach leg = RUN-WALL - WSET-WARM (fleet"
echo "   convention 2026-09-22: instance time_to_ready splits explicitly"
echo "   into attach+arena-create vs lease bookkeeping)."
index=1
while [ "$index" -le "$RUNS" ]; do
  echo "-- warm run $index/$RUNS"
  began_ns=$(date +%s%N)
  SPARK_WEIGHTD_EXPERT_POOL_BYTES="$POOL" \
    build/weightd_warm "$SOCKET" "$PACK" "$SHA" x 16 \
    --family dsv4_pro --world-rank "$RANK" --wset "$WSET" 300
  ended_ns=$(date +%s%N)
  printf 'RUN-WALL run=%s elapsed_ms=%s\n' "$index" \
    "$(( (ended_ns - began_ns) / 1000000 ))"
  index=$((index + 1))
done
echo "WARM-RECEIPT-DONE rank=$RANK pool=$POOL chunk=$CHUNK runs=$RUNS"
