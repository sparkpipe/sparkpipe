#!/bin/bash
# k3 timed warm receipt (lane 3 M3): the fleet-standard cold-launch
# numbers (dsv4_pro M3 shape, PR #1122 RUN-WALL convention) on the
# shared weightd, for the committed repetition-head preload set:
#
#   1. daemon cold-warm    - first warm after a daemon restart: lazy-arena
#                            create + pool-chunk materialization dominate
#                            (the once-per-daemon cost)
#   2. warm-daemon preload - a cold-launching instance's batch preload when
#                            the daemon already holds the set resident:
#                            WSET-WARM elapsed_ms. THE < 5 s claim.
#   3. steady-state        - repeated preloads against the warm daemon
#
# k3 specifics vs the dsv4 receipt:
#   - the committed working set spans ALL 92 routed layers, so each rank
#     filters it to its PP stage's layers first (a rank pack manifest
#     holds only its stage's layers; weightd_warm fails closed on
#     foreign keys). Per-stage subsets: s0/s1/s2/s3 printed by the filter.
#   - the warm presents the k3 runner arena identity: model kimi-k3 /
#     revision mxfp4 / topology 4 (--family k3 in weightd_warm); arena
#     bytes stay the pack size per the daemon's size-mismatch contract
#     (WDATTACH) - anything else is INVALID_ARGUMENT.
#
# Bases: smoke_set_raw_bytes_per_node (exact expert range bytes of the
# rank-local subset) and smoke_set_chunked_bytes_per_node (ceil-per-span
# at the pool chunk size, the fleet receipt basis), with the exact
# mapped-union figure alongside (pool chunk = 2 MiB measured on sm_121a).
#
# Queue invocation (bare repo-resident path): --cmd 'bash
# tools/devcycle/k3_warm_receipt.sh' --per-node --nodes spark0,...,sparkf
#
# Env:
#   SPARK_WEIGHTD_SOCKET   the shared daemon socket (required)
#   K3_WSET                full-model working set (default: the committed
#                          model-families/k3/smoke-k3-v1.wset)
#   K3_POOL_BYTES          expert pool bound (default 3 GiB: the preload
#                          set is 1,333.5 MiB raw per node, ~1.6 GiB
#                          chunked)
#   K3_WARM_RUNS           timed runs (default 5: 1 cold + 1 claim + 3
#                          steady)
#   K3_POOL_CHUNK_BYTES    chunk basis (default 2097152; measured sm_121a)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOCKET="${SPARK_WEIGHTD_SOCKET:?shared weightd socket required}"
WSET="${K3_WSET:-$REPO/model-families/k3/smoke-k3-v1.wset}"
POOL="${K3_POOL_BYTES:-3221225472}"
RUNS="${K3_WARM_RUNS:-5}"
CHUNK="${K3_POOL_CHUNK_BYTES:-2097152}"

HOST="$(hostname)"
case "$HOST" in
spark*) RANK=$((16#${HOST#spark})) ;;
*) echo "unexpected host: $HOST" >&2; exit 2 ;;
esac
STAGE=$((RANK / 4))
ROOT="/home/$HOST/sparkdata/k3.mxfp4.tp4pp4"
PACK="$ROOT/packs/k3.stage${STAGE}.rank0$((RANK % 4)).pack"
SHA="$(cut -d' ' -f1 "$PACK.sha256")"
MANIFEST="$PACK.experts"
for required in "$PACK" "$PACK.sha256" "$MANIFEST" "$WSET"; do
  [ -f "$required" ] || { echo "missing: $required" >&2; exit 2; }
done

cd "$REPO"
make -s build/weightd_warm

LOCAL_WSET="$(mktemp /tmp/k3-warm-rank$$-XXXXXX.wset)"
trap 'rm -f "$LOCAL_WSET"' EXIT
python3 - "$REPO" "$WSET" "$STAGE" "$LOCAL_WSET" <<'PYW'
import struct, sys
sys.path.insert(0, sys.argv[1] + "/tools")
import k3_smoke_experts as producer
data = open(sys.argv[2], "rb").read()
if len(data) == 0 or len(data) % 8 != 0:
    raise SystemExit("k3 warm receipt: malformed working set")
pairs = [struct.unpack_from("<II", data, index)
         for index in range(0, len(data), 8)]
local = producer.stage_pairs(pairs, int(sys.argv[3]))
with open(sys.argv[4], "wb") as out:
    out.write(producer.wset_bytes(local))
print(f"WSET-SPLIT rank_stage={sys.argv[3]} local_keys={len(local)} "
      f"of {len(pairs)} full-model keys")
PYW

echo "== warm receipt (rank $RANK, $PACK)"
echo "== pool=$POOL chunk=$CHUNK runs=$RUNS"
SPARK_WEIGHTD_EXPERT_POOL_BYTES="$POOL" \
  build/weightd_warm "$SOCKET" "$PACK" "$SHA" x 16 \
    --family k3 --identity-print

python3 - "$MANIFEST" "$LOCAL_WSET" "$CHUNK" <<'PYBASE'
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
PYBASE

echo "== timed warm (run 1 = daemon-cold-warm, run 2 = warm-daemon preload"
echo "   [the < 5 s claim], runs 3+ = steady-state)"
echo "   Each run prints RUN-WALL (whole invocation: connect + attach/"
echo "   arena-create + acquire + release + close, shell monotonic) and the"
echo "   tool's internal WSET-WARM elapsed_ms (acquire+release of the"
echo "   resident keys). The attach leg = RUN-WALL - WSET-WARM."
index=1
while [ "$index" -le "$RUNS" ]; do
  echo "-- warm run $index/$RUNS"
  began_ns=$(date +%s%N)
  SPARK_WEIGHTD_EXPERT_POOL_BYTES="$POOL" \
    build/weightd_warm "$SOCKET" "$PACK" "$SHA" x 16 \
    --family k3 --wset "$LOCAL_WSET" 300
  ended_ns=$(date +%s%N)
  printf 'RUN-WALL run=%s elapsed_ms=%s\n' "$index" \
    "$(( (ended_ns - began_ns) / 1000000 ))"
  index=$((index + 1))
done
echo "WARM-RECEIPT-DONE rank=$RANK pool=$POOL chunk=$CHUNK runs=$RUNS"
