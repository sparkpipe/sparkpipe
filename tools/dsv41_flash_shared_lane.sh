#!/bin/sh
# dsv41_flash shared-lane family wrapper (the GLM shared-socket pattern,
# docs/MULTIDEV_QUICKSTART.md) for lane 4: prepare a private deployment
# under the queue runtime root, enforce the one-digest weightd pack law,
# and exec the resident launch inside this queue cgroup.
#
# Required environment (the queue contract):
#   SPARK_QUEUE_RUNTIME_ROOT  private per-attempt runtime root
#   SPARK_QUEUE_RANK          this rank's index in the job's --nodes list
#   SPARK_WEIGHTD_SOCKET      the SHARED weightd socket (never start one)
# Optional (lane 4 defaults):
#   SPARK_WEIGHTD_LANE=4      weightd lane id for the shared socket
#   SPARK_TP_MESH_RANKS=4,5,6,7
#
# Usage:
#   dsv41_flash_shared_lane.sh prepare DEPLOYMENT_TREE PACK_DIR [RESIDENTD_BIN]
#
#   DEPLOYMENT_TREE  output of tools/dsv41_flash_gen_deployment.py
#   PACK_DIR         directory holding rank<R>.spstage + rank<R>.spstage.sha256
#   RESIDENTD_BIN    when given, the verified sparkpipe_model_residentd to
#                    exec after preparation passes (omit for prepare-only,
#                    e.g. until the family serving adapter lands)
#
# Exit codes: 2 usage, 3 missing environment/input, 4 pack law violation.
# The prepared runtime tree is private to this attempt and never reused.
set -eu

fail() {
    code=$1
    shift
    printf 'dsv41_flash_shared_lane: %s\n' "$*" >&2
    exit "$code"
}

[ "$#" -ge 2 ] && [ "$#" -le 3 ] || fail 2 "usage: $0 prepare DEPLOYMENT_TREE PACK_DIR [RESIDENTD_BIN]"
[ "$1" = prepare ] || fail 2 "unknown mode: $1 (only prepare exists)"
tree=$2
pack_dir=$3
residentd=${4:-}

for name in SPARK_QUEUE_RUNTIME_ROOT SPARK_QUEUE_RANK SPARK_WEIGHTD_SOCKET; do
    eval "value=\${$name:-}"
    [ -n "$value" ] || fail 3 "missing environment: $name"
done
rank=$SPARK_QUEUE_RANK
root=$SPARK_QUEUE_RUNTIME_ROOT
lane=${SPARK_WEIGHTD_LANE:-4}
mesh=${SPARK_TP_MESH_RANKS:-4,5,6,7}

[ -f "$tree/model_resident.json" ] || fail 3 "not a deployment tree: $tree"
stage_src=$(printf '%s/config/stage_%02d.json' "$tree" "$rank")
[ -f "$stage_src" ] || fail 3 "missing stage config: $stage_src"

pack_name=$(printf 'rank%x.spstage' "$rank")
[ -f "$pack_dir/$pack_name" ] || fail 3 "missing rank pack: $pack_dir/$pack_name"
[ -f "$pack_dir/$pack_name.sha256" ] || fail 4 "missing digest sidecar: $pack_name.sha256"

# Private runtime tree: config, packs (exactly one *.sha256), KV backing.
mkdir -p "$root/config" "$root/packs" "$root/kvcache"
cp "$stage_src" "$root/config/stage.json"
ln -sfn "$(cd "$pack_dir" && pwd)/$pack_name" "$root/packs/$pack_name"
ln -sfn "$(cd "$pack_dir" && pwd)/$pack_name.sha256" "$root/packs/$pack_name.sha256"

digest_count=$(find -L "$root/packs" -maxdepth 1 -name '*.sha256' | wc -l)
[ "$digest_count" -eq 1 ] || fail 4 "packs/ must hold exactly one *.sha256 digest, found $digest_count"

# Rewrite this rank's node entry onto the private runtime root and the
# shared socket; other nodes prepare their own copies on their hosts.
python3 - "$tree/model_resident.json" "$root/deployment.json" "$root" "$rank" \
    "$SPARK_WEIGHTD_SOCKET" <<'PY'
import json, sys

src, dst, root, rank, socket_path = sys.argv[1:6]
rank = int(rank)
document = json.load(open(src))
nodes = document.get("nodes", [])
mine = [n for n in nodes if n.get("rank_index") == rank]
if len(mine) != 1:
    raise SystemExit(f"rank_index {rank} appears {len(mine)} times, want exactly 1")
mine[0]["runtime_root"] = root
mine[0]["kv_backing_directory"] = root + "/kvcache"
document.setdefault("weightd", {})["socket_path"] = socket_path
with open(dst, "w", encoding="utf-8") as handle:
    json.dump(document, handle, indent=1)
    handle.write("\n")
PY

printf 'prepared rank %s lane %s mesh %s\n' "$rank" "$lane" "$mesh"
printf 'pack: %s\n' "$(readlink "$root/packs/$pack_name")"
printf 'digest: %s\n' "$(cat "$root/packs/$pack_name.sha256")"

if [ -n "$residentd" ]; then
    [ -x "$residentd" ] || fail 3 "residentd not executable: $residentd"
    export SPARK_WEIGHTD_SOCKET SPARK_WEIGHTD_LANE="$lane" SPARK_TP_MESH_RANKS="$mesh"
    exec "$residentd" --deployment "$root/deployment.json" --rank-index "$rank"
fi

printf 'launch line (adapter pending for this family):\n'
printf '  SPARK_WEIGHTD_SOCKET=%s SPARK_WEIGHTD_LANE=%s SPARK_TP_MESH_RANKS=%s \\\n' \
    "$SPARK_WEIGHTD_SOCKET" "$lane" "$mesh"
printf '  sparkpipe_model_residentd --deployment %s --rank-index %s\n' \
    "$root/deployment.json" "$rank"
exit 0
