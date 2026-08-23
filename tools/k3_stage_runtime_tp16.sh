#!/usr/bin/env bash
# k3_stage_runtime_tp16.sh SOURCE_HOST CONFIG_DIR [ONLY_HOST ...] — pre-stage
# the K3 TP16 runtime trees (/home/<host>/sparkdata/k3.mxfp4.tp16) on the
# sparks. Companion to tools/k3_stage_runtime.sh, which hardcodes the
# TP4xPP4 root and pulls binaries off the sparka build box.
#
#   SOURCE_HOST  a host whose k3.mxfp4.tp4pp4 root already holds a good
#                bin/sparkpipe_model_residentd + lib/ tree (copied verbatim;
#                the pack loader reads tile_k from the manifest, so no rebuild
#                is expected - docs/K3_TP16_E2E_RUN_PLAN.md Phase 2).
#   CONFIG_DIR   directory holding the generated TP16 set: one
#                spark<hex>.json per rank (tools/k3_gen_adapter_configs.sh 16)
#                and model_resident.json (tools/k3_gen_deployment.sh OUT 16).
#   ONLY_HOST    optional rest args: stage just those hosts (default all 16).
#                Resumable: unreachable hosts are skipped loudly, rerun later.
#
# File placement only - no residentd is started; the window swap stays
# tools/fleet_swap.sh k3 AFTER the registry row points at the tp16 root.
set -euo pipefail
SRC_HOST="${1:?usage: k3_stage_runtime_tp16.sh SOURCE_HOST CONFIG_DIR [ONLY_HOST...]}"
CONFIG_DIR="${2:?usage: k3_stage_runtime_tp16.sh SOURCE_HOST CONFIG_DIR [ONLY_HOST...]}"
shift 2 || true
test -s "$CONFIG_DIR/model_resident.json" || { echo "missing $CONFIG_DIR/model_resident.json" >&2; exit 1; }

fail=0
for idx in $(seq 0 15); do
  hex=$(printf '%x' "$idx")
  host="spark$hex"
  if [ "$#" -gt 0 ]; then
    skip=1
    for want in "$@"; do [ "$want" = "$host" ] && skip=0 && break; done
    [ "$skip" = "1" ] && continue
  fi
  runtime="/home/$host/sparkdata/k3.mxfp4.tp16"
  if ! ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" "mkdir -p $runtime/bin $runtime/lib $runtime/config $runtime/kvcache"; then
    echo "SKIP $host unreachable (rerun when it returns)" >&2
    fail=1
    continue
  fi
  # binaries + nccl from the source host's staged tp4pp4 root, host-to-host
  ssh "$SRC_HOST" "rsync -a /home/$SRC_HOST/sparkdata/k3.mxfp4.tp4pp4/bin/ $host:$runtime/bin/" \
      || { echo "FAIL bin $host" >&2; fail=1; continue; }
  ssh "$SRC_HOST" "rsync -a /home/$SRC_HOST/sparkdata/k3.mxfp4.tp4pp4/lib/ $host:$runtime/lib/" \
      || { echo "FAIL lib $host" >&2; fail=1; continue; }
  scp -q "$CONFIG_DIR/model_resident.json" "$host:$runtime/config/model_resident.json" \
      || { echo "FAIL deployment json $host" >&2; fail=1; continue; }
  scp -q "$CONFIG_DIR/$host.json" "$host:$runtime/config/adapter.json" \
      || { echo "FAIL adapter json $host" >&2; fail=1; continue; }
  echo "staged $host (tp16)"
done
[ "$fail" = "0" ] || { echo "some hosts skipped/failed - rerun when reachable" >&2; exit 1; }
echo "k3 tp16 runtime staged"
