#!/usr/bin/env bash
# k3_tp16_tuesday_boot.sh - the exact ssh command sheet for the K3 TP16 boot
# on the Tuesday fleet return ("all 16 return Tuesday", ARCHITECTURE_MAP).
#
# Chains, in order, ONLY commands already documented in:
#   docs/K3_TP16_E2E_RUN_PLAN.md      (Tuesday run sheet T0/T1/T2/W)
#   docs/INCIDENT_RECOVERY_PLAYBOOK.md section 5 (per-host return checklist)
#   tools/k3_stage_runtime_tp16.sh / tools/k3_deploy_tp16.sh headers
# so this file is a driver, not a new procedure: every remote command below
# appears verbatim (modulo the host variable) in one of those sources.
#
# Phases (select with the argument list; default runs everything):
#   wait     poll until all 16 sparks admit BatchMode ssh (playbook 5.1)
#   recover  fabric link check/bounce + applies, mount checks, fsck-health
#            install, boot-unblock drop-ins (playbook 5.2, 5.5, 5.11, 5.12)
#   gate     stage the minimal proof tree to GATE_HOST and run the F1
#            owed-evidence multi-rank NCCL exchange gate there
#            (tests/test_k3_nccl_multirank_proof.cu via
#            tools/k3_nccl_multirank_gate.sh) - closes the DISPOSITION
#            "OWED to the ring" BEFORE any ring time is spent
#   configs  generate the TP16 adapter set + model_resident.json locally
#   stage    pre-stage runtime trees host-to-host (k3_stage_runtime_tp16.sh)
#   deploy   push + sha256-verify the sixteen rank packs from PACK_NODE
#            (k3_deploy_tp16.sh, executed ON the pack node: no workstation
#            in the data path)
#   registry verify the k3 registry row points at the tp16 root (the swap
#            starts K3 from whatever the registry names)
#   window   WITH WINDOW_REQUEST=1 only: save the fleet_status probe WITH
#            the receipt, then tools/fleet_swap.sh k3 (Phase 3)
#   boot     poll all sixteen control ports (registry 21480) until every
#            rank's residentd listens (Phase 4 step 1 abort signal)
#
# Usage:
#   bash tools/k3_tp16_tuesday_boot.sh [PHASE,...]
# Env:
#   PACK_NODE       node holding k3tp16prod rank packs   (default spark0)
#   PACK_DIR        pack source dir on PACK_NODE          (default /home/$PACK_NODE/k3tp16prod)
#   SOURCE_HOST     host whose k3.mxfp4.tp4pp4 bin/lib seed the trees
#                                                         (default $PACK_NODE)
#   GATE_HOST       returned spark the NCCL gate runs on  (default spark0)
#   GATE_WORLDS     worlds for the multirank gate         (default 2,4,8,16)
#   CONFIG_DIR      local config output dir               (default tmp/k3_tp16_tuesday)
#   WINDOW_REQUEST  1 = execute Phase 3 swap after prep    (default 0)
#   WAIT_TIMEOUT_S  wait-phase budget                      (default 7200)
#   PROBE_INTERVAL_S ssh probe cadence                     (default 120)
# Abort posture: nothing here consumes ring time before "window"; a failed
# phase exits nonzero and everything reruns idempotently (scp/sudo installs
# are convergent). Rollback stays tools/fleet_swap.sh (<60 s).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

HOSTS=(spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7
       spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf)
PACK_NODE="${PACK_NODE:-spark0}"
PACK_DIR="${PACK_DIR:-/home/$PACK_NODE/k3tp16prod}"
SOURCE_HOST="${SOURCE_HOST:-$PACK_NODE}"
GATE_HOST="${GATE_HOST:-spark0}"
GATE_WORLDS="${GATE_WORLDS:-2,4,8,16}"
CONFIG_DIR="${CONFIG_DIR:-tmp/k3_tp16_tuesday}"
WINDOW_REQUEST="${WINDOW_REQUEST:-0}"
WAIT_TIMEOUT_S="${WAIT_TIMEOUT_S:-7200}"
PROBE_INTERVAL_S="${PROBE_INTERVAL_S:-120}"
K3_CONTROL_PORT=21480          # fleet_registry.json k3.control_port_base
STAMP="$(date +%Y%m%dT%H%M%SZ)"
RECEIPT_DIR="$ROOT/tmp/gate_tmp/k3_tp16_tuesday_$STAMP"
mkdir -p "$RECEIPT_DIR"

SSH="ssh -o BatchMode=yes -o ConnectTimeout=8"
failures=0

note() { echo "[k3tp16-boot] $*"; }
phase_hdr() { echo; echo "==== $1 ===="; }

host_up() { $SSH "$1" true >/dev/null 2>&1; }

phase_wait() {
    phase_hdr "wait: BatchMode ssh admission on all 16 sparks (playbook 5.1)"
    local deadline=$((SECONDS + WAIT_TIMEOUT_S)) h n_up missing
    while :; do
        missing=""
        n_up=0
        for h in "${HOSTS[@]}"; do
            if host_up "$h"; then n_up=$((n_up+1)); else missing="$missing $h"; fi
        done
        if [ "$n_up" -eq 16 ]; then
            note "all 16 sparks reachable"
            return 0
        fi
        if [ "$SECONDS" -ge "$deadline" ]; then
            note "FAIL still unreachable after ${WAIT_TIMEOUT_S}s:$missing"
            return 1
        fi
        note "waiting on $((16 - n_up))/16:$missing (next probe in ${PROBE_INTERVAL_S}s)"
        sleep "$PROBE_INTERVAL_S"
    done
}

recover_host() {
    local h="$1" links bounced
    # playbook 5.1 full login
    if ! host_up "$h"; then
        note "FAIL $h not reachable"; failures=$((failures+1)); return 1
    fi
    # playbook 5.2 fabric link check + bounce (CX7 hotplug power-saving)
    links="$($SSH "$h" 'sudo ethtool enp1s0f0np0 enp1s0f1np1 2>/dev/null | grep -i "link detected"' || true)"
    bounced=""
    if echo "$links" | grep -qi "link detected: no"; then
        $SSH "$h" 'sudo ip link set enp1s0f0np0 down; sudo ip link set enp1s0f0np0 up; sudo ip link set enp1s0f1np1 down; sudo ip link set enp1s0f1np1 up'
        $SSH "$h" 'sudo timeout 30 /usr/local/sbin/ds4-switched-fabric-apply; sudo timeout 30 /usr/local/sbin/ds4-direct-pair-fabric-apply'
        bounced=" (bounced)"
        links="$($SSH "$h" 'sudo ethtool enp1s0f0np0 enp1s0f1np1 2>/dev/null | grep -i "link detected"' || true)"
        if echo "$links" | grep -qi "link detected: no"; then
            note "FAIL $h fabric link still down after bounce"
            failures=$((failures+1)); return 1
        fi
    fi
    note "$h fabric ok$bounced"
    # playbook 5.5 root + data mounts; every runtime root lives under sparkdata
    if ! $SSH "$h" "test -d /home/$h/sparkdata"; then
        note "FAIL $h /home/$h/sparkdata missing (mount check)"
        failures=$((failures+1)); return 1
    fi
    # playbook 5.11 post-boot fsck health check (convergent reinstall)
    if ! $SSH "$h" 'test -x /usr/local/bin/sparkpipe_fsck_health.sh'; then
        scp -q tools/devcycle/sparkpipe_fsck_health.sh tools/sparkpipe-fsck-health.service "$h:/tmp/" \
            && $SSH "$h" 'sudo cp /tmp/sparkpipe_fsck_health.sh /usr/local/bin/ && sudo chmod 0755 /usr/local/bin/sparkpipe_fsck_health.sh && sudo cp /tmp/sparkpipe-fsck-health.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now sparkpipe-fsck-health.service' \
            || { note "FAIL $h fsck-health install"; failures=$((failures+1)); return 1; }
        $SSH "$h" 'cat /var/lib/sparkpipe/fsck-health/last.json' >/dev/null 2>&1 \
            || note "WARN $h fsck-health last.json not written yet (non-fatal)"
    fi
    # playbook 5.12 boot-unblock drop-ins: services fail soft when fabric down
    if ! $SSH "$h" 'systemctl show -p TimeoutStartUSec ds4-switched-fabric.service 2>/dev/null | grep -q "=2min"'; then
        scp -q tools/devcycle/boot-unblock/*.conf "$h:/tmp/" \
            && $SSH "$h" 'for f in /tmp/*.conf; do u=$(basename "$f" .conf); sudo mkdir -p "/etc/systemd/system/$u.d"; sudo cp "$f" "/etc/systemd/system/$u.d/10-boot-timeout.conf"; done; sudo systemctl daemon-reload' \
            || { note "FAIL $h boot-unblock drop-ins"; failures=$((failures+1)); return 1; }
    fi
    note "$h recovered (login, fabric, sparkdata, fsck-health, boot-unblock)"
}

phase_recover() {
    phase_hdr "recover: playbook section-5 essentials on every returning host"
    local h
    for h in "${HOSTS[@]}"; do recover_host "$h"; done
    [ "$failures" -eq 0 ] || { note "FAIL recovery had $failures failure(s); fix before staging"; return 1; }
}

phase_gate() {
    phase_hdr "gate: live multi-rank NCCL exchange proof on $GATE_HOST (F1 owed evidence)"
    if ! host_up "$GATE_HOST"; then
        note "FAIL $GATE_HOST unreachable - rerun this phase when it returns"; return 1
    fi
    # minimal repo mirror: exactly what the gate compiles plus its header pins
    tar -cf - \
        include/sparkpipe/spark_tp_device_collective.h \
        include/sparkpipe/spark_hidden_transport.h \
        include/sparkpipe/spark_status.h \
        src/spark_status.c \
        ring/transport/tp_device_collective.c \
        ring/transport/tp_device_collective_nccl.c \
        ring/transport/tp_device_collective_nccl.h \
        ring/transport/hidden_transport.c \
        tests/test_k3_nccl_multirank_proof.cu \
        tools/k3_nccl_multirank_gate.sh \
        | $SSH "$GATE_HOST" "mkdir -p k3_gate_tree && tar -xf - -C k3_gate_tree" \
        || { note "FAIL staging gate tree to $GATE_HOST"; return 1; }
    # prefer the staged fleet libnccl over whatever ldconfig finds; the gate
    # SKIPs (exit 0) rather than false-reds a box without nvcc/nccl
    local remote_script='set -e
cd k3_gate_tree
NCCL=""
for cand in /home/GATEHOSTPLACEHOLDER/sparkdata/k3.mxfp4.tp16/lib/runtime_libs/libnccl.so.2 /home/GATEHOSTPLACEHOLDER/sparkdata/k3.mxfp4.tp4pp4/lib/runtime_libs/libnccl.so.2; do
    [ -e "$cand" ] && { NCCL="$cand"; break; }
done
if [ -n "$NCCL" ]; then export SPARK_NCCL_MODULE="$NCCL"; fi
bash tools/k3_nccl_multirank_gate.sh nvcc WORLDSPLACEHOLDER'
    remote_script="${remote_script//GATEHOSTPLACEHOLDER/$GATE_HOST}"
    remote_script="${remote_script//WORLDSPLACEHOLDER/$GATE_WORLDS}"
    $SSH "$GATE_HOST" "$remote_script" | tee "$RECEIPT_DIR/multirank_gate.log"
    grep -q "PASS k3_nccl_multirank overall" "$RECEIPT_DIR/multirank_gate.log" \
        || { note "FAIL multirank NCCL exchange gate did NOT pass (log: $RECEIPT_DIR/multirank_gate.log) - RED means STOP, do not request the window"; return 1; }
    note "multirank NCCL exchange proof PASSED (worlds $GATE_WORLDS) - DISPOSITION owed-evidence closed on $GATE_HOST"
}

phase_configs() {
    phase_hdr "configs: TP16 adapter set + deployment json (run-plan Phase 2)"
    mkdir -p "$CONFIG_DIR"
    bash tools/k3_gen_adapter_configs.sh "$CONFIG_DIR" 16 || return 1
    bash tools/k3_gen_deployment.sh "$CONFIG_DIR/model_resident.json" 16 || return 1
    note "config set under $CONFIG_DIR (one spark<hex>.json per rank + model_resident.json)"
}

phase_stage() {
    phase_hdr "stage: pre-stage the sixteen tp16 runtime trees (file placement only)"
    bash tools/k3_stage_runtime_tp16.sh "$SOURCE_HOST" "$CONFIG_DIR"
}

phase_deploy() {
    phase_hdr "deploy: rank packs from $PACK_NODE:$PACK_DIR (no workstation in the data path)"
    if ! host_up "$PACK_NODE"; then
        note "FAIL pack node $PACK_NODE unreachable"; return 1
    fi
    $SSH "$PACK_NODE" "test -d '$PACK_DIR' && test -s '$PACK_DIR/SHA256SUMS.tp16'" \
        || { note "FAIL $PACK_DIR (or its SHA256SUMS.tp16) missing on $PACK_NODE"; return 1; }
    scp -q tools/k3_deploy_tp16.sh "$PACK_NODE:/tmp/" \
        && $SSH "$PACK_NODE" "bash /tmp/k3_deploy_tp16.sh '$PACK_DIR'" | tee "$RECEIPT_DIR/deploy.log"
    grep -q "16/16 rank packs verified" "$RECEIPT_DIR/deploy.log" || { note "FAIL deploy did not report 16/16 verified (see $RECEIPT_DIR/deploy.log)"; return 1; }
}

phase_registry() {
    phase_hdr "registry: k3 row must point at the tp16 root before any swap"
    python3 - <<'PY'
import json, sys
reg = json.load(open("tools/devcycle/fleet_registry.json"))
root = reg["models"]["k3"]["runtime_root"]
pack = reg["models"]["k3"].get("pack_dir", "")
ok = root.endswith("k3.mxfp4.tp16") and pack.endswith("k3.mxfp4.tp16/packs")
print("registry k3.runtime_root =", root, "(OK)" if ok else "(WRONG ROOT)")
sys.exit(0 if ok else 1)
PY
    if [ $? -ne 0 ]; then
        note "FAIL flip tools/devcycle/fleet_registry.json models.k3.runtime_root AND pack_dir to k3.mxfp4.tp16 (land via PR with tools/sparkpipe_github_pat.sh), then rerun this phase"
        return 1
    fi
    note "registry row ready for the tp16 window"
}

phase_window() {
    phase_hdr "window: Phase 3 request (exclusive whole-fleet hour)"
    if [ "$WINDOW_REQUEST" != "1" ]; then
        note "skipped (set WINDOW_REQUEST=1 to execute): save probe + tools/fleet_swap.sh k3"
        return 0
    fi
    bash tools/devcycle/fleet_status.sh | tee "$RECEIPT_DIR/fleet_status.txt" || return 1
    note "fleet probe saved WITH the receipt (fleet rule); swapping k3 in"
    bash tools/fleet_swap.sh k3
}

phase_boot() {
    phase_hdr "boot: wait for sixteen residentd control ports (registry base $K3_CONTROL_PORT)"
    if [ "$WINDOW_REQUEST" != "1" ]; then
        note "skipped (rank verification applies after a WINDOW_REQUEST=1 swap; run this phase alone post-swap if needed)"
        return 0
    fi
    local deadline=$((SECONDS + 900)) h n
    while :; do
        n=0
        for h in "${HOSTS[@]}"; do
            if $SSH "$h" "ss -ltn 2>/dev/null | grep -q ':$K3_CONTROL_PORT '" \
                && $SSH "$h" 'pgrep -f bin/sparkpipe_model_residentd >/dev/null'; then
                n=$((n+1))
            fi
        done
        note "ranks up: $n/16"
        if [ "$n" -eq 16 ]; then
            note "TP16 BOOT COMPLETE: 16/16 ranks listening on $K3_CONTROL_PORT - proceed to Phase 4 (smoke -> capture -> B1)"
            return 0
        fi
        if [ "$SECONDS" -ge "$deadline" ]; then
            note "FAIL boot incomplete after 900s (Phase 4 abort signal: a rank never bound its control port)"
            return 1
        fi
        sleep 15
    done
}

PHASES="${1:-wait,recover,gate,configs,stage,deploy,registry,window,boot}"
OLDIFS="$IFS"; IFS=','; set -- $PHASES; IFS="$OLDIFS"
overall_rc=0
for p in "$@"; do
    phase_hdr "phase $p"
    "phase_$p" || { overall_rc=1; note "phase $p FAILED - stopping (rerun the script; phases are idempotent)"; break; }
done

echo
if [ "$overall_rc" -eq 0 ]; then
    note "requested phases complete. Receipts: $RECEIPT_DIR"
    note "rollback posture: tools/fleet_swap.sh restores the previous holder in <60 s; tp4pp4 roots/configs untouched."
else
    note "FAILED at a phase boundary - zero ring time consumed unless WINDOW_REQUEST=1 already ran."
fi
exit "$overall_rc"
