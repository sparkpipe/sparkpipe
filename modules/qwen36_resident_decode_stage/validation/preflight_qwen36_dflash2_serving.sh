#!/usr/bin/env bash
#
# DFlash2 serving preflight: does the process that is actually RUNNING carry the
# selector wiring, and does its environment arm the drafter?
#
# Two real misdeploys motivated this, both of which looked exactly like a kernel
# bug (drafts = [committed, 0, 0, ...] with no diagnostics in the log):
#   1. the daemon was started with a stripped environment, so
#      SPARK_QWEN36_DSPARK_PACK_PATH and SPARK_QWEN36_SERVING_SPEC_METHOD were
#      absent inside the process and the drafter was never armed nor even asked
#      for (the adapter falls back to MTP and never sets DSPARK_DRAFT_AFTER);
#   2. the driver was published into one runtime root while the service ran from
#      another, so the loaded model_driver.so predated the W7 host wiring.
# Both are invisible from the build side and obvious from here.
#
# usage: preflight_qwen36_dflash2_serving.sh [RUNTIME_ROOT]
# exit 0 = the deployed process can draft with the selector; nonzero = it cannot.
set -uo pipefail

failures=0
note() { printf '%-46s %s\n' "$1" "$2"; }
check() { if [ "$2" = "PASS" ]; then note "$1" "PASS"; else note "$1" "FAIL  $3"; failures=$((failures + 1)); fi; }

environment_file="${SPARKPIPE_RESIDENTD_ENV:-/etc/sparkpipe/residentd.env}"
runtime_root="${1:-}"
if [ -z "${runtime_root}" ] && [ -r "${environment_file}" ]; then
    runtime_root="$(sed -n 's/^RUNTIME_ROOT=//p' "${environment_file}" | tail -1)"
fi
runtime_root="${runtime_root:-${RUNTIME_ROOT:-}}"
note "runtime root" "${runtime_root:-(unresolved)}"
[ -n "${runtime_root}" ] && [ -d "${runtime_root}" ] \
    && check "runtime root exists" PASS \
    || check "runtime root exists" FAIL "set RUNTIME_ROOT or pass it as \$1"

driver="${runtime_root}/lib/model_driver.so"
if [ -r "${driver}" ]; then
    note "driver" "${driver} ($(stat -c %s "${driver}") B, $(stat -c %y "${driver}" | cut -c1-16))"
    # String literals, not symbol names: the emit sequence is a static inline
    # function that is inlined away, so its NAME is absent even from a correct
    # build. These three literals exist only in a W7 build - a pre-W7 driver
    # carries dspark_head / dspark_layer and none of them.
    for marker in "dspark_pack_loaded" "dspark_block_forward entry" "dspark_selector"; do
        if strings "${driver}" | grep -qF "${marker}"; then check "driver carries '${marker}'" PASS; else
            check "driver carries '${marker}'" FAIL "pre-W7 artifact - published to a different runtime root?"; fi
    done
else
    check "driver readable" FAIL "${driver}"
fi

pid="$(pgrep -f sparkpipe_model_residentd | head -1)"
if [ -n "${pid}" ] && [ -r "/proc/${pid}/environ" ]; then
    note "residentd pid" "${pid}"
    live_root="$(tr '\0' '\n' < "/proc/${pid}/environ" | sed -n 's/^RUNTIME_ROOT=//p' | tail -1)"
    note "root the PROCESS runs from" "${live_root:-(none)}"
    [ "${live_root}" = "${runtime_root}" ] && check "process root == deploy root" PASS \
        || check "process root == deploy root" FAIL "deploying to ${runtime_root} while serving from ${live_root}"
    pack="$(tr '\0' '\n' < "/proc/${pid}/environ" | sed -n 's/^SPARK_QWEN36_DSPARK_PACK_PATH=//p' | tail -1)"
    method="$(tr '\0' '\n' < "/proc/${pid}/environ" | sed -n 's/^SPARK_QWEN36_SERVING_SPEC_METHOD=//p' | tail -1)"
    [ -n "${pack}" ] && check "process has DSPARK_PACK_PATH" PASS || check "process has DSPARK_PACK_PATH" FAIL "unset inside the process (a stripped unit environment)"
    [ "${method}" = "dspark" ] && check "process spec method is dspark" PASS || check "process spec method is dspark" FAIL "${method:-(unset)} -> the adapter never builds a DSPARK_DRAFT_AFTER frame"
    if [ -n "${pack}" ]; then
        [ -r "${pack}" ] && check "drafter pack readable" PASS || check "drafter pack readable" FAIL "${pack}"
    fi
else
    check "residentd running" FAIL "no sparkpipe_model_residentd process"
fi

if [ "${failures}" -eq 0 ]; then
    echo "PREFLIGHT PASS - the running process can draft with the DFlash2 selector"
else
    echo "PREFLIGHT FAIL - ${failures} check(s); the drafter cannot run as deployed"
fi
exit "${failures}"
