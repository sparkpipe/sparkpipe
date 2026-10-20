#!/usr/bin/env bash
set -uo pipefail
: "${SPARK_FLEET_NODE:?set SPARK_FLEET_NODE to the Spark the node tests drive}"
: "${SPARK_FLEET_API_HOST:?set SPARK_FLEET_API_HOST}"
: "${SPARK_FLEET_API_PORT:?set SPARK_FLEET_API_PORT}"
: "${SPARK_FLEET_API_RESTART:?set SPARK_FLEET_API_RESTART to the command that restarts the API on its host}"
: "${SPARK_FLEET_HUB:?set SPARK_FLEET_HUB to the host holding the fleet heartbeats}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
failures=0
for test in test_expert_io_perf test_jit_kv_page_fault test_lossless_doorbell; do
    if python3 "$root/tests/$test.py" "$SPARK_FLEET_NODE"; then
        echo "PASS $test node=$SPARK_FLEET_NODE"
    else
        echo "FAIL $test node=$SPARK_FLEET_NODE"
        failures=$((failures + 1))
    fi
done
if python3 "$root/tests/test_transport_stability.py" "${SPARK_FLEET_STABILITY_MINUTES:-5}" "${SPARK_FLEET_STABILITY_SEED:-42}"; then
    echo "PASS test_transport_stability"
else
    echo "FAIL test_transport_stability"
    failures=$((failures + 1))
fi
echo "fleet tests failures=$failures"
exit $((failures != 0))
