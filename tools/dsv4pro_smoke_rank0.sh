#!/usr/bin/env bash
# DSV4 Pro TP4xPP4 rank-0 smoke: launch the residentd against the DEPLOYED
# pack and wait for the ready line (adapter+driver+pack load happen before
# it). Canonical launcher pattern: TERM-kill, wait, launch, poll the log.
# Parameterized: --target HOST (default spark0), --rank N (default 0).
set -euo pipefail

target="spark0"
rank="0"
timeout_s="${SPARK_DSV4PRO_SMOKE_TIMEOUT:-300}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target) target="$2"; shift 2 ;;
        --rank) rank="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

ssh -o BatchMode=yes "${target}" "
set -u
deploy=/home/\$(hostname)/sparkdata/dsv4_pro.tp4pp4
log=\${deploy}/smoke_rank${rank}.log

# TERM-kill any prior residentd, wait, no GPU reset needed for a fresh load.
pids=\$(pgrep -f sparkpipe_model_residentd || true)
if [[ -n \"\${pids}\" ]]; then
    kill -TERM \${pids} 2>/dev/null || true
    for _ in \$(seq 1 20); do pgrep -f sparkpipe_model_residentd >/dev/null || break; sleep 1; done
fi

cd \"\${deploy}\"
nohup ./bin/sparkpipe_model_residentd --deployment config/model_resident.json --rank-index ${rank} > \"\${log}\" 2>&1 &
echo \"launched residentd rank=${rank} pid=\$!\"

deadline=\$(( \$(date +%s) + ${timeout_s} ))
ready=0
while [[ \$(date +%s) -lt \${deadline} ]]; do
    if grep -q 'model_residentd ready' \"\${log}\" 2>/dev/null; then ready=1; break; fi
    if ! pgrep -f sparkpipe_model_residentd >/dev/null; then echo \"residentd exited early\"; break; fi
    sleep 2
done
echo \"--- log tail ---\"
tail -40 \"\${log}\" || true
if [[ \${ready} -ne 1 ]]; then
    echo \"SMOKE_FAIL rank=${rank}: ready line not observed within ${timeout_s}s\"
    exit 1
fi
echo \"SMOKE_PASS rank=${rank}\"
" 2>&1
