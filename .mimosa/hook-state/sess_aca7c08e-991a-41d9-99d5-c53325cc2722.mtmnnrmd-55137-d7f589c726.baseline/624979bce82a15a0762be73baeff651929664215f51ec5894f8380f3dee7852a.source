#!/usr/bin/env bash
# Fleet smoke for DSV4 Pro TP4xPP4: launch residentd on every listed host
# (boundary transports need their peers up, so all nodes start together),
# poll each daemon log for the ready line, print a summary, then TERM the
# daemons unless KEEP=1. Fully parameterized: --hosts "spark0 spark3 ...".
set -euo pipefail

hosts=""
timeout_s="${SPARK_DSV4PRO_SMOKE_TIMEOUT:-480}"
keep="${SPARK_DSV4PRO_SMOKE_KEEP:-0}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --hosts) hosts="$2"; shift 2 ;;
        --timeout) timeout_s="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "${hosts}" ]] || { echo '--hosts "spark0 spark3 ..." required' >&2; exit 2; }

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
launcher="${repo}/tools/dsv4pro_smoke_launch_rank.sh"

echo "== distributing launcher + starting daemons ($(date))"
for host in ${hosts}; do
    r=$(( 16#${host#spark} ))
    rsync -q -a "${launcher}" "${host}:/home/${host}/sparkdata/dsv4_pro.tp4pp4/smoke_launch.sh"
    ssh -o BatchMode=yes "${host}" "SMOKE_RANK=${r} bash /home/${host}/sparkdata/dsv4_pro.tp4pp4/smoke_launch.sh" &
done
wait
echo "== all launch commands issued; polling for ready lines (timeout ${timeout_s}s)"

deadline=$(( $(date +%s) + timeout_s ))
declare -A ready fail
while [[ $(date +%s) -lt ${deadline} ]]; do
    pending=0
    for host in ${hosts}; do
        r=$(( 16#${host#spark} ))
        [[ -n "${ready[${host}]:-}" || -n "${fail[${host}]:-}" ]] && continue
        log="/home/${host}/sparkdata/dsv4_pro.tp4pp4/smoke_rank${r}.log"
        if ssh -o BatchMode=yes -o ConnectTimeout=5 "${host}" "grep -q 'model_residentd ready' ${log} 2>/dev/null"; then
            ready[${host}]=1
        elif ! ssh -o BatchMode=yes -o ConnectTimeout=5 "${host}" "pgrep -f sparkpipe_model_residentd >/dev/null 2>&1"; then
            fail[${host}]=1
        else
            pending=$(( pending + 1 ))
        fi
    done
    [[ ${pending} -eq 0 ]] && break
    sleep 10
done

echo "== results"
status=0
for host in ${hosts}; do
    r=$(( 16#${host#spark} ))
    log="/home/${host}/sparkdata/dsv4_pro.tp4pp4/smoke_rank${r}.log"
    line=$(ssh -o BatchMode=yes "${host}" "grep -h 'model_residentd ready' ${log} 2>/dev/null | tail -1")
    if [[ -n "${line}" ]]; then
        echo "PASS rank=${r} ${host}: ${line}"
    else
        tail=$(ssh -o BatchMode=yes "${host}" "tail -2 ${log} 2>/dev/null | tr '\n' ' '")
        echo "FAIL rank=${r} ${host}: ${tail}"
        status=1
    fi
done

if [[ "${keep}" != "1" ]]; then
    echo "== stopping daemons (TERM)"
    for host in ${hosts}; do
        ssh -o BatchMode=yes "${host}" "pgrep -f sparkpipe_model_residentd >/dev/null && pkill -TERM -f sparkpipe_model_residentd || true" &
    done
    wait
fi
exit ${status}
