#!/usr/bin/env bash
# Static per-node launcher for the DSV4 Pro TP4xPP4 smoke. Copied to each
# deploy node's runtime root and executed there; derives everything from the
# node's own deployment. Env: SMOKE_RANK (default 0).
set -u
deploy="/home/$(hostname)/sparkdata/dsv4_pro.tp4pp4"
rank="${SMOKE_RANK:-0}"
log="${deploy}/smoke_rank${rank}.log"
cd "${deploy}" || exit 1

pids=$(pgrep -f sparkpipe_model_residentd || true)
if [[ -n "${pids}" ]]; then
    kill -TERM ${pids} 2>/dev/null || true
    for _ in $(seq 1 20); do
        pgrep -f sparkpipe_model_residentd >/dev/null || break
        sleep 1
    done
fi

rm -f "${log}"
nohup ./bin/sparkpipe_model_residentd \
    --deployment config/model_resident.json \
    --rank-index "${rank}" > "${log}" 2>&1 &
echo "launched rank=${rank} pid=$! host=$(hostname)"
