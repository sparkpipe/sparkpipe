#!/usr/bin/env bash
# g5dsa_wave_task.sh — glm5-dsa lane wave task (runs on spark0 via the queue
# dispatcher; holds all 16 nodes until done). Sequence: probe-vec wave
# (registrars -> clean slate -> 16 residentds -> ready -> api), wait for
# served:0, ONE canonical cold fixture request (case recNu3MXkvWUzHZr9,
# max_tokens 24, temperature 0), then preserve the rank-0 dump log.
#
# Dumps armed: SPARK_GLM5_NEXT_PROBE_VEC=1 (+PASSES) for the L0 KDA ladder
# and SPARK_GLM5_NEXT_PROBE_VEC_DSA=1 SPARK_GLM5_NEXT_PROBE_VEC_LAYER=3 for
# the glm5-dsa lane's DSA-site ladder (HC stages + the whole MLA chain).
set -uo pipefail
cd /home/spark0/g5rt2-src
export G5N_VEC_PASSES="${G5N_VEC_PASSES:-200}"
export G5N_VEC_DSA="${G5N_VEC_DSA:-1}"
export G5N_VEC_LAYER="${G5N_VEC_LAYER:-3}"

echo "== wave: probe-vec full (PASSES=$G5N_VEC_PASSES DSA=$G5N_VEC_DSA LAYER=$G5N_VEC_LAYER) =="
bash tools/glm5_next_wave.sh --probe-vec full || { echo "WAVE-FAIL"; exit 1; }

echo "== waiting for api health =="
health=""
for i in $(seq 1 60); do
  health=$(curl -s --max-time 3 http://localhost:8433/health || true)
  echo "health[$i]: $health"
  case "$health" in *'"status":"ok"'*) break ;; esac
  sleep 5
done
case "$health" in *'"status":"ok"'*) ;; *) echo "API-NEVER-READY"; exit 1 ;; esac

sleep 2
echo "== canonical cold fixture request (recNu3MXkvWUzHZr9, max_tokens 24, temperature 0) =="
curl -s --max-time 1200 -X POST http://localhost:8433/v1/completions \
  -H 'Content-Type: application/json' \
  -d @"$HOME/g5dsa_cold_request.json" | tee /tmp/g5dsa_cold_curl.json
echo
curl -s --max-time 5 http://localhost:8433/health
echo
cp /home/spark0/sparkdata/glm5_next.tp16/residentd.log /tmp/g5dsa_vec1.log
ls -la /tmp/g5dsa_vec1.log /tmp/g5dsa_cold_curl.json
echo TASK-DONE
