#!/bin/bash
# dsv4flash TP16 cell1b: relaunch with FIXED stage configs (graphs 130 +
# full tp_collective peer arrays/algorithms/thresholds). Reuses cell1's
# published module, compiled driver, and staged runtime binaries; only
# restages configs, then launch -> exactness gate x3 -> TERM.
set -uo pipefail
BASE=/home/spark5/lane-dsv4flash-m1
CFG=$BASE/tp16-configs
RANK_HOST="spark0 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf spark1"
rank_of() { case $1 in spark0)echo 0;; spark1)echo 15;; spark2)echo 1;; spark3)echo 2;; spark4)echo 3;; spark5)echo 4;; spark6)echo 5;; spark7)echo 6;; spark8)echo 7;; spark9)echo 8;; sparka)echo 9;; sparkb)echo 10;; sparkc)echo 11;; sparkd)echo 12;; sparke)echo 13;; sparkf)echo 14;; esac; }
rt_of() { echo "/home/$1/sparkdata/dsv4flash.tp16/rt"; }
term_all() {
  for h in $RANK_HOST; do
    ssh -o BatchMode=yes $h "for p in \$(pgrep -f 'bin/sparkpipe_model_[r]esidentd' 2>/dev/null); do [ \"\$(readlink /proc/\$p/cwd)\" = \"\$(rt_of $h)\" ] && kill -TERM \$p; done; true"
  done
}
[ -f $BASE/driver-tp16/model_driver.so ] || { echo "NO-DRIVER (cell1 compile missing)"; exit 1; }

echo "== [1/3] restage fixed configs"
for h in $RANK_HOST; do
  rt=$(rt_of $h)
  ssh -o BatchMode=yes $h "mkdir -p $rt/config $rt/kv"
  if [ "$h" != "spark5" ]; then
    rsync -aq -e "ssh -o BatchMode=yes" $CFG/config/stage.json.$h $h:$rt/config/stage.json &
    rsync -aq -e "ssh -o BatchMode=yes" $CFG/config/model_resident.json $h:$rt/config/ &
  else
    cp $CFG/config/stage.json.$h $rt/config/stage.json
    cp $CFG/config/model_resident.json $rt/config/
  fi
done
wait
echo CONFIGS-STAGED

echo "== [2/3] TERM stragglers + drop caches + launch"
term_all
sleep 3
for h in $RANK_HOST; do
  r=$(rank_of $h); rt=$(rt_of $h)
  ssh -o BatchMode=yes $h "sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; cd $rt && rm -f residentd.log && nohup ./bin/sparkpipe_model_residentd --deployment config/model_resident.json --rank-index $r > residentd.log 2>&1 < /dev/null & echo launched-$h-r$r" &
done
wait

echo "== [3/3] ready-wait (8 min cap; warm packs) then O128 x3"
FAIL=0
for h in $RANK_HOST; do
  rt=$(rt_of $h); ok=""
  for i in $(seq 1 48); do
    if ssh -o BatchMode=yes $h "grep -q 'model_residentd ready' $rt/residentd.log 2>/dev/null"; then ok=1; echo "$h READY"; break; fi
    if ! ssh -o BatchMode=yes $h "pgrep -f 'bin/sparkpipe_model_[r]esidentd' >/dev/null"; then echo "$h DIED:"; ssh -o BatchMode=yes $h "grep -m3 'schema_error\|status=\|free()\|error' $rt/residentd.log | head -3"; FAIL=1; break; fi
    sleep 10
  done
  [ -n "$ok" ] || { [ $FAIL -eq 0 ] && { echo "$h TIMEOUT"; FAIL=1; }; }
done
[ $FAIL -eq 0 ] || { echo READY-FAIL; exit 1; }

trap term_all EXIT
for i in 1 2 3; do
  ssh -o BatchMode=yes spark0 "cd /home/spark0/sparkdata/dsv4flash.tp16/rt && ./bin/sparkpipe_model_batch --deployment config/model_resident.json --runtime-root /home/spark0/sparkdata/dsv4flash.tp16/rt --batch /home/spark5/lane-dsv4flash-m1/staging/devcycle-o128-batch.json" 2> $BASE/results/tp16_cell1b_run$i.stderr | python3 -u -c "
import sys, time
for line in sys.stdin:
    sys.stdout.write(f'{time.time():.6f} {line}')
    sys.stdout.flush()" > $BASE/results/tp16_cell1b_run$i.jsonl
done
python3 << 'PY'
import json
base = "/home/spark5/lane-dsv4flash-m1/results/tp16_cell1b_run"
exp = json.load(open("/home/spark5/lane-dsv4flash-m1/tp16-configs/expected_o128_tokens.json"))
for i in (1, 2, 3):
    ev = []
    for raw in open(base + str(i) + ".jsonl"):
        ts, _, payload = raw.partition(" ")
        try:
            ev.append((float(ts), json.loads(payload)))
        except Exception:
            pass
    toks = sorted([e for e in ev if e[1].get("event") == "TOKEN" and e[1].get("status") == 0],
                  key=lambda p: p[1]["token_index"])
    ids = [e[1]["token_id"] for e in toks]
    if not ids:
        print(f"run{i}: NO-TOKENS")
        continue
    if ids == exp:
        span = toks[-1][0] - toks[0][0]
        print(f"run{i}: EXACT tokens={len(ids)} decode_tok_s={(len(ids) - 1) / span:.3f} span_s={span:.3f}")
    else:
        n = min(len(ids), len(exp))
        first = next((k for k in range(n) if ids[k] != exp[k]), n)
        print(f"run{i}: MISMATCH len={len(ids)} first_divergence={first}")
PY
echo CELL1B-DONE
