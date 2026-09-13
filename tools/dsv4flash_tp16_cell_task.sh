#!/bin/bash
# dsv4flash TP16 B1 O128 cell: publish b1 (GPU validator) -> compile driver ->
# stage 16-node runtime -> launch -> exactness gate x3 -> TERM.
set -uo pipefail
SRC=/home/spark5/lane-dsv4flash-m1/src2
BASE=/home/spark5/lane-dsv4flash-m1
CFG=$BASE/tp16-configs
VALSLICE=/home/spark5/lane-dsv4bisect/packs/dsv4_flash_v4_val3.spstage
RANK_HOST="spark0 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf spark1"
rank_of() { case $1 in spark0)echo 0;; spark1)echo 15;; spark2)echo 1;; spark3)echo 2;; spark4)echo 3;; spark5)echo 4;; spark6)echo 5;; spark7)echo 6;; spark8)echo 7;; spark9)echo 8;; sparka)echo 9;; sparkb)echo 10;; sparkc)echo 11;; sparkd)echo 12;; sparke)echo 13;; sparkf)echo 14;; esac; }
rt_of() { echo "/home/$1/sparkdata/dsv4flash.tp16/rt"; }
term_all() {
  for h in $RANK_HOST; do
    ssh -o BatchMode=yes $h "for p in \$(pgrep -f 'bin/sparkpipe_model_[r]esidentd' 2>/dev/null); do [ \"\$(readlink /proc/\$p/cwd)\" = \"\$(rt_of $h)\" ] && kill -TERM \$p; done; true"
  done
}

echo "== [1/6] publish b1 module (GPU validator)"
cd $SRC
make -C modules/dsv4_resident_decode_stage publish_variants MODULE_BATCH_VARIANT_BUCKETS=1 CUDA_ARCH=sm_121a \
  STAGE_PACK_PATH=$VALSLICE STAGE_COUNT=13 STAGE_INDEX=1 STAGE_FIRST_LAYER=3 STAGE_LAYER_COUNT=3 \
  MAX_ACTIVE_SEQUENCES=1 PIPELINE_SLOT_COUNT=1 2>&1 | grep -E "validation|module=|error|FAIL" | tail -4
[ ${PIPESTATUS[0]} -eq 0 ] || { echo PUBLISH-FAIL; exit 1; }

echo "== [2/6] compile driver"
if [ -d $BASE/driver-tp16 ]; then rm -r $BASE/driver-tp16; fi
build/sparkpipe_model_compile --model examples/model_descriptions/dsv4_resident_decode_stage_firmware_b1.json \
  --stage dsv4_resident_decode_stage --library build/module_library --output $BASE/driver-tp16 --include include \
  --cc-arg -L/usr/local/cuda/lib64 --cc-arg -lcuda --cc-arg -lcudart --cc-arg -lstdc++ --cc-arg -ldl --cc-arg -lm --cc-arg -pthread \
  2>&1 | tail -1
[ -f $BASE/driver-tp16/model_driver.so ] || { echo COMPILE-FAIL; exit 1; }
sha256sum $BASE/driver-tp16/model_driver.so

echo "== [3/6] stage runtime to 16 nodes"
for h in $RANK_HOST; do
  r=$(rank_of $h); rt=$(rt_of $h)
  ssh -o BatchMode=yes $h "mkdir -p $rt/bin $rt/lib $rt/config $rt/packs $rt/kv"
  if [ "$h" != "spark5" ]; then
    rsync -aq -e "ssh -o BatchMode=yes" $SRC/build/sparkpipe_model_residentd $SRC/build/sparkpipe_model_batch $h:$rt/bin/ &
    rsync -aq -e "ssh -o BatchMode=yes" $SRC/build/libdsv4_tp16_serving_adapter.so $h:$rt/lib/model_serving_adapter.so &
    rsync -aq -e "ssh -o BatchMode=yes" $SRC/build/libhidden_transport_spark_host_rdma_verbs.so $h:$rt/lib/hidden_transport.so &
    rsync -aq -e "ssh -o BatchMode=yes" $CFG/config/stage.json.$h $h:$rt/config/stage.json &
    rsync -aq -e "ssh -o BatchMode=yes" $CFG/config/model_resident.json $h:$rt/config/ &
  else
    cp $SRC/build/sparkpipe_model_residentd $SRC/build/sparkpipe_model_batch $rt/bin/
    cp $SRC/build/libdsv4_tp16_serving_adapter.so $rt/lib/model_serving_adapter.so
    cp $SRC/build/libhidden_transport_spark_host_rdma_verbs.so $rt/lib/hidden_transport.so
    cp $CFG/config/stage.json.$h $rt/config/stage.json
    cp $CFG/config/model_resident.json $rt/config/
  fi
  ssh -o BatchMode=yes $h "ln -sf /home/$h/sparkdata/dsv4flash.tp16/packs/dsv4flash.tp16.rank$r.spstage $rt/packs/dsv4flash.tp16.rank$r.spstage; ln -sf /home/$h/sparkdata/glm5_next.tp16/lib/libnccl.so.2 $rt/lib/libnccl.so.2" &
done
wait
for h in $RANK_HOST; do
  rt=$(rt_of $h)
  if [ "$h" != "spark5" ]; then rsync -aq -e "ssh -o BatchMode=yes" $BASE/driver-tp16/model_driver.so $h:$rt/lib/ & else cp $BASE/driver-tp16/model_driver.so $rt/lib/; fi
done
wait
echo STAGED

echo "== [4/6] drop caches + launch 16 residentd"
for h in $RANK_HOST; do
  r=$(rank_of $h); rt=$(rt_of $h)
  ssh -o BatchMode=yes $h "sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; cd $rt && rm -f residentd.log && nohup ./bin/sparkpipe_model_residentd --deployment config/model_resident.json --rank-index $r > residentd.log 2>&1 < /dev/null & echo launched-$h-r$r" &
done
wait

echo "== [5/6] ready-wait (max 25 min)"
FAIL=0
for h in $RANK_HOST; do
  rt=$(rt_of $h); ok=""
  for i in $(seq 1 150); do
    if ssh -o BatchMode=yes $h "grep -q 'model_residentd ready' $rt/residentd.log 2>/dev/null"; then ok=1; echo "$h READY"; break; fi
    if ! ssh -o BatchMode=yes $h "pgrep -f 'bin/sparkpipe_model_[r]esidentd' >/dev/null"; then echo "$h DIED:"; ssh -o BatchMode=yes $h "tail -8 $rt/residentd.log"; FAIL=1; break; fi
    sleep 10
  done
  [ -n "$ok" ] || { [ $FAIL -eq 0 ] && { echo "$h TIMEOUT"; FAIL=1; }; }
done
[ $FAIL -eq 0 ] || { echo READY-FAIL; exit 1; }

echo "== [6/6] O128 cell x3, exactness gate before timing"
trap term_all EXIT
for i in 1 2 3; do
  ssh -o BatchMode=yes spark0 "cd /home/spark0/sparkdata/dsv4flash.tp16/rt && ./bin/sparkpipe_model_batch --deployment config/model_resident.json --runtime-root /home/spark0/sparkdata/dsv4flash.tp16/rt --batch /home/spark5/lane-dsv4flash-m1/staging/devcycle-o128-batch.json" 2> $BASE/results/tp16_cell_run$i.stderr | python3 -u -c "
import sys, time
for line in sys.stdin:
    sys.stdout.write(f'{time.time():.6f} {line}')
    sys.stdout.flush()" > $BASE/results/tp16_cell_run$i.jsonl
done
python3 << 'PY'
import json
base = "/home/spark5/lane-dsv4flash-m1/results/tp16_cell_run"
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
echo CELL-DONE
