#!/bin/sh
cd /home/sparke/build-kernels-w5
O=results/$1
shift
mkdir -p $O
: > $O/ab.txt
nvidia-smi --query-gpu=utilization.gpu,clocks.sm,power.draw --format=csv,noheader > $O/gpu_before.txt 2>&1
for round in $(seq 1 ${ROUNDS:-5}); do
  for v in "$@"; do
    echo "round=$round variant=$v" >> $O/ab.txt
    ./bin/$v --batches 1,8 --context 1024 --iterations 100 --graph 1 --route-readback 0 2>&1 | grep -E "ROOFLINE-GRAPH|hidden_finite" >> $O/ab.txt
  done
done
python3 - $O/ab.txt <<'PY'
import sys,re,statistics as st
d={}
v=None
for l in open(sys.argv[1]):
    m=re.match(r'round=\d+ variant=(\S+)',l)
    if m: v=m.group(1); continue
    m=re.search(r'ROOFLINE-GRAPH rows=(\d+).*median_ms=([\d.]+) min_ms=([\d.]+)',l)
    if m: d.setdefault((v,int(m.group(1))),[]).append((float(m.group(2)),float(m.group(3))))
for (v,r),xs in sorted(d.items(),key=lambda x:(x[0][1],x[0][0])):
    med=[x[0] for x in xs]; mn=[x[1] for x in xs]
    print("AB rows=%d variant=%s n=%d median_of_medians=%.3f stdev=%.3f min=%.3f medians=%s"%(r,v,len(med),st.median(med),st.pstdev(med),min(mn),",".join("%.3f"%x for x in med)))
PY
