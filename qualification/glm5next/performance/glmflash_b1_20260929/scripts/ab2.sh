#!/bin/sh
cd /home/sparkf/build-glmflash-b1
O=results/$1
shift
mkdir -p $O
: > $O/ab.txt
nvidia-smi --query-gpu=utilization.gpu,clocks.sm,power.draw --format=csv,noheader > $O/gpu_before.txt 2>&1
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >> $O/gpu_before.txt 2>&1
for round in $(seq 1 ${ROUNDS:-5}); do
  for spec in "$@"; do
    name=${spec%%=*}
    rest=${spec#*=}
    bin=${rest%%,*}
    flags=$(echo "${rest#*,}" | tr ',' ' ')
    [ "$bin" = "$rest" ] && flags=""
    echo "round=$round variant=$name" >> $O/ab.txt
    ./bin/$bin --batches ${BATCHES:-1,8} --context 1024 --iterations 100 --graph 1 --route-readback 0 $flags 2>&1 | grep -E "ROOFLINE-GRAPH|ROOFLINE-HASH|ROOFLINE-FAIL" >> $O/ab.txt
  done
done
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > $O/gpu_after.txt 2>&1
python3 - $O/ab.txt <<'PY'
import sys,re,statistics as st
d={};h={};v=None
for l in open(sys.argv[1]):
    m=re.match(r'round=\d+ variant=(\S+)',l)
    if m: v=m.group(1); continue
    m=re.search(r'ROOFLINE-GRAPH rows=(\d+).*median_ms=([\d.]+) min_ms=([\d.]+)',l)
    if m: d.setdefault((v,int(m.group(1))),[]).append((float(m.group(2)),float(m.group(3))))
    m=re.search(r'ROOFLINE-HASH rows=(\d+) (.*)',l)
    if m: h.setdefault((v,int(m.group(1))),set()).add(m.group(2).strip())
    if 'FAIL' in l: print(v,l.strip())
for (v,r),xs in sorted(d.items(),key=lambda x:(x[0][1],x[0][0])):
    med=[x[0] for x in xs]; mn=[x[1] for x in xs]
    print("AB rows=%d variant=%s n=%d median_of_medians=%.3f stdev=%.3f min=%.3f medians=%s"%(r,v,len(med),st.median(med),st.pstdev(med),min(mn),",".join("%.3f"%x for x in med)))
for k,s in sorted(h.items(),key=lambda x:(x[0][1],x[0][0])):
    print("HASH rows=%d variant=%s %s"%(k[1],k[0]," | ".join(sorted(s))))
PY
