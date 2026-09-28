#!/bin/sh
D=$(dirname "$0")
cd "$D"
O=results/$1
mkdir -p $O
: > $O/ab.txt
for round in 1 2 3 4 5 6; do
  for rb in 1 0; do
    echo "round=$round" >> $O/ab.txt
    ./glm5_next_batch_roofline --batches 1,8 --context 1024 --iterations 100 --graph 1 --route-readback $rb 2>&1 | grep -E "ROOFLINE-GRAPH|^ROOFLINE rows" >> $O/ab.txt
  done
done
grep -c GRAPH $O/ab.txt
