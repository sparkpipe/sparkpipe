#!/bin/bash
set -u
OUT=/tmp/glm_ab.json
rm -f $OUT
python3 ~/g4bench.py 8433 o128 2 $OUT glm-B1-gemma-idle
python3 ~/g4bench.py 8436 streams8 1 /tmp/glm_ab_gemma.json gemma-8stream-load > /tmp/glm_ab_gemma.jsonl &
G=$!
sleep 10
python3 ~/g4bench.py 8433 o128 2 $OUT glm-B1-gemma-8stream-busy
wait $G
cat /tmp/glm_ab_gemma.jsonl
