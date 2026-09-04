#!/bin/bash
Q="/Users/mac/sparkpipe/tools/spark_queue.py"
python3 "$Q" add \
  --id glm53-fixedbench4 \
  --nodes spark9,spark0,spark1,spark2,spark3,spark4,spark6,spark7,spark8,sparka,sparkb,sparkc,sparkd,sparke,sparkf \
  --cmd-file /Users/mac/lane-glm53/staging/ring28.cmd \
  --resources gpu --priority 0 --klass short --ttl-min 12 \
  --by glm53flash \
  --notes "clean DSO rebuild - fixed slot bench"
python3 "$Q" dispatch
