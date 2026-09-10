#!/bin/bash
set -uo pipefail
cd $HOME/dsv4pro_checkout
mkdir -p /home/spark6/sparkdata/dsv4_pro.probe /mnt/model-warm/packbuild/dsv4pro
LOG=/mnt/model-warm/packbuild/dsv4pro/probe_pack.log
exec > "$LOG" 2>&1
export PATH=/usr/local/cuda/bin:$PATH
python3 tools/dsv4_pro_stagepack.py \
  --model-dir /mnt/model-warm/deepseek-v4-pro-0813-ga \
  --output /home/spark6/sparkdata/dsv4_pro.probe/probe_l0_3.spstage \
  --first-layer 0 --layer-count 3
rc=$?
echo "PROBE-PACK-RC=$rc"
[ $rc -eq 0 ] && python3 tools/dsv4_pro_stagepack.py --verify-pack /home/spark6/sparkdata/dsv4_pro.probe/probe_l0_3.spstage
[ $rc -eq 0 ] && sha256sum /home/spark6/sparkdata/dsv4_pro.probe/probe_l0_3.spstage | awk '{print $1}' > /home/spark6/sparkdata/dsv4_pro.probe/probe_l0_3.spstage.sha256
exit $rc
