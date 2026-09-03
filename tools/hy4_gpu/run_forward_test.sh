#!/bin/bash
# hy4 full-forward GPU test runner: internal redirects, disconnect-immune.
cd /home/spark2/hy4-gpu || exit 1
exec stdbuf -oL -eL ./hy4_forward_test /home/spark2/hy4-allranks 299 \
  >> /home/spark2/hy4-gpu/fwd_test_run.log 2>&1 < /dev/null
