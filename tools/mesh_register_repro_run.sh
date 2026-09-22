#!/bin/bash
# mesh-register repro runner (lane 0): compile and run the CUDA host-register
# probe under the queue cgroup. Queue cmd: bash tools/mesh_register_repro_run.sh
set -euo pipefail
CHECKOUT="$(pwd)"
cc -O2 -I "$CHECKOUT/include" -I /usr/local/cuda/include "$CHECKOUT/tools/mesh_register_repro.c" \
   -o /tmp/mesh_register_repro.$$ -L/usr/local/cuda/lib64 -lcudart
trap 'rm -f /tmp/mesh_register_repro.$$' EXIT
/tmp/mesh_register_repro.$$
