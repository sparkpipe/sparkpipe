#!/bin/bash
cd "$HOME/hy4-gpu" || exit 1
export HY4_DUMP_PFX="$HOME/hy4-anchor/gpu/gpu"
exec stdbuf -oL -eL ./hy4_forward_test "$HOME/hy4-allranks" -1 \
  >> "$HOME/hy4-gpu/fwd_anchor.log" 2>&1 < /dev/null
