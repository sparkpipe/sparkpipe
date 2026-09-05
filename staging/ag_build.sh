#!/bin/bash
set -e
mkdir -p "$HOME/agb/sparkpipe"
cp "$HOME/spark_fixed_ring.h" "$HOME/agb/sparkpipe/" 2>/dev/null || true
cp "$HOME/spark_status.h" "$HOME/agb/sparkpipe/"
cp "$HOME/mock_allgather.c" "$HOME/agb/"
cd "$HOME/agb"
gcc -O2 -Wall -o mock_allgather mock_allgather.c -libverbs -lnng
md5sum mock_allgather
