#!/bin/bash
set -e
mkdir -p "$HOME/frb/sparkpipe"
cp "$HOME/spark_fixed_ring.h" "$HOME/frb/sparkpipe/"
cp "$HOME/spark_status.h" "$HOME/frb/sparkpipe/"
cp "$HOME/fixed_ring.c" "$HOME/frb/"
cp "$HOME/mock_allreduce2.c" "$HOME/frb/"
cd "$HOME/frb"
gcc -O2 -Wall -I. -o mock_allreduce2 fixed_ring.c mock_allreduce2.c -libverbs -lnng
md5sum mock_allreduce2
