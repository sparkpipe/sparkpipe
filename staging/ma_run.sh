#!/bin/bash
R=$1
D=$2
S=$3
P=$4
export PATH=/usr/bin:/bin
$HOME/mock_allreduce "$R" "$D" "$S" "$P" > /tmp/ma_run_$P.log 2>&1
