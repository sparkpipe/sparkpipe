#!/bin/bash
# meshbench per-rank runner (systemd-run invokes this; survives ssh exit).
# args: rank nranks iters bytes burst dual idfile tag
R=$1; NR=$2; I=$3; B=$4; BU=$5; D=$6; ID=$7; TAG=$8
export NCCL_SOCKET_IFNAME=enp1s0f1np1
export NCCL_IB_HCA=rocep1s0f1
export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=WARN
export LD_LIBRARY_PATH="$HOME/nccl-src/build/lib:$LD_LIBRARY_PATH"
[ -f /tmp/mb_extra.env ] && . /tmp/mb_extra.env
exec stdbuf -oL "$HOME/nccl_burst" "$R" "$NR" "$I" "$B" "$BU" "$D" "$ID" > "/tmp/mb_${TAG}_r${R}.log" 2>&1
