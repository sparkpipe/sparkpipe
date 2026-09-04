#!/bin/bash
# meshbench rank-0 root runner (systemd-run invokes this; survives ssh exit).
# args: nranks iters bytes burst dual
NR=$1; I=$2; B=$3; BU=$4; D=$5
export NCCL_SOCKET_IFNAME=enp1s0f1np1
export NCCL_IB_HCA=rocep1s0f1
export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=WARN
export LD_LIBRARY_PATH="$HOME/nccl-src/build/lib:$LD_LIBRARY_PATH"
[ -f /tmp/mb_extra.env ] && . /tmp/mb_extra.env
rm -f /tmp/nccl_id.bin
exec stdbuf -oL "$HOME/nccl_burst" root 0 "$NR" "$I" "$B" "$BU" "$D" > /tmp/mb_root.log 2>&1
