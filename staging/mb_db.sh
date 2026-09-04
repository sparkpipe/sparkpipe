#!/bin/bash
# doorbell bench runner: args rank iters rows mode
R=$1; I=$2; ROWS=$3; MODE=$4
export SPARK_TP_COLLECTIVE_PROFILE=${MB_PROFILE:-0}
export SPARKPIPE_HIDDEN_SPARK_HOST_RDMA_LANES=${MB_LANES:-1}
exec stdbuf -oL "$HOME/mb_doorbell" "$R" 16 "$I" "$ROWS" "$MODE" \
    "$HOME/mb_transport/hidden_transport.so" \
    > "/tmp/mb_db_r${R}.log" 2>&1
