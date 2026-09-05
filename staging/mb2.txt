#!/bin/bash
D=$1
P=$2
E=${3:-128}
export PATH=/usr/bin:/bin
$HOME/ma_broker "$D" "$P" "$E" > /tmp/ma_broker_$P.log 2>&1
