#!/bin/bash
D=$1
P=$2
export PATH=/usr/bin:/bin
$HOME/ma_broker "$D" "$P" > /tmp/ma_broker_$P.log 2>&1
