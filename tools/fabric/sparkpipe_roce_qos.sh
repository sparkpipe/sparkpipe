#!/bin/sh
set -eu
interface=enp1s0f1np1
ethtool -A "$interface" rx off tx off
/usr/local/sbin/sparkpipe-dscp-trust set "$interface"
dcb pfc set dev "$interface" prio-pfc all:off 3:on
dcb buffer set dev "$interface" prio-buffer all:0 3:1 buffer-size all:0 0:262144 1:459360
ethtool --set-tunable "$interface" pfc-prevention-tout 100
/usr/local/sbin/sparkpipe-dscp-trust check "$interface"
dcb pfc show dev "$interface" prio-pfc | grep -q "3:on"
dcb buffer show dev "$interface" prio-buffer | grep -q "3:1"
