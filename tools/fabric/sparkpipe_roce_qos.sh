#!/bin/sh
set -eu
interface=enp1s0f1np1
ethtool -A "$interface" rx off tx off
mlnx_qos -i "$interface" --trust dscp
mlnx_qos -i "$interface" --pfc 0,0,0,1,0,0,0,0
mlnx_qos -i "$interface" --prio2buffer 0,0,0,1,0,0,0,0 --buffer_size 262144,459360,0,0,0,0,0,0
ethtool --set-tunable "$interface" pfc-prevention-tout 100
mlnx_qos -i "$interface" | grep -q "Priority trust state: dscp"
dcb pfc show dev "$interface" prio-pfc | grep -q "3:on"
dcb buffer show dev "$interface" prio-buffer | grep -q "3:1"
