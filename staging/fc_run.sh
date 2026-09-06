#!/bin/bash
R=$1
export PATH=/usr/bin:/bin
$HOME/fabric_check "$R" > /tmp/fc_run.log 2>&1
