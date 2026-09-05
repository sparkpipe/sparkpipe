#!/bin/bash
R=$1
export PATH=/usr/bin:/bin
$HOME/mock_allreduce "$R" > /tmp/ma_run.log 2>&1
