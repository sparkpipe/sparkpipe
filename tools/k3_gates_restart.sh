#!/usr/bin/env bash
# Restart the k3 offline-gates run on the personal build node (spark7).
# Exact-name TERM (never pattern-match the ssh wrapper), clean build dir,
# detached relaunch.
set -u
pkill -x make 2>/dev/null && sleep 2
pkill -x cc 2>/dev/null
rm -rf /home/spark7/k3gate/build /home/spark7/k3gate/gates.log
cd /home/spark7/k3gate || exit 1
nohup make offline-gates > gates.log 2>&1 < /dev/null &
echo "relaunched pid $!"
