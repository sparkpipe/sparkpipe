#!/usr/bin/env bash
for i in 0 1 2 3; do
  n=$((i+4))
  ssh -o BatchMode=yes "spark$n" 'pkill -9 -f residentd_mx' &
done
wait
echo mx-daemons-cleared
