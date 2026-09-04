#!/bin/bash
# Light sweep: receipt files mark placed packs. Read-only, tiny I/O.
for n in spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf; do
  (timeout 25 ssh -o ConnectTimeout=8 "$n" "find /home /srv /data /raid /mnt/sparkdata -maxdepth 5 -name '*.receipt.json' -size +1k 2>/dev/null | grep -viE 'mirror|backup|stagepack-backup|archive' | head -40" 2>/dev/null | sed "s|^|$n: |") &
done
wait
