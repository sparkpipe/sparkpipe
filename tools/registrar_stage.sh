#!/usr/bin/env bash
# registrar_stage.sh — stage the fleet startup registrar on the glm5_next
# fleet: compile the source ONCE on spark0 (aarch64 Linux, system cc), then
# copy the binary to every rank's runtime root. The registrar is pure POSIX
# C with zero library dependencies, so one node build serves the fleet.
#
# usage: tools/registrar_stage.sh [source.c]   (default tools/sparkpipe_registrar.c)
set -euo pipefail

REPO="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
SOURCE="${1:-$REPO/tools/sparkpipe_registrar.c}"
ALL_HOSTS=(spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7
           spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf)
PORT_BASE=22480

runtime_root() { echo "/home/$1/sparkdata/glm5_next.tp16"; }

echo "== compile on spark0 =="
scp -o BatchMode=yes "$SOURCE" spark0:/tmp/sparkpipe_registrar.c
rr="$(runtime_root spark0)"
ssh -o BatchMode=yes spark0 "cc -O2 -std=c11 -Wall -Wextra -D_POSIX_C_SOURCE=200809L -D_DEFAULT_SOURCE /tmp/sparkpipe_registrar.c -o '$rr/bin/sparkpipe_registrar.new' && chmod +x '$rr/bin/sparkpipe_registrar.new' && mv '$rr/bin/sparkpipe_registrar.new' '$rr/bin/sparkpipe_registrar'"

echo "== copy binary to fleet =="
for h in "${ALL_HOSTS[@]:1}"; do
    rr="$(runtime_root "$h")"
    scp -o BatchMode=yes -q "spark0:$rr/bin/sparkpipe_registrar" "$h:/tmp/sparkpipe_registrar"
    ssh -o BatchMode=yes "$h" "mv /tmp/sparkpipe_registrar '$rr/bin/sparkpipe_registrar' && chmod +x '$rr/bin/sparkpipe_registrar'"
    echo "$h: $(ssh -o BatchMode=yes "$h" "sha256sum < '$rr/bin/sparkpipe_registrar'" | cut -c1-16)"
done
echo "== staged on ${#ALL_HOSTS[@]} hosts =="
