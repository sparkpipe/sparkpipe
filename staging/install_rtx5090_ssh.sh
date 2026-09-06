#!/bin/bash
# install_rtx5090_ssh.sh — prepend a Host rtx5090 block (ProxyJump sparkf,
# user spec) to ~/.ssh/config on every spark, once, with a .bak snapshot.
set -uo pipefail
HOSTS=(spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7
       spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf)
BLOCK='Host rtx5090
  HostName 10.10.250.2
  User spec
  ProxyJump sparkf
  BatchMode yes
  StrictHostKeyChecking accept-new
'
for h in "${HOSTS[@]}"; do
    ssh -o BatchMode=yes -o ConnectTimeout=5 "$h" "
        cd ~/.ssh || exit 1
        if ! grep -q '^Host rtx5090' config 2>/dev/null; then
            [ -f config ] && cp config config.bak
            printf '%s\n' '$BLOCK' > config.new
            [ -f config.bak ] && cat config.bak >> config.new
            mv config.new config
        fi
        ssh -o BatchMode=yes -o ConnectTimeout=8 rtx5090 hostname 2>/dev/null | tail -1
    " 2>/dev/null && echo " <- $h" || echo "$h: FAIL"
done
