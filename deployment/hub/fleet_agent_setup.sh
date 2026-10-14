#!/usr/bin/env bash
set -euo pipefail
AGENT=sparkagent
KEY=/srv/workspace/sparkpipe/fleet-gateway/agent_ed25519
NODES="spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf"
if [ ! -f "$KEY" ]; then
  ssh-keygen -q -t ed25519 -N '' -C "$AGENT@sparkpipe-fleet-gateway" -f "$KEY"
fi
chmod 600 "$KEY"
PUB=$(cat "$KEY.pub")
read -r -d '' SETUP <<'REMOTE' || true
set -eu
if ! id -u sparkagent >/dev/null 2>&1; then
  useradd --create-home --shell /bin/bash --comment "SparkPipe Chat fleet agent, no sudo" sparkagent
fi
for group in sudo admin wheel docker adm; do gpasswd -d sparkagent "$group" >/dev/null 2>&1 || true; done
chmod 750 /home/sparkagent
install -d -m 700 -o sparkagent -g sparkagent /home/sparkagent/.ssh
printf '%s\n' "$PUB" > /home/sparkagent/.ssh/authorized_keys
chown sparkagent:sparkagent /home/sparkagent/.ssh/authorized_keys
chmod 600 /home/sparkagent/.ssh/authorized_keys
printf 'sparkagent hard nproc 256\nsparkagent hard nofile 4096\nsparkagent hard fsize 1048576\n' > /etc/security/limits.d/90-sparkagent.conf
echo "$(hostname): sparkagent uid $(id -u sparkagent), groups: $(id -Gn sparkagent)"
REMOTE
echo "== rtx5090"
sudo -n env PUB="$PUB" bash -s <<< "$SETUP"
for node in $NODES; do
  echo "== $node"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$node" "sudo -n env PUB='$PUB' bash -s" <<< "$SETUP" || echo "$node: setup FAILED"
done
echo "== checks (each line should say no-sudo)"
sudo -n -H -u "$AGENT" -- bash -lc 'echo "$(hostname): $(id -un) $(sudo -n true 2>/dev/null && echo HAS-SUDO || echo no-sudo)"' || echo "rtx5090: check FAILED"
for node in $NODES; do
  ssh -o BatchMode=yes -o ConnectTimeout=8 -i "$KEY" -o IdentitiesOnly=yes -l "$AGENT" "$node" 'echo "$(hostname): $(id -un) $(sudo -n true 2>/dev/null && echo HAS-SUDO || echo no-sudo)"' || echo "$node: check FAILED"
done
