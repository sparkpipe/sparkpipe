#!/bin/sh
# Deploy the shared weightd unit on one Spark (run on the node itself, via sudo).
# Usage: weightd_shared_deploy.sh <rank> <user>
# Stages the release bundle (sha 097165339f969bd57d0409092684ffd8f3237af96dc617534cdb7639772194ca,
# sparkpipe-shared-serving-linux-arm64-cuda13.tar.gz) at ~/sparkpipe/shared-serving-20260922,
# renders the unit with the node's rank/user, enables + starts it, and verifies the socket.
set -eu
RANK="$1"; USER_="$2"
mkdir -p "$HOME/sparkpipe/shared-serving-20260922"
tar xzf /tmp/sp-bundle.tar.gz -C "$HOME/sparkpipe/shared-serving-20260922"
sed -e "s/%i/$USER_/g" -e "s/%k/$RANK/g" tools/devcycle/sparkpipe-weightd-shared.service | \
  sudo tee /etc/systemd/system/sparkpipe-weightd-shared.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now sparkpipe-weightd-shared.service
sleep 3
systemctl is-active sparkpipe-weightd-shared.service
ls -la /run/sparkpipe-weightd-shared/weightd.sock
