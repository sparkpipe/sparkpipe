#!/bin/sh
# Deploy the shared weightd unit on one Spark (run on the node itself, via sudo).
# Usage: weightd_shared_deploy.sh <rank> <user>
# Stages the release bundle (sha 097165339f969bd57d0409092684ffd8f3237af96dc617534cdb7639772194ca,
# sparkpipe-shared-serving-linux-arm64-cuda13.tar.gz) at ~/sparkpipe/shared-serving-20260922,
# renders the unit with the node's rank/user, enables + starts it, and verifies the socket.
set -eu
RANK="$1"; USER_="$2"
if systemctl is-active --quiet sparkpipe-weightd-shared.service; then
    printf '%s\n' 'Drain model jobs, stop the shared service and untrack it before replacing artifacts.' >&2
    exit 1
fi
printf '%s  %s\n' 097165339f969bd57d0409092684ffd8f3237af96dc617534cdb7639772194ca /tmp/sp-bundle.tar.gz | sha256sum --check --strict
if [ -d "$HOME/sparkpipe/shared-serving-20260922" ]; then
    (cd "$HOME/sparkpipe/shared-serving-20260922" && sha256sum --quiet --strict --check SHA256SUMS)
fi
mkdir -p "$HOME/sparkpipe/shared-serving-20260922"
tar xzf /tmp/sp-bundle.tar.gz -C "$HOME/sparkpipe/shared-serving-20260922"
(cd "$HOME/sparkpipe/shared-serving-20260922" && sha256sum --quiet --strict --check SHA256SUMS)
sed -e "s/%i/$USER_/g" -e "s/%k/$RANK/g" tools/devcycle/sparkpipe-weightd-shared.service | \
  sudo tee /etc/systemd/system/sparkpipe-weightd-shared.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now sparkpipe-weightd-shared.service
sleep 3
systemctl is-active sparkpipe-weightd-shared.service
ls -la /run/sparkpipe-weightd-shared/weightd.sock
