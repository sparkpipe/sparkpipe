#!/bin/sh
set -eu

REPO=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BIN="$REPO/build/sparkpipe_weightsd"
CTL="$REPO/build/weightdctl"
SOCKET_DEFAULT=/run/sparkpipe-weightsd/weightsd.sock
ENV_FILE=/etc/sparkpipe/weightsd.env
UNIT=/etc/systemd/system/sparkpipe_weightsd.service

make -j "$(nproc)" -C "$REPO" build/sparkpipe_weightsd build/weightdctl
test -x "$BIN"
test -x "$CTL"

if ! id -un 1000 >/dev/null 2>&1; then
	echo "weightsd-deploy: no uid-1000 account on $(uname -n)" >&2
	exit 1
fi
RUN_USER=$(id -un 1000)

if [ ! -f "$ENV_FILE" ]; then
	printf 'WEIGHTSD_BIN=%s\nWEIGHTSD_SOCKET=%s\nWEIGHTSD_EXTRA_ARGS=\n' \
		"$BIN" "$SOCKET_DEFAULT" | sudo -n tee "$ENV_FILE" >/dev/null
else
	sudo -n sed -i "s#^WEIGHTSD_BIN=.*#WEIGHTSD_BIN=$BIN#" "$ENV_FILE"
	sudo -n grep -q '^WEIGHTSD_SOCKET=' "$ENV_FILE"
fi
SOCKET=$(sudo -n sed -n 's/^WEIGHTSD_SOCKET=//p' "$ENV_FILE")
test -n "$SOCKET"

sudo -n install -m 0644 "$REPO/tools/devcycle/sparkpipe_weightsd.service" "$UNIT"
sudo -n mkdir -p /etc/systemd/system/sparkpipe_weightsd.service.d
printf '[Service]\nUser=%s\n' "$RUN_USER" \
	| sudo -n tee /etc/systemd/system/sparkpipe_weightsd.service.d/10-user.conf >/dev/null
sudo -n systemctl daemon-reload
sudo -n systemctl enable sparkpipe_weightsd >/dev/null

STATE=$(sudo -n systemctl is-active sparkpipe_weightsd || true)
if [ "$STATE" = active ] && [ "${WEIGHTSD_DEPLOY_RESTART:-0}" != 1 ]; then
	echo "weightsd-deploy: service already active; refusing restart (set WEIGHTSD_DEPLOY_RESTART=1 for an announced upgrade)"
else
	sudo -n systemctl reset-failed sparkpipe_weightsd 2>/dev/null || true
	sudo -n systemctl restart sparkpipe_weightsd
fi

i=0
while [ "$i" -lt 200 ]; do
	if [ -S "$SOCKET" ] && [ "$(sudo -n systemctl is-active sparkpipe_weightsd)" = active ]; then
		break
	fi
	if sudo -n systemctl is-failed --quiet sparkpipe_weightsd; then
		sudo -n systemctl status sparkpipe_weightsd --no-pager -l | tail -5 >&2
		echo "weightsd-deploy: unit failed on $(uname -n)" >&2
		exit 1
	fi
	i=$((i + 1))
	sleep 0.1
done
if [ "$i" -ge 200 ]; then
	echo "weightsd-deploy: socket $SOCKET never appeared on $(uname -n)" >&2
	exit 1
fi

HEALTH=$(SPARK_WEIGHTD_SOCKET="$SOCKET" "$CTL" reclaim)
SHA=$(sha256sum "$BIN" | cut -c1-16)
echo "weightsd-deploy: node=$(uname -n) user=$RUN_USER sha16=$SHA socket=$SOCKET health=\"$HEALTH\""
