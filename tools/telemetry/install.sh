#!/bin/sh
set -eu

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SRC="$ROOT/tools/telemetry"
DEPLOY="$ROOT/deployment/telemetry"
DEST=${SPARKPIPE_TELEMETRY_DIR:-$HOME/.local/share/ds4_telemetry}
NODES=${SPARKPIPE_TELEMETRY_NODES:-"spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf"}
MODE=${1:-all}

install_mac() {
	mkdir -p "$DEST" /tmp/ds4_telemetry/mac
	for file in spark_telemetry_common.py spark_telemetry_collect.py spark_telemetry_dashboard.py; do
		install -m 0644 "$SRC/$file" "$DEST/$file"
	done
	install -m 0600 "$DEPLOY/telemetry_ssh_config" "$DEST/ssh_config"
	for label in com.ds4.spark-telemetry-collector com.ds4.spark-telemetry-dashboard; do
		target="$HOME/Library/LaunchAgents/$label.plist"
		sed -e "s|@DEST@|$DEST|g" -e "s|@HOME@|$HOME|g" "$DEPLOY/$label.plist" > "$target.tmp"
		plutil -lint "$target.tmp" > /dev/null
		mv "$target.tmp" "$target"
		launchctl bootout "gui/$(id -u)/$label" 2> /dev/null || true
		launchctl bootstrap "gui/$(id -u)" "$target"
	done
	echo "mac: collector and dashboard installed from $SRC"
}

install_nodes() {
	for node in $NODES; do
		ssh -o BatchMode=yes "$node" 'mkdir -p ~/bin ~/.config/systemd/user' < /dev/null
		scp -q "$SRC/spark_node_telemetry_monitor.py" "$SRC/spark_telemetry_stop.py" "$SRC/spark_telemetry_common.py" "$node:bin/"
		scp -q "$DEPLOY/ds4-spark-telemetry.service" "$node:.config/systemd/user/ds4-spark-telemetry.service"
		ssh -o BatchMode=yes "$node" 'systemctl --user daemon-reload && systemctl --user enable ds4-spark-telemetry.service > /dev/null 2>&1 && systemctl --user restart ds4-spark-telemetry.service && echo "$(hostname): $(systemctl --user is-active ds4-spark-telemetry.service)"' < /dev/null
	done
}

case "$MODE" in
	mac) install_mac ;;
	nodes) install_nodes ;;
	all) install_nodes; install_mac ;;
	*) echo "usage: $0 [mac|nodes|all]" >&2; exit 2 ;;
esac
