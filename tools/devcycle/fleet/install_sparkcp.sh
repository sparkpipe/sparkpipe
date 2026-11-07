#!/bin/sh
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
TOOL="$HERE/../../sparkcp.py"
SPARKS="spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf"
LENOVOS="lenovo7 lenovo8 lenovo9"
HUB="rtx5090"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=8"
BEGIN="# BEGIN cisco primary path"
END="# END cisco primary path"

install_tool() {
	want="$(shasum -a 256 "$TOOL" | cut -d' ' -f1)"
	for host in $HUB $SPARKS $LENOVOS; do
		scp -q -o BatchMode=yes "$TOOL" "$host:/tmp/sparkcp.install"
		got="$($SSH "$host" 'sudo -n install -m 0755 /tmp/sparkcp.install /usr/local/bin/sparkcp && rm -f /tmp/sparkcp.install && sha256sum /usr/local/bin/sparkcp' | cut -d' ' -f1)"
		if [ "$got" != "$want" ]; then
			echo "$host: installed sparkcp does not match" >&2
			return 1
		fi
		echo "$host: sparkcp installed"
	done
	mkdir -p "$HOME/.local/bin"
	install -m 0755 "$TOOL" "$HOME/.local/bin/sparkcp"
}

install_port() {
	for host in "$@"; do
		for file in ds4_cisco_port_apply.sh ds4-cisco-port.service ds4-cisco-port.timer; do
			scp -q -o BatchMode=yes "$HERE/$file" "$host:/tmp/$file"
		done
		$SSH "$host" 'sudo -n install -m 0755 /tmp/ds4_cisco_port_apply.sh /usr/local/sbin/ds4-cisco-port-apply && sudo -n install -m 0644 /tmp/ds4-cisco-port.service /tmp/ds4-cisco-port.timer /etc/systemd/system/ && rm -f /tmp/ds4_cisco_port_apply.sh /tmp/ds4-cisco-port.service /tmp/ds4-cisco-port.timer && sudo -n systemctl daemon-reload && sudo -n systemctl enable ds4-cisco-port.timer && sudo -n systemctl restart ds4-cisco-port.service && systemctl is-active ds4-cisco-port.service'
		echo "$host: cisco port persisted"
	done
}

hub_block() {
	echo "$BEGIN"
	rank=0
	for host in $SPARKS; do
		cisco="10.10.201.$rank"
		printf "Match originalhost %s exec \"timeout 0.3 bash -c '</dev/tcp/%s/22' 2>/dev/null\"\n" "$host" "$cisco"
		printf '    HostName %s\n    HostKeyAlias 10.10.100.%d\n' "$cisco" "$((rank + 10))"
		rank=$((rank + 1))
	done
	echo "Host *"
	echo "$END"
}

install_hub_ssh() {
	block="$(mktemp)"
	hub_block > "$block"
	scp -q -o BatchMode=yes "$block" "$HUB:/tmp/cisco-primary.conf"
	rm -f "$block"
	$SSH "$HUB" "cp -p ~/.ssh/config ~/.ssh/config.bak-cisco-primary && sed -i '/^$BEGIN\$/,/^$END\$/d' ~/.ssh/config && cat /tmp/cisco-primary.conf ~/.ssh/config > ~/.ssh/config.new && chmod 600 ~/.ssh/config.new && mv ~/.ssh/config.new ~/.ssh/config && rm -f /tmp/cisco-primary.conf"
	echo "$HUB: ssh by name prefers cisco"
}

case "${1:-all}" in
	tool) install_tool ;;
	port) shift; install_port "${@:-$LENOVOS}" ;;
	hub-ssh) install_hub_ssh ;;
	hub-block) hub_block ;;
	all) install_tool; install_port $LENOVOS; install_hub_ssh ;;
	*) echo "usage: $0 [all|tool|port [HOST...]|hub-ssh]" >&2; exit 2 ;;
esac
