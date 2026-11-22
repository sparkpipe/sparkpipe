#!/usr/bin/env bash
set -euo pipefail

usage() {
	echo "usage: TAILSCALE_AUTH_KEY_FILE=FILE tailscale_join.sh --ssh TARGET --hostname NAME --tags TAG[,TAG...] [--operator USER] [--key-dir DIR]" >&2
	exit 2
}

fail() {
	echo "tailscale_join: $*" >&2
	exit 1
}

target="" hostname="" tags="" operator="" key_dir=/run/sparkpipe-tailscale-join
while [ $# -gt 0 ]; do
	[ $# -ge 2 ] || usage
	case "$1" in
		--ssh) target="$2" ;;
		--hostname) hostname="$2" ;;
		--tags) tags="$2" ;;
		--operator) operator="$2" ;;
		--key-dir) key_dir="$2" ;;
		*) usage ;;
	esac
	shift 2
done
[ -n "$target" ] && [ -n "$hostname" ] && [ -n "$tags" ] || usage
[[ "$target" =~ ^[A-Za-z0-9._@-]+$ ]] || fail "--ssh $target is not an SSH host or user@host"
[[ "$hostname" =~ ^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$ ]] || fail "--hostname $hostname is not a DNS label of at most 63 characters"
[[ "$tags" =~ ^tag:[a-z0-9-]+(,tag:[a-z0-9-]+)*$ ]] || fail "--tags $tags is not a comma list of tag:name"
[ -z "$operator" ] || [[ "$operator" =~ ^[a-z_][a-z0-9_-]*$ ]] || fail "--operator $operator is not a user name"
[[ "$key_dir" =~ ^/[A-Za-z0-9._/-]+$ ]] && [ "${key_dir%/*}" != "" ] || fail "--key-dir $key_dir is not an absolute path below a mount point's directory"

key_file="${TAILSCALE_AUTH_KEY_FILE:-}"
[ -n "$key_file" ] || fail "TAILSCALE_AUTH_KEY_FILE names no file; it must name the 0600 auth key or OAuth client secret file"
[ -f "$key_file" ] && [ ! -L "$key_file" ] || fail "$key_file is not a regular file"
[ -n "$(find "$key_file" -maxdepth 0 -user "$(id -u)")" ] || fail "$key_file is not owned by $(id -un)"
[ -z "$(find "$key_file" -maxdepth 0 \( -perm -g=r -o -perm -g=w -o -perm -g=x -o -perm -o=r -o -perm -o=w -o -perm -o=x \))" ] || fail "$key_file is readable or writable by others; it must be mode 0600"
[ "$(wc -c < "$key_file")" -le 4096 ] || fail "$key_file is longer than 4096 bytes"
[ "$(wc -l < "$key_file")" -le 1 ] && [ "$(grep -c . "$key_file")" = 1 ] || fail "$key_file must hold one line"
grep -Eq '^tskey-(auth|client)-[A-Za-z0-9_-]+(\?[A-Za-z0-9=&_-]+)?$' "$key_file" || fail "$key_file holds neither a Tailscale auth key nor an OAuth client secret"
if grep -q '^tskey-client-' "$key_file"; then
	grep -Eq '[?&]ephemeral=false(&|$)' "$key_file" && grep -Eq '[?&]preauthorized=true(&|$)' "$key_file" || fail "$key_file holds an OAuth client secret without ?ephemeral=false&preauthorized=true; without them the node would be ephemeral and removed when it goes offline, or wait for approval"
fi

status_of() {
	ssh -o BatchMode=yes -o ConnectTimeout=10 "$target" sudo -n tailscale status --json < /dev/null
}

check_status() {
	python3 -I -c '
import json, sys
mode, hostname, tags = sys.argv[1], sys.argv[2], sorted(sys.argv[3].split(","))
status = json.load(sys.stdin)
state = status.get("BackendState")
node = status.get("Self") or {}
name, online, held = node.get("HostName"), node.get("Online"), sorted(node.get("Tags") or [])
joined = state == "Running" and online is True and name == hostname and held == tags
if mode == "before" and joined:
    print("joined")
elif mode == "before" and state in ("NeedsLogin", "NoState", "Stopped"):
    print("ready")
elif mode == "before":
    print("the node is %s as %r with tags %s, not logged out; log it out first or check its name and tags" % (state, name, held))
    sys.exit(1)
elif joined:
    print(" ".join(node.get("TailscaleIPs") or []))
else:
    print("the node is %s as %r, online %s, tags %s; expected Running and online as %r with %s" % (state, name, online, held, hostname, tags))
    sys.exit(1)
' "$1" "$hostname" "$tags"
}

before="$(status_of)" || fail "could not read tailscale status --json on $target"
verdict="$(check_status before <<< "$before")" || fail "$target: $verdict"
if [ "$verdict" = joined ]; then
	echo "tailscale_join: $target is already Running as $hostname with $tags; no key was sent"
	exit 0
fi

remote='set -euo pipefail
dir="$1" hostname="$2" tags="$3" operator="${4:-}"
[ "$(findmnt -n -o FSTYPE --target "${dir%/*}")" = tmpfs ] || { echo "tailscale_join: ${dir%/*} is not on tmpfs" >&2; exit 1; }
[ ! -e "$dir" ] || { echo "tailscale_join: $dir exists; remove what a previous join left there" >&2; exit 1; }
umask 077
mkdir -m 0700 "$dir"
cleanup() { rm -f -- "$dir/key"; rmdir -- "$dir"; }
trap cleanup EXIT
trap "exit 129" HUP
trap "exit 130" INT
trap "exit 143" TERM
head -c 4097 > "$dir/key"
[ "$(wc -c < "$dir/key")" -le 4096 ] || { echo "tailscale_join: the key sent is longer than 4096 bytes" >&2; exit 1; }
tailscale up --auth-key="file:$dir/key" --hostname="$hostname" --advertise-tags="$tags" ${operator:+--operator="$operator"} --timeout=120s'
script="$(printf '%s' "$remote" | base64 | tr -d '\n')"
remote_command="sudo -n bash -c \"\$(printf %s $script | base64 -d)\" tailscale-join $key_dir $hostname $tags $operator"
ssh -o BatchMode=yes -o ConnectTimeout=10 "$target" "$remote_command" < "$key_file" || fail "tailscale up failed on $target"
ssh -o BatchMode=yes -o ConnectTimeout=10 "$target" sudo -n test ! -e "$key_dir" < /dev/null || fail "$key_dir is still on $target; remove it"
for attempt in 1 2 3 4 5; do
	after="$(status_of)" || fail "could not read tailscale status --json on $target after the join"
	if addresses="$(check_status after <<< "$after")"; then
		echo "tailscale_join: $target is Running and online as $hostname with $tags at $addresses; the key file on the node is gone"
		exit 0
	fi
	[ "$attempt" = 5 ] || sleep 2
done
fail "$target: $addresses"
