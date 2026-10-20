#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TARGET="$HOME/.config/systemd/user"
ENABLE=0
case "${1:-}" in
  "") ;;
  --enable) ENABLE=1 ;;
  *) echo "usage: $0 [--enable]" >&2; exit 2 ;;
esac
mkdir -p "$TARGET"
for unit in "$HERE"/systemd/*.service; do
  name="$(basename "$unit")"
  install -m 0644 "$unit" "$TARGET/$name.new"
  mv "$TARGET/$name.new" "$TARGET/$name"
  echo "installed $name"
done
systemctl --user daemon-reload
if [ "$ENABLE" = 1 ]; then
  for unit in "$HERE"/systemd/*.service; do
    systemctl --user enable "$(basename "$unit")"
  done
  loginctl show-user "$USER" -p Linger | grep -qx 'Linger=yes' || { echo "linger is off for $USER: enabled units start only at login" >&2; exit 1; }
fi
