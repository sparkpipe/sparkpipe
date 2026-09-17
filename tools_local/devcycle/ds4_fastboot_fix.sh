#!/bin/bash
# ds4_fastboot_fix.sh - apply ON one spark (via ssh): make data mounts
# nofail + device-timeout + fs_passno=0 so boot-time fsck never blocks SSH.
set -u
FSTAB=/etc/fstab
if ! sudo -n true >/dev/null 2>&1; then
  echo "FASTBOOT-FIX: no passwordless sudo; fstab unchanged"
  exit 1
fi
cp "$FSTAB" "$FSTAB.ds4bak.$(date +%s)"
python3 - "$FSTAB" <<'PYEOF'
import sys
path = sys.argv[1]
lines = open(path).read().splitlines()
protected = {'/', '/boot', '/boot/efi', '/var', '/usr', '/tmp', '/var/log'}
changed = []
out = []
for ln in lines:
    s = ln.strip()
    if not s or s.startswith('#'):
        out.append(ln); continue
    parts = ln.split()
    if len(parts) < 6:
        out.append(ln); continue
    dev, mp, fstype, opts = parts[0], parts[1], parts[2], parts[3]
    is_data = mp not in protected and (
        mp == '/home' or any(t in mp for t in ('extnvme', 'sparkdata', 'kv', 'nvme', 'raid')))
    if not is_data:
        out.append(ln); continue
    o = parts[3].split(',')
    if 'nofail' not in o: o.append('nofail')
    if not any(x.startswith('x-systemd.device-timeout') for x in o):
        o.append('x-systemd.device-timeout=10s')
    parts[3] = ','.join(o)
    parts[5] = '0'
    changed.append(mp)
    out.append(' '.join(parts))
open(path, 'w').write('\n'.join(out) + '\n')
print('FASTBOOT-FIX patched mounts:', changed if changed else '(none matched)')
PYEOF
systemctl daemon-reload 2>/dev/null
echo 'FASTBOOT-FIX done; backup at' $FSTAB.ds4bak.*
