import os, struct, subprocess, sys
base = int(sys.argv[1])
wpid = subprocess.check_output(["pgrep", "-f", "sparkdata/weightd/sparkpipe_weightd"]).decode().split()[0]
f = os.open("/proc/%s/fd/6" % wpid, os.O_RDONLY)
band = 1 * 32 * 16 * 1024 * 1024
SLOT = 16 * 1024 * 1024
vals = []
for r in range(1, 16):
    m = 0
    for p in (0, 1):
        os.lseek(f, band + (r * 2 + p) * SLOT + SLOT - 8, 0)
        m = max(m, struct.unpack("<Q", os.read(f, 8))[0] - base)
    vals.append(m)
print("rounds:", vals)
os.close(f)
