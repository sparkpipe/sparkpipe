import sys, time
for line in sys.stdin:
    sys.stdout.write(f"{time.time():.6f} {line}")
    sys.stdout.flush()
