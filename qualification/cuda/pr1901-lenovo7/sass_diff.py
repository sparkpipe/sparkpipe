import hashlib, subprocess, sys
def kernels(path):
    out = subprocess.run(["cuobjdump", "-sass", path], capture_output=True, text=True).stdout
    found, name, body = {}, None, []
    for line in out.splitlines():
        if "Function :" in line:
            if name:
                found[name] = hashlib.sha256("\n".join(body).encode()).hexdigest()
            name, body = line.split("Function :", 1)[1].strip(), []
        elif name:
            body.append(line.strip())
    if name:
        found[name] = hashlib.sha256("\n".join(body).encode()).hexdigest()
    return found
base, head, adapters = sys.argv[1], sys.argv[2], sys.argv[3:]
total = same = 0
changed = []
for adapter in adapters:
    a, b = kernels(f"{base}/{adapter}"), kernels(f"{head}/{adapter}")
    names = sorted(set(a) | set(b))
    for name in names:
        total += 1
        if a.get(name) == b.get(name):
            same += 1
        else:
            changed.append((adapter.rsplit("/", 1)[-1], name[:120], "missing" if name not in a or name not in b else "differs"))
print(f"{len(adapters)} adapters, {total} kernels, {same} with identical SASS, {len(changed)} different")
for row in changed[:40]:
    print(" ", *row)
