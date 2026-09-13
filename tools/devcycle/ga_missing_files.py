#!/usr/bin/env python3
"""List GA repo files that are missing or wrong-sized on disk (one per line)."""
import json
import os

tree = json.load(open("/tmp/ga-tree.json"))
missing = []
for entry in tree:
    if entry.get("type") != "file":
        continue
    path = entry["path"]
    full = os.path.join(os.getcwd(), path)
    if not os.path.exists(full) or os.path.getsize(full) != entry["size"]:
        missing.append(path)
with open("/tmp/ga-missing.txt", "w") as out:
    out.write("\n".join(missing) + "\n")
print("missing files:", len(missing))
