#!/usr/bin/env python3
import json
import sys

path = sys.argv[1]
index = int(sys.argv[2])
with open(path, "r", encoding="utf-8") as f:
    deployment = json.load(f)
print(json.dumps(deployment["nodes"][index], indent=1, sort_keys=True))
