import json
import os
from pathlib import Path

home = Path(os.environ.get("HOME", "")).resolve()
root = home / "sparkdata" / "glm5_next.tp16" / "config"
path = root / "stage.json"
backup = root / "stage.json.nccl-era"
if str(path.resolve())[: len(str(home))] != str(home) or not path.is_file():
    raise SystemExit("unexpected path")

with path.open() as f:
    data = json.load(f)
if not backup.exists():
    with backup.open("w") as f:
        json.dump(data, f, indent=1)

tc = data["tp_collective"]
tc["backend"] = "hidden_transport"
tc["backend_module_path"] = "lib/hidden_transport.so"
tc["algorithms"] = ["recursive_doubling"]
tc["direct_all_to_all_max_payload_bytes"] = 0
tc["split_ring_min_payload_bytes"] = 0
tc["rail_peer_hosts"] = [
    ["10.10.200.%d" % i for i in range(16)],
    ["10.10.100.%d" % (10 + i) for i in range(16)],
]
tc["step_rail_indices"] = [0] + [1] * 15
with path.open("w") as f:
    json.dump(data, f, indent=1)
print("flipped rank", data["tp_rank"], "backend", tc["backend"])
