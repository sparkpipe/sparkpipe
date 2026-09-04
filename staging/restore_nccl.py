import json
import os
from pathlib import Path

home = Path(os.environ.get("HOME", "")).resolve()
root = home / "sparkdata" / "glm5_next.tp16" / "config"
backup = root / "stage.json.nccl-era"
path = root / "stage.json"
if str(path.resolve())[: len(str(home))] != str(home) or not backup.is_file():
    raise SystemExit("unexpected path or missing backup")
with backup.open() as f:
    data = json.load(f)
with path.open("w") as f:
    json.dump(data, f, indent=1)
tc = data["tp_collective"]
print("restored rank", data["tp_rank"], "backend", tc["backend"])
