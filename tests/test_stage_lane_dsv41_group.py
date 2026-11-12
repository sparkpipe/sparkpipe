#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import stage_lane  # noqa: E402

FAILURES = []


def check(condition, message):
    if not condition:
        FAILURES.append(message)


def main() -> int:
    whole = stage_lane.MODELS["dsv41_flash"]
    group = stage_lane.MODELS["dsv41_flash_tp4"]
    makefile = (ROOT / "Makefile").read_text()
    tags = dict(re.findall(r"DSV41_FLASH_LINK,(\d+)u,[^,)]+,([a-z0-9_]+)\)", makefile))
    check(stage_lane.lane_ranks(group) == 4 and stage_lane.lane_ranks(whole) == stage_lane.WORLD, "the group entry runs on four ranks, the model's own entry on all")
    check(stage_lane.lane_hosts(group, 0) == ["spark0", "spark1", "spark2", "spark3"] and stage_lane.lane_hosts(group, 12)[0] == "sparkc",
          "a four-rank lane takes an aligned group of four nodes")
    try:
        stage_lane.lane_hosts(group, 2)
        check(False, "a group that starts on node 2 is refused")
    except SystemExit:
        pass
    check(tags.get("4") == group["kv_label"] and tags.get("16") == whole["kv_label"] and group["kv_label"] != whole["kv_label"],
          f"each entry's KV label is its adapter's module tag in the Makefile ({tags})")
    check(group["adapter"] == "libdsv41_flash_tp4_serving_adapter.so" and f"build/{group['adapter'].replace('.so', '')}" in makefile,
          "the group entry names the four-rank adapter the Makefile builds")
    files = [Path(path).name for path in stage_lane.rank_weight_files(group, 3, 0)]
    check(files == ["rank3.spstage", "rank3.spstage.experts", "rank3.spstage.sha256", "engram.spengram", "engram.spengram.digest", "engram_tables.bin"],
          f"a group rank's weight files are its pack, manifest, digest, engram shard, shard digest and hash tables ({files})")
    check("engram.spengram.experts" in whole["pack_siblings"] and "engram.spengram.experts" not in group["pack_siblings"],
          "only the sixteen-rank lane attaches its engram shard through weightd")
    check(stage_lane.rank_pack_path(group, 1, 4) == "/home/spark5/sparkdata/dsv41flash-1017.mxfp4.tp4.xs/packs/rank1.spstage",
          "rank 1 of the group on nodes 4-7 reads its pack on spark5")
    check(group["kv_page_bytes"] == whole["kv_page_bytes"] * stage_lane.WORLD and group["recurrent_bytes"] > whole["recurrent_bytes"],
          "a group rank keeps whole KV pages and the whole sliding window")
    if FAILURES:
        for failure in FAILURES:
            print("FAIL stage lane dsv41 group:", failure)
        return 1
    print("PASS stage lane dsv41 group: the four-rank entry runs on an aligned group, names its own adapter, KV label and weight files, and keeps whole KV pages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
