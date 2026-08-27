# Pack agent rules (shared across all pack-building agents)

## Mission
Take a model from warm/cold storage to deployed stage packs on its
assigned sparks, ready for inference. One agent per model. Parallel.

## Write set
Same as lane rules: only your model's directories, tools, tests.
Shared sharding utilities in tools/spark_pack_common.py are the ONE
place for data-movement primitives. If your model needs a new primitive
(a different sharding axis, a new format, a boundary rule), REQUEST it
from the coordinator with the exact signature - do not fork the file.

## Warm storage discipline (hard rules)
- Scratch dir: /mnt/model-warm/packbuild/<your-model>/
- Warm has 54T free. Keep your total footprint under 1.5T. Delete your
  scratch when packs are deployed.
- Read throughput: ~663 MB/s. Streaming reads during pack building are
  fine; do NOT copy the whole model to warm then read it back.
- Final packs go to each spark's local NVMe, not warm. Warm holds
  reference sources + build scratch only. No AI slop.

## Node assignment (disjoint)
Check your brief for your nodes. spark2 is a dev instance (usable). One
agent per model, one heavy build per node at a time (a prior lane
rebooted a spark with 3 concurrent pack jobs).

## The one build chain
1. Verify source identity against the model contract (or freeze it if
   first run): pinned sha256s, geometry from live config.json.
2. Build the pack per your model's packer (tools/<model>_stagepack.py),
   using spark_pack_common.py primitives. Default topology: TP4xPP4 for
   large models, TP1 for single-spark models. The packer emits a
   receipt: every tensor, byte count, source sha.
3. Verify: run the pack verifier. Exit must be PASS before deploy.
4. Deploy: rsync the rank packs to each spark's
   /home/<host>/sparkdata/<model>.<topology>/packs/
5. Smoke: the serving binary must load the pack (module ready line in
   the daemon log) on the rank-0 node. Full GPU validation only if your
   brief's milestone ladder calls for it.

## Report
After each milestone: what ran, raw output, failures with exact stderr.
