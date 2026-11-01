# K3 DFlash2 drafter fine-tune on five SparkPipe files (2026-11-01)

The first fine-tune of the production K3 drafter: lightseekorg/kimi-k3-dflash2
at `e77935fb`, model sha256 `1f2a5376...52c2`. It ran on `lenovo7`, a GB10
outside the fleet. Results and reading:
[docs/NIGHTLY_DRAFTER.md](../../../docs/NIGHTLY_DRAFTER.md#customer-zero-validation-on-lenovo7-2026-11-01).

- **Data:**
  - `k3-20261016T022856Z.spt2` from the hub's cold storage, reduced with
    `tools/spec_offline/latent_block_train.py select --entries
    o19,o37,o66,o78,o90,final --requests 100001-100031` (52,712 rows);
  - requests 100001-100020 are the 20 tasks of
    `../k3-edits-20261015/corpus.json` in order, and 100022-100031 are story
    prompts.
- **Training:** files 01-05 (requests 100001-100010).
  `latent_block_train.py train --steps 300 --anchors 48 --lr 2e-5` gave
  `tuned/TRAINING-RECEIPT.json` and `tuned/train.log`. The tuned model's
  sha256 is `4c78ab6b...30b3`.
- **Scoring:** `block_drafter.py --max-anchors 400` over replies or prompts,
  on the held-out files 06-10 (`test`), the training files (`train`) and the
  story replies (`story`). The files are `base/score-*.json` and
  `tuned/score-*.json`. `validate.sh` is the driver.
