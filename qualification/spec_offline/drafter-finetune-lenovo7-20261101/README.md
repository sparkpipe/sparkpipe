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

## Licenses of what was used

The run recorded no license; the training tool now requires one per source.
What is known:

- **Base checkpoint**, lightseekorg/kimi-k3-dflash2 at `e77935fb`: unknown.
  The model card says `license: other` and ships no license file. The tuned
  checkpoint is a derivative of it.
- **Tap file** (`k3-edits-20261016.spt2`, the training and scoring data):
  K3's hidden states and tokens over prompts built from ten SparkPipe files.
  - The prompts are SparkPipe's own code under its
    [license](../../../LICENSE) (FSL-1.1-ALv2 with
    [the additional permission](../../../LICENSE-PERMISSION.md)), wrapped in
    the corpus's instruction text.
  - K3's outputs (the replies, the hidden states, the `final` tap) come
    from Kimi K3, whose license is `modified-mit`
    (`model_contracts/k3_authoritative.json`). Whether and how that license
    covers outputs and hidden states has not been reviewed: unknown.
  - Where the ten story prompts (requests 100022-100031, scored but not
    trained on) came from is not recorded: unknown.
- **Target shard** (`model-00094-of-000096.safetensors`, K3's embedding and
  head, frozen and not written into the checkpoint): Kimi K3, `modified-mit`.

## Source and reproducibility

- No source commit was recorded by the run, and the receipt and scores do
  not name one.
- The scores came from `tools/spec_offline/block_drafter.py` as of commit
  `4ae09777a`. That is inferred from git history, not recorded. That commit
  is the only one with the `--requests` and `--region` flags `validate.sh`
  uses whose `arm` record has no `region` or `requests`, and these files
  have neither. The trainer is the same in `4ae09777a` and the later commits
  of this PR.
- The scores predate the current scorer, which also records the region, the
  requests, `max_anchors`, the drafter's sha256 and the identity of the
  target shard and tap files.
- **Gate check: not run.** `autopilot_drafter.py gate` refuses these scores,
  since they name no region and no drafter. Rescoring needs the GPU and the
  tap file.
- `tuned/TRAINING-RECEIPT.json` predates the license and `origin_sha256`
  fields, so `autopilot_drafter.py install` refuses it today.
