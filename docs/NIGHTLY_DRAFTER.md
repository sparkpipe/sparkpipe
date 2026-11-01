# Nightly repo-trained drafter: feasibility

The first autopilot function: each night, in hours the customer already
holds, retrain Kimi K3's speculative-decoding drafter on the customer's
repository and on K3's own outputs from the customer's sessions, then serve
the better of the old and new drafter the next day. The drafter is the
customer's: it lives in their scratch area, only their lane loads it, and
they can export and restore it.

This note decides whether it pays off and how to build it. Figures are graded
MEASURED (receipt or live check named), DERIVED (arithmetic on measured
figures, shown) or ASSUMED (no measurement yet), as
[FLEET_RELEASE_RUNBOOK.md](FLEET_RELEASE_RUNBOOK.md#9-laws) law 13 requires.
Checked on 2026-11-01 against main `1807177be` and read-only inspection of the
hub.

## Verdict

- **It fits.** Fine-tune the drafter that serves K3 today, the lightseek
  DFlash2 block drafter. Only its weights change: same taps, same daemon,
  same verify.
- **K3's hidden states and training time both bind.** The drafter trains on
  K3's tap rows. K3 makes those at 590 tok/s of prefill, and the day's
  sessions emit them for free while they serve.
  - **Training rate:** the first trainer runs 16 blocks per second on one GB10
    at 48 GiB (MEASURED on lenovo7). Sixteen nodes would train about one pass
    over a night's 3.2M new positions per hour (DERIVED).
  - **Estimate miss:** the estimate of 10 minutes per 8M positions assumed 40%
    of the BF16 peak. The trainer reaches about 3%, so it needs a fused bf16
    attention before nights can run several epochs.
- **Net-new code gains the most; whole-file rewrites gain little.** Lookup
  already carries rewrites (126.4 tok/s). Model drafts run when lookup has no
  8-token match, which is most of net-new code. The ASSUMED range for net-new
  code at B1 is +8% to +24% tok/s. It is still unmeasured; see the
  [validation](#customer-zero-validation-on-lenovo7-2026-11-01) for what five
  files show.
- **Gains are B1-B2 only.** At B8, model drafts change nothing measurable
  today (MEASURED, below).

## The speculation path today

| Piece | Fact | Source |
| --- | --- | --- |
| Draft sources | lookup (suffix match, frames up to 255 drafts) and model drafts when lookup has no match of at least 8 tokens | [PERFORMANCE_STATUS.md](../PERFORMANCE_STATUS.md#speculation) |
| Drafter | lightseekorg/kimi-k3-dflash2 at `e77935fb`: `DFlash2DraftModel`, 5 layers (4 sliding, window 4,096, and 1 full), MLA attention, hidden 7,168, block 8 (7 drafts), top-16 candidate selector | `config.json` beside the weights on the hub; [RTX5090_SPECULATION_NODE.md](RTX5090_SPECULATION_NODE.md#latent-block-drafter-lightseek-dflash2-offline) |
| Where it runs | `sparkpipe-draftd` on the hub's RTX 5090, `tools/block_draftd.py` latent path (`LatentBlockDrafter` in `tools/spec_offline/latent_block_drafter.py`) | `deployment/lanes.json` record `kimi-k3`, key `draftd`; live process on the hub, 18.7 GB of GPU memory (MEASURED 2026-11-01) |
| Inputs | K3 layer outputs o19, o37, o66, o78, o90 from the tap rank (rank 15, sparkf), streamed as `SPT2` frames to `10.10.250.2:7701` | `deployment/lanes.json` `taps`; `include/sparkpipe/spark_tap_stream.h`; `inference/runner/stage_serving_adapter.c` (`StageServingParseTaps`) |
| Request path | DFT3 over TCP from the K3 API to `127.0.0.1:7702`, depth 7, 30 ms wait | `deployment/lanes.json` `draft_server`; `include/sparkpipe/spark_draft_bridge.h` |
| Draft latency | 9-12 ms per request, of which the draft is 6-7 ms | RTX5090_SPECULATION_NODE.md (MEASURED there) |
| Ruling | drafters run only on the rtx5090; the fleet runs exact verification | [SPECULATION_TAPS.md](SPECULATION_TAPS.md) (owner, 2026-09-29) |

The hub's GPU is full: the K3 drafter and the DeepSeek V4.1 Flash DSpark
drafter hold 32.1 of 32.6 GB (MEASURED, `nvidia-smi`, 2026-11-01 04:29 UTC).
Nothing can train there.

## Which drafter to train

| Option | Fit | Decision |
| --- | --- | --- |
| Fine-tune the production DFlash2 drafter | Same taps, same `LatentBlockDrafter`, same DFT3 reply and verify. The new weights drop into the lane's `draftd.drafter`. One forward drafts all 7 tokens | **chosen** |
| Fine-tune the RadixArk DSpark drafter | K3 production until 2026-10-19; DFlash2 was 4-5% faster per token on edits, 128K code and prose in a same-hold A/B (PERFORMANCE_STATUS.md). It still serves DeepSeek V4.1 Flash (`dspark_fast.py`), not K3 | no: it trains the drafter K3 no longer uses |
| EAGLE-style head | A new architecture in draftd and a new training stack. It drafts serially, one pass per token, against one block pass. The DFlash paper reports up to 2.5x the speedup of EAGLE-3 ([arXiv 2602.06036](https://arxiv.org/abs/2602.06036)) | no |
| Small independent draft model | No small model shares K3's 163,840-token vocabulary. It needs its own KV per sequence and serial drafting, it ignores the target's taps, and it needs a new DFT3 source | no |

What is trained (DERIVED from the checkpoint header, 3,692,877,824 parameters
in all):

- 2,518,472,704 drafter parameters: `fc` (7,168 x 35,840), five layers of
  435.1M each, and the candidate selector (85.7M, mostly its two codebooks).
- The checkpoint also carries `embed_tokens` (163,840 x 7,168). The serving
  loader ignores it and takes the embedding, head and final norm from the K3
  target shard (`block_draftd.py --target-shard`). These stay frozen. The
  drafter must keep predicting through the target's own head.

**License (open).** The model card says `license: other` and ships no license
file. The download receipt classes it `upstream-kimi-k3-under-20m-repo-license-omitted`,
and K3 itself is `modified-mit` (`model_contracts/k3_authoritative.json`). A
fine-tuned copy that a customer downloads is a derivative of lightseek's
weights. Counsel or lightseek must confirm before export is offered. Customer
Zero training on our own fleet does not wait on this.

## What it trains on

DFlash training conditions each block on the target's hidden states at every
context position. It predicts the target's next tokens from a random anchor
([arXiv 2602.06036](https://arxiv.org/abs/2602.06036): random anchors resampled
each epoch, cross-entropy weighted down with position in the block). So every
training position needs K3's five tap rows. Text alone is not enough.

| Source | How the tap rows are made | K3 time |
| --- | --- | --- |
| The customer's sessions | The tap rank already emits every computed position (prefill rows, decode and kept verify rows) for draftd. A tee writes the same `SPT2` frames into the customer's scratch | none: they are made while serving |
| The repository | Prefill of files the sessions have not read: the night's `git diff` and the files most read that day, through the customer's own API | 590 tok/s (MEASURED, K3 8K prefill, PERFORMANCE_STATUS.md) |
| Labels | Committed tokens for session rows (K3's own output). For repository rows, K3's argmax from the `final` tap through the frozen head (self-distillation), with the file's own tokens as the fallback | none |

Sizes (DERIVED):

- A position of five bf16 taps at hidden 7,168 is 71,680 bytes plus a 4-byte
  token; with the `final` tap it is 86,020 bytes.
- 90 minutes of nightly prefill is 3.2M positions: 229 GB with five taps.
- SparkPipe's own tracked code and documentation is 26.3 MB (MEASURED, `git
  ls-files`), about 7.5M tokens (ASSUMED 3.5 bytes per token). The first night
  can't prefill all of it, so the bootstrap runs in the customer's other held
  hours or starts from the most-edited files.
- Existing K3 recordings with the DFlash2 taps on the hub's cold storage hold
  about 17K code rows and 136K prose rows in `rec-union-20261018T131504Z`, and
  53K rows of unsorted content in `k3-20261016T022856Z.spt2` (DERIVED from file
  size over row size). That is enough for a training smoke test and far short
  of a repository.

Where it lives:

- Recordings, prepared shards, drafters and receipts sit in the customer's
  scratch: 500 GB per node, 8 TB per fleet.
- The tap rank fills its own 500 GB at about 7M positions, so each night fans
  the day's frames out across all sixteen nodes' scratch before training.
- The function keeps a rolling budget the customer sets (default 2 TB, about
  28M positions).
- The scratch data rule applies unchanged: kept while the customer holds the
  fleet, wiped when another customer's hour starts and in any case 24 hours
  after last use. A customer who stops for more than a day loses the drafter
  unless they exported it.
- Recording runs only on the customer's own lane, only while the function is
  on, and never reaches the hub's disks. Today's draftd `--record-dir` exists
  for SparkPipe's hold corpora on cold storage and must never record a
  customer.

## One nightly window on one fleet

Compute (DERIVED; 90 TFLOPS BF16 per node is the measured cuBLASLt peak in
PERFORMANCE_STATUS.md, 40% utilization is ASSUMED):

- Per block row through the five layers: 2 x 435.1M x 5 = 4.35 GFLOP.
- Attention in the expanded form: about 1 GFLOP per row at an 8K context.
- The head on drafted rows: 2 x 7,168 x 163,840 = 2.35 GFLOP per row.
- One 8-row block, forward and backward with a frozen head, is about 161
  GFLOP. Context rows add about 1.9 GFLOP per position.
- One block per 4 positions makes about 42 GFLOP per position.
- At 40% of the 90 TFLOPS peak, 16 nodes would train 13.7K positions per
  second: an 8M-position epoch in 10 minutes.
- **Measured instead** (lenovo7, one GB10, SM clock up to 3.0 GHz against the
  fleet's 2.2 GHz cap; `latent_block_train.py`, 48 blocks per step):
  - 16.0 blocks (127.8 block rows) per second and 3.0 s per step, at 48.0 GiB
    peak with replicated fp32 Adam;
  - that is about 2.6 TFLOPS, 3% of the peak.
  - The attention runs as fp32 einsums over the whole context, and the
    elementwise work is bound by 273 GB/s.
- At the measured rate, 16 nodes train about 256 blocks per second, or 1,024
  positions per second at one block per 4 positions. One pass over 3.2M
  positions takes 52 minutes. A fused bf16 attention for training (the
  serving path already has a Triton latent kernel) is the first speedup.

Memory per node (DERIVED; with K3 swapped out, since K3's lane holds an
expert pool of 99.6 GB per node):

| Item | Data parallel | Optimizer sharded over 16 nodes |
| --- | ---: | ---: |
| bf16 weights and gradients | 10.1 GB | 10.1 GB |
| fp32 master and Adam moments | 30.2 GB | 1.9 GB |
| frozen K3 embedding and head | 4.7 GB | 4.7 GB |
| activations, 16K positions per step | about 16 GB (ASSUMED) | about 16 GB |
| total | about 61 GB | about 33 GB |

The first trainer keeps fp32 weights and fp32 gradients, and peaked at 48.0
GiB with 48 blocks per step (MEASURED). It fits a Spark only with K3 swapped
out.

Gradient sync (DERIVED): 10 GB of fp32 gradients per step is about 1.6 s on a
ring at 98-109 Gb/s per link, against 3.0 s of compute per step. That is
about 50% overhead until the gradients cross as bf16 buckets, or the steps
grow.

The software (MEASURED on spark0, 2026-11-01): torch 2.14.1+cu130 with CUDA,
121 GB of memory, 1.8 TB free on its NVMe. Multi-node NCCL over the fleet's
RoCE has not been tried (ASSUMED to work; gloo over TCP is the fallback at
this sync size).

A 3-hour window:

| Time | K3 | Step |
| --- | --- | --- |
| 0:00-1:30 | serving | prefill the repository delta; taps tee into scratch (3.2M positions at most) |
| 1:30-1:32 | swapped out | swap time 46 s median, 79 s worst (MEASURED, PERFORMANCE_STATUS.md) |
| 1:32-2:30 | out | train from the incumbent: one pass over the night's 3.2M positions at the measured rate, more once the trainer is faster |
| 2:30-2:45 | out | offline acceptance on the held-out last hours: base, incumbent, candidate |
| 2:45-2:47 | swapped in | draftd loads the winner from scratch |
| 2:47-3:00 | serving | slack |

With the first trainer, both K3 time and training time bind.

## Expected change in acceptance and tok/s

Measured today:

| Workload, K3 B1 | Rate | Source |
| --- | --- | --- |
| Plain decode | 16.2 tok/s | PERFORMANCE_STATUS.md |
| Greedy story, lookup only | 21.15 tok/s | PERFORMANCE_HISTORY_2026-10.md, hold BW2 |
| Greedy story, lookup and model drafts (DSpark era) | 26.41 tok/s | hold BW |
| Whole-file rewrite, lookup | 126.4 tok/s | `qualification/k3/spec-long-frames-20261015/` |
| SEARCH/REPLACE edit | 28.0 tok/s | same |
| B8 with model drafts on and off | 62.5-64.8 against 63.7-63.9 tok/s | `qualification/batch-sweep-20261018/` |
| Net-new code with model drafts | **not measured on our engine** | the prototype's first number |

The lightseek card reports 4.43 tokens per round on SPEED-Bench coding and
5.08 on HumanEval, at temperature 1 with 7 proposals (49% and 58% of proposals
accepted). That was TokenSpeed on GB300, not our engine.

Round cost (DERIVED, with one ASSUMED step):

- A round is one 8-row verify frame plus the draft.
- K3's B8 step reads 8 rows at 63.8 tok/s, 125 ms per step. A one-sequence
  8-row frame is ASSUMED to cost the same.
- With 9-12 ms of draft, a round is 135-140 ms.
- tok/s = tokens per round / round time:

| Tokens per round | 3.5 | 4.0 | 4.5 | 5.0 | 5.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| tok/s at 135-140 ms | 25-26 | 29-30 | 32-33 | 36-37 | 39-41 |

The gain from training (ASSUMED):

- Online Speculative Decoding raised acceptance by 0.10-0.65 by distilling
  small drafters on live queries ([ICML 2024](https://proceedings.mlr.press/v235/liu24y.html)).
- This drafter is large and already strong on code, so the assumption here is
  +5 to +15 points of proposal acceptance on the customer's own code. That is
  +0.35 to +1.05 tokens per round at 7 proposals: from 4.4 to 4.75-5.45, or
  +8% to +24% tok/s at B1 on net-new code.
- The owner's "40 to 45 tok/s" (+12.5%) sits inside that range.

Other workloads:

- **Rewrites.** Lookup copies the unchanged text. Model drafts fill only the
  lines that change, so whole-file rewrites gain a few percent at most.
  SEARCH/REPLACE replies, which are mostly new text, gain more.
- **Prose and chat.** No gain expected unless the customer's sessions are
  prose.
- **Regression risk.** A drafter fit to one repository can lose acceptance
  elsewhere. Each night therefore mixes a replay share of general hold-corpus
  rows into training and publishes a candidate only if:
  - it beats the incumbent on the customer's held-out hours, and
  - it stays within 2% of the base drafter on a general set.

  Otherwise the incumbent stays.

## Customer Zero validation on lenovo7 (2026-11-01)

The first fine-tune ran without fleet time. Inputs:

- **Recording:** K3's recording of the 20-task edit corpus, taken on
  2026-10-16:
  - ten SparkPipe files (`qualification/spec_offline/k3-edits-20261015/corpus.json`);
  - each one answered greedily as a whole file and as SEARCH/REPLACE blocks;
  - the DFlash2 taps and `final`;
  - `k3-20261016T022856Z.spt2` on the hub's cold storage, reduced with
    `latent_block_train.py select` to 52,712 rows.
- **Mapping:** request 100001+i is corpus task i; prompt lengths and token
  streams match the 10-15 streams.
- **Training:** files 01-05 (requests 100001-100010, 19,152 positions), 300
  steps of 48 blocks, lr 2e-5, 15 minutes.
- **Held out:** files 06-10 (requests 100011-100020).
- **Prose check:** ten 286-token story replies from the same recording.
- **Scoring:** `block_drafter.py`:
  - the serving kernels;
  - the same anchors for both arms;
  - tokens per round = 1 + accepted drafts, at most 8;
  - "replies" are K3's generated tokens; "file text" is the prompt K3 read,
    scored against the file's own next tokens.

| Set | Anchors | Base | Tuned | Change |
| --- | ---: | ---: | ---: | ---: |
| Held-out files, replies | 3,203 | 7.373 | 7.449 | +1.0% |
| Held-out files, file text | 3,363 | 3.827 | 4.143 | +8.3% |
| of which the 8 prompts recorded from position 0 | 3,200 | 3.943 | 3.999 | +1.4% |
| of which the 2 recorded mid-sequence | 163 | 1.56, 1.61 | 6.75, 7.16 | |
| Training files, file text | 4,000 | 3.547 | 6.830 | +93% |
| Training files, replies | 2,864 | 7.513 | 7.859 | +4.6% |
| Prose, story replies | 2,421 | 2.693 | 2.640 | -2.0% |

What it shows:

1. **The pipeline works on the real drafter.**
   - The 3.69B checkpoint's 86 drafter tensors match the contract.
   - It trains on one GB10 at 48 GiB.
   - The new checkpoint serves through the unchanged scorer and kernels.
   - The 5.04 GB drafter installs in 8.2 s, exports in 6.8 s and restores on a
     second scratch in 8.8 s. A different target pack is refused.
2. **The drafter learns what it trains on.** On the training files' own
   text, acceptance nearly doubles (3.55 to 6.83 tokens per round). A drafter
   trained on the repository at night predicts the code it saw almost word
   for word.
3. **Five files do not carry over to other files.**
   - The +8.3% on held-out file text comes from two short SEARCH/REPLACE
     prompts recorded from mid-sequence. Their text is the instruction
     boilerplate shared with the training prompts.
   - On the eight complete held-out prompts the change is +1.4%, inside the
     spread from file to file (-3.1% to +8.9%).
   - Replies gain 1.0%; they are mostly copies at 7.4 of 8 already.
   - Whether a whole repository carries over to next-day net-new code is
     still the open measurement.
4. **Prose fell 2.0%**, exactly at the gate's floor. Training mixed in no
   general rows, so the nightly run needs a replay share.
5. **The gate must score replies only.** Prompt text is prefilled, never
   drafted, and its boilerplate inflated the file-text score. `autopilot_drafter.py
   gate` now refuses scores of any region but `reply`.

## The customer-run function

The night's steps are record, prepare, train, evaluate, gate, publish, and,
on the customer's request, export and restore.

Built:

- `tools/spec_offline/latent_block_train.py train`:
  - fine-tunes a latent block drafter from recorded tap streams;
  - the context projector (`fc`, `hidden_norm`) stays frozen, so each
    position's features (14,336 bytes, against 71,680 for the five raw taps)
    can be computed once and kept;
  - the loss is the head's cross-entropy plus the selector lattice's, given
    the true previous token, weighted `exp(-(k-1)/4)` by block position;
  - it runs data-parallel under `torchrun`: each rank trains on its own share
    of the requests, gradients are averaged, and the ranks' weights are
    checked equal before rank 0 writes the checkpoint and
    `TRAINING-RECEIPT.json` (base sha256, tap entries, target pack identity,
    data, steps, throughput).
- `latent_block_train.py select`: a tap stream reduced to the named entries
  and requests, for storage and transfer.
- `tools/spec_offline/block_drafter.py --requests --region`: offline
  acceptance on chosen requests, over the target's replies or the prompts it
  read.
- `tools/autopilot_drafter.py`:
  - `install` places a trained checkpoint under
    `<scratch>/autopilot/drafter/drafters/<id>`, where the id is the first 16
    hex digits of the weights' sha256;
  - `publish` points `published` at a verified drafter;
  - `export` writes a tar of `MANIFEST.json`, `config.json`,
    `model.safetensors` and `TRAINING-RECEIPT.json`;
  - `restore` takes such an archive, on any fleet;
  - `status` lists the drafters and the published one;
  - `gate` publishes a candidate only if it beats the incumbent on the
    replies of held-out hours, and stays within 2% of the base on a general
    set. It refuses scores of any other region.

  Install and restore refuse:
  - files other than those four;
  - an unsupported config;
  - weights that are not exactly the drafter's tensors (BF16, the config's
    shapes, contiguous);
  - a receipt that names other weights;
  - taps the lane does not stream;
  - another target pack.

  Restore also refuses links, duplicates, path tricks and a manifest that
  does not describe its files, and leaves nothing behind when it does.

Still to build:

1. **Tap tee.** `StageServingParseTaps` takes exactly one of `"file"` and
   `"stream"`. A lane that both serves drafts and records needs both: the
   stream to draftd, and a size-capped file in the customer's scratch on the
   tap rank. This is shared adapter code and stays model-neutral.
2. **The night's driver.** One command per night on the customer's fleet:
   - pick the repository delta and prefill it;
   - swap K3 out and fan the shards out;
   - run `torchrun` on the sixteen Sparks;
   - score base, incumbent and candidate on the held-out hours;
   - gate, install, publish, and swap K3 back.

   Multi-node NCCL over the fleet's RoCE is untried. Gradients cross as
   fp32 (10 GB a step), and bf16 buckets would halve that.
3. **Serving the customer's drafter.**
   - At lane start, draftd loads the published drafter from the customer's
     scratch over the sparkf link, about 5 s for 5 GB.
   - It holds the drafter in GPU memory only and drops it when the lane
     stops or another customer's hour starts.
   - It refuses a drafter whose manifest names a different base, tap set or
     target pack identity.
   - `hub_draftd.sh` today accepts weights only from the hub's own NVMe, so it
     needs this source added.
4. **Download and upload.** Moving an exported archive to the customer and
   an uploaded one onto their fleet is the account API's job: it hands the
   file to `export` and `restore`.

Each fleet will have its own hub (owner, 2026-11-01). The customer's drafter
therefore serves from their fleet's hub GPU, and no other customer's lane
ever reaches that daemon. Today's hub GPU is full (two drafters, 32.1 of
32.6 GB), so a production hub runs the K3 drafter only, or has a larger GPU.

Training stays on the customer's Sparks. A full fine-tune needs about 33-61
GB per device, more than a 32 GB hub GPU holds. Moving the customer's
recordings to any other machine would also take their working state off
their fleet.

## Prototype plan: Customer Zero

SparkPipe's own repository is the customer, and its data may leave the fleet.
Training runs on `lenovo7`, the build host: a GB10 outside the fleet with
torch 2.14 and CUDA (owner, 2026-11-01). The fleet is needed only to record
K3's taps. Freeze a training commit C0 on main. Hold out everything after it.

- **Training data:**
  - the most-edited 2M tokens of the repository at C0, prefilled with taps;
  - K3 replies on about 200 tasks built from commits before C0: the commit
    message plus the files it touched, asking for the change.
- **Test data**, 40 tasks from commits after C0:
  - 20 net-new: write the function, test or tool a commit added, given its
    files;
  - 20 rewrites: `tools/spec_edit_corpus.py` whole-file and SEARCH/REPLACE
    edits on files changed after C0.
  - Each runs greedy and at temperature 1, top-p 0.95, as chat serves.
- **Arms:** the base drafter against the fine-tuned one, on the same
  recorded test streams.
- **Measurements:**
  - offline tokens per round for both arms;
  - end-to-end tok/s with draftd serving each drafter in turn, in one hold.
- **Training time:** at the measured 16 blocks per second, 2M positions at
  one block per 8 positions take about 4.3 hours on lenovo7. A fused bf16
  attention, or lenovo9 as a second rank, shortens that.
- **Fleet time:** about 4 hours exclusive with K3, since training runs off
  the fleet:
  - about 1 hour of prefill;
  - about 1.5 hours generating training replies;
  - about 1.5 hours of test generation and the two serving arms.

Recording the prototype's tap streams needs one of two things. The tee
(item 1) can ship first, through a reviewed release. Or the production draftd
can restart with `--record-dir` on cold storage for the hold, the documented
hold-corpus procedure. That restart is a hub unit restart, which AGENTS.md
forbids without the owner's sign-off. Either way, the second serving arm
needs the lane's `draftd.drafter` pointed at the candidate for the hold.
