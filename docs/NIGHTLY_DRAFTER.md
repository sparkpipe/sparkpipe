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
- **Training is cheap; producing K3's hidden states is not.** One epoch over
  8M positions takes about 10 minutes of fleet compute. The drafter trains on
  K3's tap rows, and K3 makes those at 590 tok/s of prefill. The day's
  sessions emit them for free while they serve. Only the repository delta
  needs K3 time at night.
- **Net-new code gains the most; whole-file rewrites gain little.** Lookup
  already carries rewrites (126.4 tok/s). Model drafts run when lookup has no
  8-token match, which is most of net-new code. The ASSUMED range for net-new
  code at B1 is +8% to +24% tok/s. The prototype measures it before anything
  ships.
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
- 16 nodes x 90 TFLOPS x 40% = 576 TFLOPS gives 13.7K positions per second.
  One epoch over 3.2M positions takes 4 minutes; over 8M, 10 minutes. At 15%
  utilization the 8M epoch takes 26 minutes.

Memory per node (DERIVED; with K3 swapped out, since K3's lane holds an
expert pool of 99.6 GB per node):

| Item | Data parallel | Optimizer sharded over 16 nodes |
| --- | ---: | ---: |
| bf16 weights and gradients | 10.1 GB | 10.1 GB |
| fp32 master and Adam moments | 30.2 GB | 1.9 GB |
| frozen K3 embedding and head | 4.7 GB | 4.7 GB |
| activations, 16K positions per step | about 16 GB (ASSUMED) | about 16 GB |
| total | about 61 GB | about 33 GB |

Gradient sync (DERIVED): 5.04 GB of bf16 gradients per step is about 0.8 s on a
ring at 98-109 Gb/s per link. A step of 16K positions per node is about 19 s
of compute, so the sync costs about 4%.

The software (MEASURED on spark0, 2026-11-01): torch 2.14.1+cu130 with CUDA,
121 GB of memory, 1.8 TB free on its NVMe. Multi-node NCCL over the fleet's
RoCE has not been tried (ASSUMED to work; gloo over TCP is the fallback at
this sync size).

A 3-hour window:

| Time | K3 | Step |
| --- | --- | --- |
| 0:00-1:30 | serving | prefill the repository delta; taps tee into scratch (3.2M positions at most) |
| 1:30-1:32 | swapped out | swap time 46 s median, 79 s worst (MEASURED, PERFORMANCE_STATUS.md) |
| 1:32-2:30 | out | train from the incumbent: 2-3 epochs over the newest 15-25M positions |
| 2:30-2:45 | out | offline acceptance on the held-out last hours: base, incumbent, candidate |
| 2:45-2:47 | swapped in | draftd loads the winner from scratch |
| 2:47-3:00 | serving | slack |

The binding cost is K3 time to make tap rows. Training is not.

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

## The customer-run function

One model-neutral tool drives the night, keyed by lane record and scratch
root. Its steps are `record`, `prepare`, `train`, `evaluate`, `publish`,
`export` and `restore`.

Still to build:

1. **Tap tee.** `StageServingParseTaps` takes exactly one of `"file"` and
   `"stream"`. A lane that both serves drafts and records needs both: the
   stream to draftd, and a size-capped file in the customer's scratch on the
   tap rank. This is shared adapter code and stays model-neutral.
2. **Training.** A differentiable training step for `LatentBlockDrafter`,
   checked against its inference forward on the tiny random checkpoint of
   `tests/test_latent_block_drafter.py`. It needs random anchors, the
   position-weighted loss, frozen target embedding and head, and
   `torch.distributed` across the sixteen nodes. It writes only the drafter's
   own tensors, about 5.0 GB in bf16.
3. **Evaluation.** Offline acceptance per drafter on the recorded held-out
   streams, the scoring `tools/spec_offline/long_drafter.py` and
   `block_drafter.py` already do. With keyed-noise sampling the committed
   stream does not depend on the draft, so the replay scores sampled sessions
   exactly.
4. **Serving the customer's drafter.**
   - At lane start, draftd loads the published drafter from the customer's
     scratch over the sparkf link, about 5 s for 5 GB.
   - It holds the drafter in GPU memory only and drops it when the lane
     stops or another customer's hour starts.
   - It refuses a drafter whose manifest names a different base, tap set or
     target pack identity.
   - `hub_draftd.sh` today accepts weights only from the hub's own NVMe, so it
     needs this source added.
5. **Export and restore.**
   - **Export** packs the drafter's tensors with a manifest: base drafter repo
     and sha256, tap layers, target pack identity, training receipt (data
     hours, positions, held-out acceptance) and a sha256 per file.
   - **Restore** checks every one of those against the target fleet's K3
     lane before it places the drafter in that fleet's scratch. It works on
     any fleet the customer holds.

Open items for the 30-fleet facility:

- Each fleet's K3 lane needs a drafter GPU. One RTX 5090 serves one fleet
  today and is full.
- If drafters stay hub-only by the owner's ruling, every fleet needs its own
  drafter host or a slot on a shared one.

## Prototype plan: Customer Zero

SparkPipe's own repository is the customer. Freeze a training commit C0 on
main. Hold out everything after it.

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
- **Fleet time:** about 5 hours exclusive with K3:
  - about 1 hour of prefill;
  - about 1.5 hours generating training replies;
  - about 1 hour of training with K3 swapped out;
  - about 1.5 hours of test generation and the two serving arms.

Recording the prototype's tap streams needs one of two things. The tee
(item 1) can ship first, through a reviewed release. Or the production draftd
can restart with `--record-dir` on cold storage for the hold, the documented
hold-corpus procedure. That restart is a hub unit restart, which AGENTS.md
forbids without the owner's sign-off. Either way, the second serving arm
needs the lane's `draftd.drafter` pointed at the candidate for the hold.
