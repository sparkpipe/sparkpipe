# Diagnostic probes, removed

Six C probes and three Python audit scripts, 4,879 lines, deleted after the tree
they probed was replaced. They are in git history; this records what each did so
that rewriting one is a decision rather than a rediscovery.

Probe code is cheap to write and expensive to keep. Each of these was written to
answer one question about one version of the system, and every one of them
outlived the answer — several still referenced the decode stage months after the
paths they exercised had changed underneath them.

## Ring and transport

**`sparkpipe_glm52_pp13_ring_check.c`** (610) — walked the rank ring and verified
every hop's hidden-transport packet arrived with the expected shape and sequence
number. Answered "is the ring wired correctly" before a run.

**`sparkpipe_glm52_pp13_loopback_probe.c`** (635) — launched a hidden-state
payload from a rank back to itself through the transport and timed it. Its output
line was `loopback_launch rank=%u sequence=%u wall_ns=%llu`. This is where the
29 µs/hop software floor recorded in the calibration doc came from.

**`sparkpipe_hidden_transport_preflight.c`** (127) — checked that a transport
could be created with the model's hidden dimensions before a run committed to
them.

**`sparkpipe_glm52_pp13_rank_gate.c`** (836) — validated a rank's runtime
configuration against the stage plan: layer ownership, slot counts, whether this
rank's slice matched what the plan said it owned.

## Resident state

**`sparkpipe_glm52_cuda_resident_gate.c`** (326) — checked the CUDA resident
daemon's IPC handles were live and the weights it claimed to hold were mapped.

## Prefill

**`sparkpipe_glm52_prefill_dryrun.c`** (676) — tokenized a prompt and walked the
prefill admission path without launching kernels, reporting the chunking the
scheduler would choose. Useful for reasoning about chunk sizes without a GPU.

## Repository audits

**`audit_core_boundaries.py`** (233) — checked that layers did not include across
boundaries they should not. Superseded by the directory structure making the
boundaries visible.

**`run_deep_audit_validation.py`** (536) and **`package_audited_proposal.py`**
(572) — a workflow for packaging a change with its evidence. Superseded by
`tools/gates.sh`.

## Kept

**`length_gate.py`** (26) — exits non-zero on any Spark-prefixed function body
over fifty lines, so shell `&&` chains stop. Small, still true, still enforced.

## What to write instead, when the ring is available

The three transport probes answered one question between them: does a hidden
state cross a rank boundary intact and how long does it take. One probe over
`ring/transport/hidden_transport.h` answers it for any rank count, where these
three were written against a thirteen-rank assumption and a decode stage that no
longer exists.
