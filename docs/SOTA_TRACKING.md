# SOTA Tracking - primary-source scanner and comparable-target ledger

SparkPipe targets at least 110% of the best genuinely comparable public
inference result for every supported exact model. This document defines how
those public results are discovered, recorded, compared, and turned into a
target for every exact benchmark cell. The implementation is
`tools/update_sota_ledger.py`; the
observation contract is `schema/sota_observation.schema.json`.

## What counts as evidence

Every observation binds all of the following or it does not enter the ledger:

| Binding | Field(s) |
| --- | --- |
| Source URL | `source.url` |
| Retrieval time | `source.retrieved_utc` |
| Source revision or publication date | `source.revision` / `source.publication_date` (at least one) |
| Evidence class | `source.evidence_class` |
| Exact checkpoint | `checkpoint.name`, `checkpoint.revision` |
| Quality mode | `quality_mode` plus structured `precision.*` weight dtype/codec, activation, accumulator, and KV dtype/codec |
| Numerical gate | `quality_gate.method`, `status`, `output_parity`, `atol`, `rtol` |
| Runtime routes | `execution.compute_route`, `execution.kv_route` |
| Hardware | `hardware.accelerator_name`, `hardware.accelerator_count` |
| Fabric | `fabric.interconnect`, collective backend, per-device bandwidth, RDMA state |
| Power / clocks / thermal | every field under `power_state` |
| Topology | `topology.kind`, `topology.tp_size`, `topology.pp_size`, `topology.ep_size` |
| Batch | `batch.batch_size` |
| Prompt / output / window lengths | `workload.prompt_tokens`, `workload.output_tokens`, `workload.context_window_tokens` |
| Context occupancy | `workload.context_occupancy_pct` (prompt as percent of context window) |
| Shape distribution | fixed/distributed B, prompt, output, and context p50/p95 values under `request_distribution` |
| Prefix cache | `prefix_cache.enabled`, `state`, `matched_tokens` |
| Speculation | `speculation.enabled`, `speculation.provider`, `speculation.draft_length` |
| TTFT / ITL SLO | `service_level_objectives.ttft_ms_max`, `itl_ms_max` (explicit `null` means unconstrained) |
| Statistical design | randomization, paired state, warmups, sample count, summary statistic, confidence method/level |
| Confidence result | `statistics.confidence_interval.lower`, `upper` (recorded evidence, not a cell selector) |
| Metric | `metric.name`, `metric.value`, `metric.unit`, `metric.direction` |
| Timing boundary | `metric.timing_boundary` |

Evidence classes split into two groups:

- Target-eligible primaries: `MEASURED_BENCHMARK_ARTIFACT`,
  `MEASURED_ISSUE_RECEIPT`, `MEASURED_OFFICIAL_DOCS`, `PAPER_MEASURED`.
  They must also carry a passed numerical gate with output parity.
- Quarantine: `VENDOR_MARKETING`, `ANONYMOUS_SCREENSHOT`,
  `AGGREGATOR_SECONDARY`, `UNVERIFIED`. Quarantined claims are recorded for
  visibility but can never become the comparison incumbent or set a target.
  Vendor marketing, anonymous screenshots, and unmatched benchmark shapes are
  never treated as SOTA evidence.

## Comparator verdicts

The scanner first partitions observations by every binding dimension except
the measured value and source provenance. The exact benchmark cell therefore
includes checkpoint and revision, quality mode, hardware, topology, batch,
structured precision and codecs, numerical gate, compute/KV routes, fabric,
power/clocks/thermal state, B/context/output distribution, prefix cache,
speculation, TTFT/ITL SLO, statistical method/sample/confidence level, metric
identity, direction, unit, and timing boundary. Confidence interval endpoints
and the metric value are measured outcomes; every other declared comparator
dimension is part of the cell. Different cells are never ranked or compared.

Within each cell, the target baseline is the best target-eligible metric value
(maximum for `higher_is_better`, minimum for `lower_is_better`). Equal values
use the lexicographically smallest deterministic `observation_id`; source list
order and `source_id` order cannot change the result. The winner is the cell's
`BASELINE`, and other observations in that same cell are `COMPARABLE` to it.

The standalone comparator emits exactly one verdict with field-level reasons:

- `COMPARABLE` - every binding dimension matches; the metric value may be
  ranked against the incumbent.
- `PARTIAL` - model and full metric identity match, but at least one
  execution dimension differs (checkpoint, quality mode, hardware count,
  precision/codec/routes, fabric, power/clocks, topology including EP, batch,
  request distribution, context, prefix/speculation, SLO, or statistics).
  Recorded, never target-setting.
- `INCOMPARABLE` - a core identity field differs (different model, different
  metric name/unit/direction/timing boundary) or any binding fact is missing
  on either side. Missing facts are never normalized into invented values;
  the reason string names the absent field.
- `BASELINE` - the deterministic optimum selected inside one exact cell (or a
  deterministic anchor when a cell contains quarantine-only evidence).

The 110% target (`target_value = round(baseline_value x 1.10, 4)`) is emitted
once for every exact cell containing target-eligible evidence. For
`lower_is_better` metrics the target relation flips explicitly to
`target_value <= baseline x 0.90` (at least 10% faster); direction and unit are
always stated on the record. A faster value from a different batch, context,
topology/fabric, power state, precision/route, prefix state, SLO, statistics,
quality mode, or timing boundary can set its own cell target but can never
replace another cell's baseline.

## Pipeline

Sources live in `performance/sota_sources.json`. Each source declares its
URL, publisher, document kind, evidence class, expected checkpoint, and an
extraction recipe. Three formats are supported:

- `sparkpipe_flat_v1` - machine-readable JSON benchmark artifact;
- `github_issue_receipt_v1` - GitHub issue whose body embeds a ```json
  receipt block plus issue `created_at`;
- `regex_text` - prose/paper text extracted by anchored patterns.

Regex extraction type names are config-validated per source (`int`, `float`,
`str`, or `bool`); an unknown name is a clear configuration error. JSON parsing,
schema validation, calculations, and serialization all reject non-finite
numbers (`NaN`, `Infinity`, and `-Infinity`).

Source registry validation is fail-closed before any source field is used.
Missing or malformed source id, URL, document kind, publisher, evidence class,
expected checkpoint, or extraction recipe errors name the model index, source
index, and source id when one was validly supplied; raw `KeyError` is never an
accepted configuration failure.

Commands:

```sh
# deterministic offline rebuild from captured envelopes (no network)
python3 tools/update_sota_ledger.py --mode offline \
    --envelope-dir examples/sota_observations

# production daily refresh straight from the primary sources
python3 tools/update_sota_ledger.py --mode fetch \
    --cache-dir cache/sota_envelopes

# verify the committed ledger matches regeneration
python3 tools/update_sota_ledger.py --check
```

Any fetch failure, unknown envelope, incomplete payload, schema violation,
or checkpoint mismatch aborts loudly with per-field reasons instead of
writing a weakened ledger.

## Determinism and provenance

Offline rebuilds read no wall clock: `generated_at` is derived from the
maximum envelope retrieval time, iteration order is sorted everywhere, and
two rebuilds are byte-identical. `tests/test_sota_ledger.py` asserts this and
also asserts the committed `performance/sota_ledger.jsonl` equals the
offline regeneration; refresh it with:

```sh
python3 tests/test_sota_ledger.py --update-committed
```

The ledger currently committed to the repository was generated offline from
the captured envelopes under `examples/sota_observations/`. Those envelopes
are labeled snapshots of the declared source shapes used to make tests
deterministic; they are not a substitute for the daily network refresh,
which rebinds fresh `retrieved_utc` values and re-runs the comparator. The
offline ledger and adversarial fixtures are not live performance claims.

## Ledger shape

`performance/sota_ledger.jsonl` contains one JSON object per line:

- `ledger_header` - generator identity, mode, generated time, target policy;
- `ingestion_rejected` - source plus field-level rejection reasons;
- `observation` - the validated observation, its comparison verdict with
  field-level reasons, exact `benchmark_cell_id`, `target_eligible`, and
  `sets_target`;
- `target` - baseline observation id/url/value plus the explicit factor,
  exact `benchmark_cell_id`, relation (`>=`/`<=`), rounded target value, and
  metric direction/units;
- `target_unavailable` - model with no comparable target-eligible cell;
- `ledger_summary` - counts and per-model arrays of exact-cell targets.

An external SparkPipe result never becomes a claim about SparkPipe
performance; it only sets the bar the next measurement must beat by 10%.

## Development-dashboard slice

The development dashboard presents B1, B8, and B64 at exactly 32,768 prompt
tokens for every supported model, with prefill and output throughput shown
separately. Each cell lists the best accepted SparkPipe value and the exact
public SOTA value, its 110% target, source, publication/retrieval date,
hardware, and accelerator count.

Only a target-eligible observation with the exact fixed workload, matching
metric/unit/direction, disabled prefix cache, and passed output parity can
populate a public value. A complete comparison receipt is still required to
compute a gap. Missing, shorter-context, or otherwise mismatched values render
as `N/A`; the UI never extrapolates them.
