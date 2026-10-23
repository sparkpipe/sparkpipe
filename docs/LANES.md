# Production lanes

Every model the swap controller serves runs from one record in
`deployment/lanes.json`. Nothing on the nodes or the hub is edited by hand:
the files a lane runs from are rendered from its record, and an audit says
whether the fleet still matches.

## The record

| Field | Meaning |
|---|---|
| `launcher` | `stage_lane` (started by the `lane.sh` that `tools/stage_lane.sh` renders) or `k3_lane` (started by the swap entry's own commands) |
| `model` | the `tools/stage_lane.py` MODELS entry: packs, adapter, contract, KV page bytes |
| `lane` | weightd mesh lane; ports and the collective identifier follow from it |
| `runtime_root`, `weightd_socket` | per-node root (`{host}` is substituted) and weightd socket |
| `sequences`, `rows`, `shared_rows` | resident sequences, rows per wave, rows of the shared prefill half |
| `context_tokens` | positions one sequence may hold; `kv_pages` is this over the model's KV block (64 tokens, DSV4.1 256) |
| `context_reason` | required while `context_tokens` is below the contract maximum: what holds it there |
| `kv_physical_pages` | pages the scheduler may reserve; weightd must be able to back them |
| `kv_logical_pages` | pages including those parked on the node's NVMe; at least `sequences` full-length sequences |
| `kv_backing_bytes`, `kv_snapshot_bytes` | per-node NVMe caps for parked pages and prefix snapshots |
| `state_budget_bytes`, `state_memory_bytes` | rank state budget; recurrent checkpoint memory (default twice the minimum) |
| `taps` | layer taps a hub drafter reads, passed to every rank's adapter |
| `expert_pool_bytes`, `memory_max` | `lane.sh` residentd environment for stage lanes |
| `tokenizer_sha256` | the compiled tokenizer the hub API loads |
| `draft_server` | the hub drafter the API asks for drafts |

`weightd` at the top holds the device ceiling and KV reserve every node's
weightd runs with.

## Rules

`tests/test_lane_records.py` holds every record to these:

- every swap-controller model has exactly one record;
- `context_tokens` is the contract maximum, or lower with a
  `context_reason`, never higher;
- one full-length sequence fits the weightd KV reserve, and so do the
  physical pages: the scheduler is never promised pages weightd cannot back.
  A model whose native maximum does not fit stays lower with its reason;
- the record renders: physical pages hold one sequence, logical pages hold
  every resident sequence at full length, the backing holds the parked
  pages, recurrent checkpoint memory holds two per sequence plus two;
- the stage lanes' swap entries in `deployment/hub/model_swap.json` are the
  ones the record renders, and the chat layer's `context_tokens` in
  `deployment/hub/chat_frontend.json` is the record's.

Concurrency follows from the pages: the batch engine admits a request only
while its prompt and output budget fit the free physical pages, so at full
length fewer sequences run at once and the rest queue; `sequences` is the
ceiling for short requests.

## Commands

```bash
python3 tools/lane_config.py render kimi-k3 /tmp/k3-lane
python3 tools/lane_config.py audit
python3 tools/lane_config.py push kimi-k3
```

`render` writes the node `deployment.json`, one `adapter.<host>.json` per
rank, the hub `model_resident.json` (the node file plus the tokenizer and the
draft server), the swap entry and, for stage lanes, one `lane.<host>.sh` per
rank.

`audit` renders every record and compares, on all 16 nodes and the hub:

- each lane's `deployment.json`, `config/adapter.json` and `lane.sh`;
- each API's `model_resident.json`;
- the hub files with a repository source: `model_swap.json`, the chat
  layer's `config.json` and the LiteLLM configurations;
- the weightd ceiling and KV reserve in every node's `weightd.log`;
- lane roots and API directories that no record names.

It prints one line per finding and exits non-zero when there is any.

`push` writes one record's rendered files to the nodes and the hub, keeps
every replaced file as `<file>.prev-<UTC>`, and audits the record. It does
not touch binaries, packs or KV, so a lane restart picks the change up.

`SPARK_LANES=<file>` renders, audits or pushes from another copy of the
records, which is how a gate tries a raised record before it is committed.
`tools/stage_lane.sh` with `STAGE_RECORD=<swap id>` and `tools/k3_lane.sh`
with `K3_RECORD=kimi-k3` render from the record when they set a lane up;
their flag form stays for experiment lanes in their own roots.

## Changing a lane

Edit the record, run `tests/test_lane_records.py`, merge, then push and
restart the lane through the swap controller. A raised `context_tokens`
ships only after the lane's gate: the engine reports the new limit, short
outputs match the previous build, a needle near the new maximum is found,
and chat memory qualifies (docs/CHAT_MEMORY.md). After any hold that touches
a production lane, `audit` must report no findings.
