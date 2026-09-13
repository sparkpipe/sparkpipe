# Single-box multi-model serving

Run SEVERAL residentd deployments side by side on ONE box, each with its
own ports, runtime root, KV backing directory, and log stream.

## Files

- `multi_model_serving.sh` - the supervisor (start/stop/status/health).
- `models.conf` - one MODEL block per deployment. Every field is real:
  it lands in the rendered schema-2 `model_resident.json` or in the
  daemon environment.
- `config/` - per-model serving-adapter configuration JSONs referenced
  by the blocks (`ADAPTER_CONFIG`).

## The isolation contract (box-scale COORDINATION.md)

| Resource | Rule |
| --- | --- |
| Control endpoint | One TCP port per model; preflight refuses overlaps and occupied ports |
| Transport | Base + 8 consecutive ports per model, checked like the above |
| Runtime root | Private tree per model; configs, logs, KV live under it |
| KV backing directory | Per model, byte-capped via `KV_BACKING_MAX_BYTES` |
| Big models | At most ONE tier=big at a time; starting one stops the running big model, always-on models stay up |

## Usage

    fleet/singlebox/multi_model_serving.sh init      # render + preflight
    fleet/singlebox/multi_model_serving.sh start     # all models
    fleet/singlebox/multi_model_serving.sh start dsv4-flash-tp4
    fleet/singlebox/multi_model_serving.sh status
    fleet/singlebox/multi_model_serving.sh health qwen38-tp4
    fleet/singlebox/multi_model_serving.sh stop [MODEL...]

Overrides: `SINGLEBOX_MODELS_CONF`, `SINGLEBOX_RESIDENTD`,
`SINGLEBOX_RUN_DIR`.

## Measurement hygiene

Every daemon starts with `SPARKPIPE_RELEASE_GENERATION` (UTC stamp),
`SPARKPIPE_RELEASE_GIT_COMMIT`, and `SPARKPIPE_RELEASE_ID=singlebox-<model>`
in its environment, so any measurement taken through a control endpoint is
attributable to an exact source revision - the repo's rule for accepted
milestones.

## Prerequisites

- `build/sparkpipe_model_residentd` built (`make build/sparkpipe_model_residentd`).
- The adapter/driver/transport shared objects each block names exist
  relative to the repository root (preflight checks them).
- Adapter config JSONs under `fleet/singlebox/config/` matching each
  model's stage pack and topology (copy from the family deploy scripts,
  e.g. `qwen38_tp4_deploy.sh`'s rank-0 config, adjusted for TP1).
