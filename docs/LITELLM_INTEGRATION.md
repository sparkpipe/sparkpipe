# LiteLLM integration (operator ask 2026-09-05)

LiteLLM proxy in front of the fleet's per-model `sparkpipe_model_api`
instances: one OpenAI-compatible endpoint that routes by model name,
serves the admin page at `/ui`, and gives the fleet a standard client
surface (any OpenAI SDK / litellm client).

Verified end-to-end locally (litellm 1.74.0, python 3.13 venv at
`~/sparkpipe/litellm-venv`, two mock backends standing in for model
APIs): `/ui` HTTP 200, `/v1/models` lists every registered arm,
`/v1/completions` routes each model name to its own backend with
byte-identical determinism across calls, and
`tools/model_api_smoke.py` passes THROUGH the proxy. The chat endpoint
carries the same routing; the local proof used completion-shaped mocks.

## Wiring

1. Generate the config from the fleet registry (one entry per serving
   arm; the model_api for each deployment listens on its own port):

       python3 tools/generate_litellm_config.py \
         --pair glm53flash.bf16.tp16=http://spark0:8433 \
         --pair qwen27b.nvfp4a16.tp4=http://spark0:8434 \
         --out /tmp/litellm_fleet.yaml

   or `--registry models.json` with
   `[{"name": "<arm>", "base_url": "http://host:port"}, ...]`.

2. Run the proxy (repo venv pins litellm 1.74 on python 3.13 — newer
   litellm requires a PostgreSQL database for any auth path, and
   python 3.14 breaks uvloop):

       export SPARK_API_KEY=sk-...          # backend key if the fleet wants one
       ~/sparkpipe/litellm-venv/bin/litellm \
         --config /tmp/litellm_fleet.yaml --port 4000 --host 0.0.0.0

3. Use it: admin page `http://<host>:4000/ui`, OpenAI-compatible
   `http://<host>:4000/v1` (models, completions, chat/completions).
   Requests carry `Authorization: Bearer <key>` when the config sets
   `general_settings.master_key`; the generated config omits it for
   config-only operation (add it plus a Postgres `database_url` if
   virtual keys / spend tracking are wanted).

4. Smoke everything through the proxy with the standard tester:

       python3 tools/model_api_smoke.py --endpoint http://<host>:4000

## Notes

- The generated `litellm_params` use the `openai/` provider with
  `api_base` pointing at each model API's `/v1` — our API speaks the
  OpenAI shape natively.
- Serving arms register under their canonical arm names
  (docs/STAGEPACK_NAMING.md), so a request's `model` field is the
  stagepack arm: routing, naming, and the weightd identity all agree.
- `drop_params: true` lets OpenAI-SDK clients send parameters our API
  ignores without erroring.
