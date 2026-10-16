# Chat memory stage B on K3, 2026-10-15

- `production-8-turns.out`: `tools/chat_memory_live.py --think 30` against the
  production chat layer (port 8433) after the stage B deploy at 00:15Z, with
  the store's records for that chat at the end.
- `regroup-ledger-150-tokens.out`: the same driver against a second chat-layer
  instance (port 8439, same code, K3 only) with `ledger_tokens` 150 and
  `group_size` 3, so the curator regroups after a few turns.
- `api-requests.txt`: every K3 model API request of both runs, from
  `request_measurements` in `/srv/workspace/sparkpipe/logs/k3-kv-api.log`:
  request id, priority (1 foreground, 0 curator, stem and pre-warm),
  prompt tokens, cached prompt tokens, engine completed.

Results are in `docs/CHAT_MEMORY.md`, "Implementation (stage B)".
