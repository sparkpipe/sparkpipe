# The memory protocol

Conversation memory is a separate program, licensed separately from sparkpipe. The chat layer reaches it only over localhost, through the generic connector in `tools/serving/memory_connector.py`. The chat layer holds none of memory's own code or formats: no ledger, store, notes, curator, parsers or tool definitions.

## Turning memory on
The chat layer's config turns memory on with a `memory` block:

| key | means |
|---|---|
| `url` | the program's address, such as `http://127.0.0.1:8435` |
| `host_listen` | where the chat layer serves the program its engine services, such as `127.0.0.1:8436` |
| `token_file` | the bearer token both sides send, at least 32 characters, readable by its owner only, relative to the chat layer's directory |
| `timeout_seconds` | how long a call may take, 30 by default |

Without the block, memory is off.

When the program is down, stops answering or speaks another protocol:
- the chat layer serves requests on their raw history;
- it logs `memory_unavailable` once per retry;
- it tries the program again after 5 seconds.

A failure inside one `prepare` is logged as `memory_skipped`, and that request is served raw.

## Transport
- Both directions are HTTP/1.1 on localhost: `POST /v1/<method>` with a JSON object for the body, and a JSON object back.
- Every request carries `Authorization: Bearer <token>`. A side refuses a request without it with 401.
- A failure is a non-200 status with `{"error": {"code", "message"}}`. The error may carry `status`, the HTTP status the chat layer gives its own client. A failure that started in the chat layer's engine services keeps that status, such as 503 while the engine is switching.
- Each side serves with handler cancellation: when a caller gives up on a request, the work behind it stops, the engine stream included.

## Hello
`POST /v1/hello` with `{"protocol": 1, "host_url", "host_methods"}` answers `{"protocol": 1, "name", "version", "methods": [...]}`. The connector logs the program's name and version with `memory_available`, so the log shows which program and version served:
- the program refuses another protocol with 409 `protocol_mismatch`;
- until it has had a hello, the program answers every other method with 409 `hello_first`. The connector then says hello and retries the call once, so a restarted program is picked up at once.

`GET /health` on the program answers `{"ok": true, "protocol": 1, "name", "version"}` without a token.

## Methods, protocol 1
A `model` is `{"id", "context_tokens", "reasoning": {"start", "end", "close"} or null}`. The program keeps any per-model settings in its own config.

**The program's methods, which the chat layer calls:**

| method | request | response |
|---|---|---|
| `prepare` | model, headers (or null), messages, tools, kwargs, target, identity (or null) | command (a name and argument, or null), identity, messages, tools, session (a handle or null), memory_off, trace |
| `run_command` | model, identity, messages, name, argument, memory_off | reply |
| `import` | model, headers, messages, tools, kwargs, batch | the import's counts |
| `pending` | model | queued, running, foreground |
| `foreground` | model, requests | — |
| `cancel_all` | — | — |
| `forget` | scope | — |
| `expire` | wiped, held_until, now | — |
| `session.claims` | session, call | claimed |
| `session.can_continue` | session | can |
| `session.continue_with` | session, content, reasoning, calls | messages |
| `session.finished` | session, plan_messages, content, reasoning, visible_calls, engine_finish, prompt_ids, token_ids | — |
| `session.release` | session | — |

- **prepare** does everything memory needs before the chat layer renders a request. The chat layer sends:
  - the headers, for a chat's signed identity;
  - or `identity`, the API's `[scope, session]`.

  It gets back the messages and tools to render: the program's own tools are among them. It also gets the fields to log in `trace`, and either a command to run or a session handle.
- **Tools stay opaque.** For every tool call the model makes, the chat layer asks `session.claims`. It hands each claimed call to `session.continue_with` and renders the messages that come back, without knowing what the tool does.
- **Notices** are sent in order, without waiting for an answer:
  - `foreground` carries the chat layer's count of requests in flight for a model, so a lost notice heals at the next one;
  - `cancel_all`;
  - `session.release`;
  - `expire`, sent when the working-state directive changes and on the chat layer's sweep: the program deletes every identity whose last use falls before `wiped`, or when `now` is 24 hours or more past the later of its last use and `held_until`, the same rule as the chat layer's own working state.
- **forget** deletes every identity whose scope is `scope` or starts with `scope/` (its members), sessions included, before it answers. The chat layer calls it for `DELETE /v1/cache` and reports memory as deleted only after the answer.
- A session the chat layer never releases is dropped after the program's `session_seconds`.
- The program keeps each identity's memory apart. The chat layer passes the identity on every request and never reads memory's store.
- `POST /v1/sparkpipe/memory/import` on the chat layer hands a client's conversation to `import`, with its headers, after resolving the model. The body may set `timeout_seconds`, how long the chat layer waits for the program (3,900 by default); a long history can need more.

**The chat layer's engine services, which the program calls on `host_url`:**

| method | request | response |
|---|---|---|
| `model_ready` | model | ready |
| `render_ids` | model, messages, tools, kwargs, suffix, start | ids, reasoning_open |
| `continuation_ids` | model, messages, tools, kwargs, prompt_ids, generated_ids, instruction, suffix, start | ids (or null), reasoning_open |
| `cut_ids` | model, messages, tools, kwargs, sentinel | ids (or null) |
| `count` | model, text | tokens |
| `complete` | model, ids, max_tokens, reasoning_open, kind, trace | text |

`complete` runs at background priority through the same sequence gate as every other request.

Within a protocol, methods only add. A caller that meets 404 `unsupported` treats the feature as absent.

## Tests
`tests/test_memory_connector.py` drives the connector against a fake program and checks:
- the raw fallback, and the retry after it;
- the token both ways;
- the host services' refusals and their own status;
- errors and timeouts;
- that an abandoned host call stops;
- that notices keep their order;
- session calls;
- that a restarted program is greeted again;
- that another protocol counts as unavailable.
