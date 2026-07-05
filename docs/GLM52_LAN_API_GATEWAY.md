# GLM52 LAN API Gateway

SparkPipe should expose one stable client API at Spark0 and keep the PP13 device
driver private behind that gateway.

## Topology

```text
client / Centaur / demo UI
    -> Spark0 API gateway on management LAN
        -> PP13 ingress queue
            -> spark0 stage 0
            -> spark1 stage 1
            -> ...
            -> sparkc stage 12
        <- final token events from sparkc to spark0
    <- HTTP response or SSE stream
```

External clients do not talk to spark1..sparkc. Those ranks only accept device
pipeline traffic from their expected neighbors.

## LAN API

The gateway binary is:

```text
build/sparkpipe_glm52_http_gateway
```

Local development:

```sh
build/sparkpipe_glm52_http_gateway \
    --bind 127.0.0.1 \
    --port 8080 \
    --api-key-file /path/to/sparkpipe_api_key
```

Management LAN:

```sh
build/sparkpipe_glm52_http_gateway \
    --bind <spark0-management-lan-ip> \
    --port 8080 \
    --api-key-file /home/spark0/sparkpipe_runtime/API_KEY
```

Supported endpoints:

```text
GET  /
GET  /health
OPTIONS /v1/chat/completions
OPTIONS /v1/completions
OPTIONS /v1/messages
POST /v1/chat/completions
POST /v1/completions
POST /v1/messages
```

`/` serves a small demo UI. `/v1/chat/completions` and `/v1/completions` use
OpenAI-compatible request JSON. `/v1/messages` uses Anthropic-compatible
request JSON. Streaming requests receive `text/event-stream`.

The demo UI supports prompt text plus multiple text-file uploads. The browser
reads files client-side and sends them as request JSON:

```json
{
  "messages": [{"role": "user", "content": "Use the attachment."}],
  "files": [{"filename": "notes.txt", "content": "file text"}],
  "stream": true
}
```

The C compatibility layer folds uploaded text into explicit prompt sections
before tokenization. Browser CORS preflight is supported for the three public
generation routes.

Until the PP13 backend is attached, the gateway returns fail-closed errors:

```text
401 missing or invalid bearer token
503 backend_unavailable
```

It does not fabricate tokens.

## Public sparkpipe.ai Access

The public website should not connect directly to random Spark rank ports.

Recommended production path:

```text
sparkpipe.ai / 62.112.9.102
    TLS termination + auth + rate limits
        -> WireGuard or mTLS tunnel
            -> spark0 management LAN gateway
```

Spark0 should bind only to a private LAN or tunnel address. The public website
can expose OpenAI-compatible and Anthropic-compatible routes and proxy them to
Spark0. Browser clients should use the website origin; server-side clients may
use the LAN gateway directly if they are on the trusted management network.

## Streaming

The client-facing stream is SSE:

```text
Content-Type: text/event-stream
event: token
data: {...}

event: done
data: {...}
```

The final PP13 rank must send token events back to Spark0. Spark0 owns the HTTP
connection and streams those events to the client. This keeps the external
control plane simple: one ingress and one egress node.

## Production Boundary

This gateway is not the device transport. It is safe to run over the 10 Mbps
management LAN because prompts and generated token events are small compared
with model traffic.

The stage hidden-state transport remains separate:

```text
spark0 -> spark1 -> ... -> sparkc
```

That transport must stay persistent, binary, and device-oriented. It must not
use HTTP, shell commands, `rsync`, `scp`, or files for per-token stage traffic.
