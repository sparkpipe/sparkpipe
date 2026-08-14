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
    --api-key-file /home/spark0/.config/sparkpipe/API_KEY \
    --service-backend-so /home/spark0/sparkdata/glm52.fp8.pp13/lib/libglm52_pp13_service_backend.so \
    --model-quantization fp8 \
    --moe-pack-root /home/spark0/sparkdata/glm52.fp8.pp13/packs/moe \
    --transport-so /home/spark0/sparkdata/glm52.fp8.pp13/lib/hidden_transport.so \
    --driver-so /home/spark0/sparkdata/glm52.fp8.pp13/lib/model_driver.so \
    --program glm52.pp13.rank.production \
    --node-target cuda.sm121.glm52.pp13.fp8 \
    --tokenizer /home/spark0/sparkdata/glm52.fp8.pp13/tokenizer/tokenizer.sptok \
    --max-active 1024 \
    --port-base 52100
```

The service backend is a shared object loaded by the gateway. It receives all
PP13 production paths through the backend configuration and exposes the first
missing production blocker through `/health`.

The PP13 service backend owns the C service runtime, scheduler, request API,
prefix cache, KV arena, tokenizer handle, and final-event listener. It does not
run Python, shell scripts, or file handoff in the request path.

Current strict behavior:

```text
backend_ready=1 means the backend loaded and can report status
pp13_ready=1 means requests may enter the production PP13 service
pp13_ready=0 means requests fail closed with backend_unavailable
first_blocker names the first missing production component
```

Current expected blocker until the last device-driver bridge lands:

```text
rank0 token-id input bridge is not connected to distributed PP13 driver
```

That is not a gateway or API problem. It means Spark0 can accept service state
and final-stage events, but it still lacks the production C/CUDA bridge that
converts request token IDs into prefill/decode frames for the resident PP13
rank runner.

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

It does not fabricate tokens. A backend may be attached and still return
`pp13_ready=0` if the production rank runner, tokenizer, transport, driver, or
rank0 token-id driver bridge is incomplete.

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
