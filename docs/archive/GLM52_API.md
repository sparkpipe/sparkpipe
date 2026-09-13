# GLM52 API

This document describes the GLM-5.2 API surface that is currently in the
SparkPipe tree. It is written as an operator and integrator contract, not as a
benchmark report.

The API has four layers:

1. Request API: queueing, prefix cache, KV ownership, prefill and decode batch
   decisions.
2. Serving engine: prompt token submission, token streaming events, and decode
   callback integration.
3. Service API: client sessions, request id mapping, event rings, and a binary
   frame protocol.
4. Compatibility API: OpenAI and Anthropic JSON-to-text request translation.

The device-driver side is separate:

1. PP13 runtime planning describes the fixed 13-rank GLM-5.2 topology.
2. The resident decode-stage production runner submits one rank's frame to the
   GLM-5.2 device driver with hidden transport and KV block-table state.

## Stable External Surface

External projects should design to the service and compatibility APIs, not to
validation tools, benchmark harnesses, or temporary rank-runner details.

Stable caller-facing contracts:

```text
SparkServiceRuntime
SparkServiceSubmitText
SparkServiceSubmitTokenIds
SparkServicePump
SparkServicePopEvent
SparkServiceCancelRequest
SparkGlm52CompatPrepareOpenAiJson
SparkGlm52CompatPrepareAnthropicJson
SparkGlm52CompatSubmitOpenAiJson
SparkGlm52CompatSubmitAnthropicJson
sparkpipe_glm52_http_gateway
```

Stable deployment-facing contracts:

```text
SparkGlm52Pp13RuntimeBuildFixedStagePlan
SparkGlm52Pp13RuntimeBuildRankPlan
SparkGlm52Pp13RuntimeValidateRankPlan
SparkGlm52Pp13RuntimeValidateStageFp8PackFiles
SparkResidentDecodeStageProductionRunnerInitialize
SparkResidentDecodeStageProductionRunnerSubmit
```

Unstable internal surfaces:

```text
validation executables
benchmark scripts
fixture-only prompt tools
temporary local pipeline gates
CUDA kernel tuning entry points
rank-specific bring-up helpers
```

The intended final shape is that other projects can integrate against the
stable service API now, while SparkPipe continues improving the implementation
behind that surface. A caller should not need to know whether the backend is a
single-Spark local preflight, a 13-rank PP13 ring, FP8, DSpark-enabled, or MTP
enabled. Those choices should appear as service capability, event, and
configuration details rather than as different external APIs.

## Scope

The API is pure C. It is not an HTTP server yet.

Current compatibility entry points parse OpenAI-style and Anthropic-style JSON
into SparkPipe service submissions. A network daemon can sit above this layer,
but the HTTP socket handling, auth policy, streaming HTTP response formatting,
and TLS termination are intentionally not part of this API surface.

Production prompt inference still requires all of these pieces to be connected:

- tokenizer or token-id input
- prompt prefill through all 78 layers
- resident KV ownership retained across prefill and decode
- PP13 decode loop for generated tokens
- token-id to text output

## Header Map

Public headers:

```text
include/sparkpipe/spark_request_api.h
include/sparkpipe/spark_serving_engine.h
include/sparkpipe/spark_service.h
include/sparkpipe/spark_glm52_compat_api.h
include/sparkpipe/spark_glm52_pp13_runtime.h
modules/glm52_resident_decode_stage/include/sparkpipe/spark_glm52_resident_decode_stage_production_runner.h
include/sparkpipe/spark_http_gateway.h
```

The LAN and public website gateway contract is described in:

```text
docs/GLM52_LAN_API_GATEWAY.md
```

Important tests:

```text
tests/test_glm52_service.c
tests/test_glm52_compat_api.c
tests/test_glm52_pp13_runtime.c
tests/test_glm52_resident_decode_stage_production_runner.c
```

## Request API

The request API owns request state transitions below the serving layer. It is
defined in:

```text
include/sparkpipe/spark_request_api.h
```

It works with token ids, not text.

Request states:

```text
FREE
QUEUED_PREFILL
RUNNING_PREFILL
READY_DECODE
RUNNING_DECODE
COMPLETED
CANCELLED
WAITING_PREFIX_COHORT
READY_SPECULATIVE_VERIFY
RUNNING_SPECULATIVE_VERIFY
```

Dispatch kinds:

```text
PREFILL
DECODE_BATCH
PREFILL_BATCH
SPECULATIVE_VERIFY_BATCH
```

The request API also exposes KV block-table views for execution. The execution
layer must treat those views as the physical KV ownership contract for the
request lanes. It should not reconstruct cache state through a side channel.

Default features include:

```text
JIT_KV_PREFETCH
DECODE_BATCHING
PREFIX_COHORTING
PREFILL_BATCHING
QUEUE_AWARE_PREFIX_CACHE_EVICTION
```

Optional flags include:

```text
ASYNC_JIT_KV_PREFETCH
DSPARK_SPECULATIVE_DECODE
MTP_COMMIT
```

## Serving Engine

The serving engine is defined in:

```text
include/sparkpipe/spark_serving_engine.h
```

It sits above the request API and below the service API.

It supports two submit forms:

```c
SparkStatus SparkGlm52ServingSubmitTokenIds(
    SparkServingEngine *engine,
    const SparkServingSubmitTokenIdsRequest *request,
    SparkServingSubmitResult *result);

SparkStatus SparkGlm52ServingSubmitText(
    SparkServingEngine *engine,
    const SparkServingSubmitTextRequest *request,
    SparkServingSubmitResult *result);
```

Text submission requires a configured tokenizer in the serving engine. Token-id
submission is the lower-risk production boundary while tokenizer integration is
being validated.

The serving engine calls the configured prefill and decode callbacks. Those
callbacks are where rank execution, local pipeline testing, or a future
distributed PP13 pipeline is attached.

The serving runtime contract must advertise production capabilities before the
default engine accepts it as production. Required contract flags are:

```text
PREFILL_ACCEPTS_BULK_TOKEN_WINDOWS
PREFILL_WRITES_RESIDENT_KV
DECODE_CONSUMES_RESIDENT_KV
DECODE_RETURNS_TOKEN_IDS
USES_REQUEST_KV_BLOCK_TABLES
INTERNAL_BATCHING_ENABLED
OFFICIAL_DSA_INDEXSHARE
BOUNDED_LONG_CONTEXT_ATTENTION
INDEXSHARE_STAGE_BOUNDARY_STATE
MLA_COMPRESSED_KV_CACHE
```

The following are useful connected-runtime flags but are not part of the base
required set:

```text
JIT_KV_PREFETCH_CONNECTED
OVERLAPPED_STAGING_READY
```

Events emitted by the serving engine:

```text
REQUEST_ACCEPTED
PREFILL_PROGRESS
TOKEN
REQUEST_COMPLETED
REQUEST_CANCELLED
ERROR
BACKPRESSURE
```

## Service API

The service API is defined in:

```text
include/sparkpipe/spark_service.h
```

It adds client sessions, client request ids, event forwarding, and frame
handling on top of the serving engine.

### Runtime Initialization

The caller owns all backing storage. The runtime does not hide heap allocation
inside initialization.

Required storage in `SparkServiceConfiguration`:

```text
serving_engine
client_sessions
client_session_capacity
request_maps
request_map_capacity
event_ring
event_ring_capacity
```

Basic initialization:

```c
SparkServiceConfiguration configuration;
SparkServiceRuntime service;

memset(&configuration,0,sizeof(configuration));
configuration.abi_version = SPARK_SERVICE_ABI_VERSION;
configuration.descriptor_bytes = SPARK_SERVICE_CONFIGURATION_DESCRIPTOR_BYTES;
configuration.flags = SPARK_SERVICE_CONFIGURATION_DEFAULT_FLAGS;
configuration.default_pump_dispatch_steps = SPARK_SERVICE_DEFAULT_PUMP_DISPATCH_STEPS;
configuration.request_id_base = SPARK_SERVICE_DEFAULT_REQUEST_ID_BASE;
configuration.serving_engine = &serving_engine;
configuration.client_sessions = client_sessions;
configuration.client_session_capacity = client_session_capacity;
configuration.request_maps = request_maps;
configuration.request_map_capacity = request_map_capacity;
configuration.event_ring = event_ring;
configuration.event_ring_capacity = event_ring_capacity;

status = SparkServiceInitialize(&service,&configuration);
```

### Client Lifecycle

Register a client:

```c
SparkServiceClientId client_id;

status = SparkServiceRegisterClient(&service,user_cookie,&client_id);
```

Disconnect a client:

```c
status = SparkServiceDisconnectClient(&service,client_id);
```

Client ids are service-local. The caller can store an external connection or
tenant id in `user_cookie`.

### Text Submission

Text submission is:

```c
SparkServiceSubmitTextRequest request;
SparkServiceSubmitResult result;

SparkServiceInitializeSubmitTextRequest(&request);
request.client_id = client_id;
request.client_request_id = client_request_id;
request.sequence_id = sequence_id;
request.text = prompt_text;
request.text_bytes = prompt_text_bytes;
request.output_token_budget = output_token_budget;
request.max_prefill_tokens_per_step = max_prefill_tokens_per_step;

status = SparkServiceSubmitText(&service,&request,&result);
```

`client_request_id` is supplied by the caller and is the cancellation/event
correlation id visible at the service boundary. `serving_request_id` and
`serving_request_handle` are returned in `SparkServiceSubmitResult`.

### Token-id Submission

Token-id submission avoids tokenizer dependency in the service path:

```c
SparkServiceSubmitTokenIdsRequest request;
SparkServiceSubmitResult result;

SparkServiceInitializeSubmitTokenIdsRequest(&request);
request.client_id = client_id;
request.client_request_id = client_request_id;
request.sequence_id = sequence_id;
request.token_ids = token_ids;
request.token_count = token_count;
request.output_token_budget = output_token_budget;

status = SparkServiceSubmitTokenIds(&service,&request,&result);
```

The service does not copy token ids for the caller after submission. The serving
engine/request API owns its own request-token storage once the submit succeeds.

### Pumping And Events

Run the service:

```c
SparkServiceStats stats;

status = SparkServicePump(&service,max_dispatch_steps,&stats);
```

Pop events:

```c
SparkServiceEvent event;

while ( SparkServicePopEvent(&service,&event) == SPARK_STATUS_OK )
    handle_event(&event);
```

Service events mirror serving events and add client ids:

```text
REQUEST_ACCEPTED
PREFILL_PROGRESS
TOKEN
REQUEST_COMPLETED
REQUEST_CANCELLED
ERROR
BACKPRESSURE
CLIENT_CONNECTED
CLIENT_DISCONNECTED
STATS
```

For token events:

```text
event.token_id
event.token_index
event.token_count
event.client_id
event.client_request_id
event.sequence_id
```

### Cancellation

Cancel by client-visible id:

```c
status = SparkServiceCancelRequest(
    &service,
    client_id,
    client_request_id);
```

The service maps that to the serving handle and emits cancellation or error
events through the event ring.

## Binary Frame Protocol

The service API also defines a compact frame protocol. This is suitable for a
persistent local or fabric-facing daemon, but it is still just a C frame API.

Every frame starts with `SparkServiceFrameHeader`:

```text
magic              SPARK_SERVICE_FRAME_MAGIC
abi_version        SPARK_SERVICE_ABI_VERSION
descriptor_bytes   sizeof(SparkServiceFrameHeader)
kind               frame kind
flags              submit flags for submit frames
body_bytes         body byte count
client_id          service client id
client_request_id  caller request id
```

Request frame kinds:

```text
SUBMIT_TEXT
SUBMIT_TOKEN_IDS
CANCEL_REQUEST
PING
```

Response frame kinds:

```text
EVENT
SUBMIT_ACK
ERROR
PONG
```

Frame helpers:

```c
SparkServiceInitializeFrameHeader(&header,SPARK_SERVICE_FRAME_KIND_SUBMIT_TEXT);
status = SparkServiceValidateFrameHeader(&header,maximum_body_bytes);
status = SparkServiceHandleSubmitTextFrame(&service,client_id,&header,body,body_bytes,&result);
status = SparkServiceBuildEventFrame(&event,&event_header,&event_body);
```

Submit text frame body:

```text
SparkServiceSubmitTextFrameBody
text bytes immediately after the body struct
```

Submit token frame body:

```text
SparkServiceSubmitTokenIdsFrameBody
uint32_t token ids immediately after the body struct
```

## OpenAI And Anthropic Compatibility API

The compatibility API is defined in:

```text
include/sparkpipe/spark_glm52_compat_api.h
```

It translates JSON request bodies into `SparkServiceSubmitText` calls.
It does not implement HTTP, SSE, auth, rate limiting, or response JSON
formatting.

OpenAI `messages` and Anthropic `messages` are rendered with the GLM-5.2
`[gMASK]<sop>` chat template, including the reasoning-effort prefix and
assistant generation marker. OpenAI `prompt` is submitted unchanged for callers
that already rendered a model prompt or intentionally use raw completion mode.

The caller owns the prompt buffer:

```c
char text[SPARK_SERVICE_MAX_TEXT_BYTES];
SparkGlm52CompatTextRequest request;

SparkGlm52CompatInitializeTextRequest(&request,text,sizeof(text));
```

OpenAI-style preparation:

```c
status = SparkGlm52CompatPrepareOpenAiJson(
    json_text,
    json_bytes,
    &request);
```

Anthropic-style preparation:

```c
status = SparkGlm52CompatPrepareAnthropicJson(
    json_text,
    json_bytes,
    &request);
```

Submit in one call:

```c
status = SparkGlm52CompatSubmitOpenAiJson(
    &service,
    json_text,
    json_bytes,
    &request,
    &result);
```

or:

```c
status = SparkGlm52CompatSubmitAnthropicJson(
    &service,
    json_text,
    json_bytes,
    &request,
    &result);
```

### OpenAI Mapping

Supported request fields:

```text
prompt
messages[].role
messages[].content
priority
max_tokens
max_completion_tokens
```

If `messages` is present, the prompt text becomes:

```text
system: <content>
user: <content>
assistant: <content>
```

If `prompt` is present and `messages` is absent, the prompt text is the prompt
string as supplied.

`max_completion_tokens` is preferred over `max_tokens` when both are present.
The selected value becomes `output_token_budget`.

`priority` is an optional unsigned integer. A larger value has preference over
every ready request with a smaller value. Sparkpipe chooses execution width
from the highest ready priority class, packs that class first, and may use
otherwise-unused rows for lower-priority work without enlarging the selected
execution width. Omitting `priority`, or supplying zero, uses the default
priority.

### Anthropic Mapping

Supported request fields:

```text
system
messages[].role
messages[].content
messages[].content[].type == "text"
messages[].content[].text
priority
max_tokens
```

The prompt text becomes:

```text
system: <system>
user: <content>
assistant: <content>
```

Text content arrays are concatenated in order. Non-text blocks are ignored by
the current compatibility layer.

## PP13 Runtime API

The fixed 13-rank GLM-5.2 runtime plan is defined in:

```text
include/sparkpipe/spark_glm52_pp13_runtime.h
```

Constants:

```text
stage count:       13
layers per stage:  6
hidden dimension:  6144
quantization:      FP8_E4M3_8BIT
default port base: 52100
```

Build the global stage plan:

```c
SparkStagePlan stage_plan;
char error_buffer[512];

status = SparkGlm52Pp13RuntimeBuildFixedStagePlan(
    &stage_plan,
    error_buffer,
    sizeof(error_buffer));
```

Build one rank plan:

```c
SparkGlm52Pp13RuntimeRankPlan rank_plan;

status = SparkGlm52Pp13RuntimeBuildRankPlan(
    rank_index,
    max_active_sequence_count,
    SPARK_GLM52_PP13_RUNTIME_DEFAULT_PORT_BASE,
    SPARK_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT,
    &rank_plan,
    error_buffer,
    sizeof(error_buffer));
```

Rank host names use:

```text
spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc
```

Rank plans include:

```text
first_layer_index
layer_count
previous_rank_index
next_rank_index
listen_port
next_port
input_endpoint
output_endpoint
bytes_per_sequence
max_packet_bytes
```

Validate FP8 pack presence for a rank:

```c
status = SparkGlm52Pp13RuntimeValidateStageFp8PackFiles(
    &rank_plan,
    pack_root,
    error_buffer,
    sizeof(error_buffer));
```

Build a layer pack path:

```c
status = SparkGlm52Pp13RuntimeBuildFp8PackPath(
    pack_root,
    layer_index,
    pack_path,
    sizeof(pack_path));
```

## Resident Decode-stage Production Runner

The rank-level device-driver runner is defined in:

```text
modules/glm52_resident_decode_stage/include/sparkpipe/spark_glm52_resident_decode_stage_production_runner.h
```

This runner is the strict C boundary between rank orchestration and the GLM-5.2
resident decode-stage driver.

Default runner requirements:

```text
driver admission must accept the frame
input hidden transport session must be present
output hidden transport session must be present
KV block table must be present
program must reject file transport
program must reject shell transport
program must own resident state and KV cache
program must use stream-ordered execution
program must advertise validated latency
```

Required program flags:

```text
STREAM_ORDERED
DRIVER_OWNS_RESIDENT_STATE
DRIVER_OWNS_KV_CACHE
NO_HOST_STAGING
FIXED_FIRMWARE
VALIDATED_LATENCY
NO_DEVICE_MEMCPY
REQUIRES_HIDDEN_TRANSPORT
NO_FILE_TRANSPORT
NO_SHELL_TRANSPORT
BULK_PREFILL
```

Initialize:

```c
SparkResidentDecodeStageProductionRunnerConfiguration configuration;
SparkResidentDecodeStageProductionRunner runner;

memset(&configuration,0,sizeof(configuration));
configuration.abi_version =
    SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_ABI_VERSION;
configuration.descriptor_bytes =
    SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_CONFIGURATION_BYTES;
configuration.flags =
    SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DEFAULT_FLAGS;
configuration.driver_interface = driver_interface;
configuration.driver_instance = driver_instance;
configuration.program = program;
configuration.execution_stream = cuda_stream;

status = SparkResidentDecodeStageProductionRunnerInitialize(
    &runner,
    &configuration);
```

Submit:

```c
SparkResidentDecodeStageProductionRunnerDispatch dispatch;

memset(&dispatch,0,sizeof(dispatch));
dispatch.abi_version =
    SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_ABI_VERSION;
dispatch.descriptor_bytes =
    SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DISPATCH_BYTES;
dispatch.request_id = request_id;
dispatch.sequence_id = sequence_id;
dispatch.sequence_position = sequence_position;
dispatch.active_sequence_count = active_sequence_count;
dispatch.new_token_count = new_token_count;
dispatch.pipeline_slot = pipeline_slot;
dispatch.kv_block_table = kv_block_table;
dispatch.hidden_input_transport_session = input_transport;
dispatch.hidden_output_transport_session = output_transport;
dispatch.hidden_input_packet = input_packet;
dispatch.hidden_output_packet = output_packet;
dispatch.completion_function = completion_function;
dispatch.completion_context = completion_context;

status = SparkResidentDecodeStageProductionRunnerSubmit(
    &runner,
    &dispatch);
```

For prefill:

```c
dispatch.flags |= SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DISPATCH_FLAG_PREFILL;
dispatch.prefill_view = prefill_view;
```

## Production Integration Order

A production process should initialize in this order:

1. Load or validate rank plan for this Spark.
2. Validate required layer `.sp*` pack files for this rank.
3. Initialize resident GLM-5.2 driver and keep rank weights resident in VRAM.
4. Open persistent hidden transport sessions for previous and next ranks.
5. Initialize KV cache, prefix cache, scheduler, request API, serving engine,
   service runtime, and compatibility adapter storage.
6. Accept token-id or JSON requests at the service boundary.
7. Pump service events and dispatches until completion.

No production path should shell out to `scp`, `rsync`, Python, a file handoff,
or a validation fixture.

## Status Handling

Every API returns `SparkStatus`.

Common expected failures:

```text
SPARK_STATUS_INVALID_ARGUMENT      malformed descriptor, bad flags, null pointer
SPARK_STATUS_CAPACITY_EXCEEDED     caller storage or queue capacity is too small
SPARK_STATUS_NOT_FOUND             unknown client or request mapping
SPARK_STATUS_BUSY                  no queue slot, admission rejected, or full ring
SPARK_STATUS_PROTOCOL_ERROR        malformed frame or JSON shape
```

Treat any status other than `SPARK_STATUS_OK` as request failure at the caller
boundary unless the specific caller is implementing retry/backpressure policy.

## Current Boundary Notes

- The compatibility API is a C JSON adapter, not an HTTP server.
- The production runner requires hidden transport sessions, but the actual
  fabric backend must be provided by the deployment.
- The service layer can submit text if the serving engine has a tokenizer; for
  bring-up, token-id submission is the stricter lower-level boundary.
- Validation and benchmark harnesses are not production API endpoints.
