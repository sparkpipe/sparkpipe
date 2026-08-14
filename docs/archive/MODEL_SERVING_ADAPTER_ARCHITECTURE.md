# Model serving adapter architecture

SparkPipe has one model-neutral resident process and one serving adapter per
model family. The adapter is the only layer allowed to know a model's frame,
stage-pack schema, KV ownership, token layout, or driver program name.

## Ownership

| Layer | Owns | Must not own |
|---|---|---|
| model driver | CUDA modules, resident weights and KV, model frame execution | sockets, request scheduling, transport selection |
| serving adapter | model identity, stage split, typed node context, generic submission translation, model-work quiescence | process lifetime, IPC framing, transport implementation |
| model resident | CUDA stream, transport sessions, adapter lifetime, completion routing, local IPC | model structs, tensor names, pack geometry, alternate backends |
| resident client | handshake, bounded submission queue, result/completion correlation | reconnect policy, model translation, resident construction |
| pipeline client | all-rank fanout, aggregate admission/completion, terminal rank failure | model translation, transport payloads, partial retry |
| batch engine | request slots, prompt chunking, decode batching, cancellation, stop/budget handling, KV-slot release | tokenization, chat templates, CUDA context, model driver construction |
| model application | tokenization, chat template, API policy, request admission | resident construction, transport, model frame translation |

The public model-driver ABI remains the CUDA/firmware boundary. The serving
adapter composes that ABI into a model-independent serving contract instead of
teaching the resident about every model.

## Startup contract

1. One strict deployment manifest is authoritative for the adapter, driver,
   transport, runtime limits, coordinator, rank/stage mapping, immutable local
   runtime roots, data-plane hosts, model configuration, and control endpoints.
   Unknown or duplicate members are schema errors.
2. Every artifact and model configuration path is normalized and relative to
   the selected node's absolute runtime root. Absolute component paths,
   `.`/`..` components, empty components, cwd lookup, `$HOME`, and environment
   expansion are rejected. This permits different Spark home directories while
   keeping one DRY release description.
3. The resident loads exactly one adapter shared object selected by that
   manifest. Its command line is exactly `--deployment PATH --rank-index N`.
4. The adapter descriptor declares model ID, immutable revision, artifact
   digest, stage split, boundary format and geometry, capacity, and
   capabilities. The current production transport accepts BF16 boundaries
   only; another format fails closed instead of being treated as BF16 bytes.
5. The resident derives each linear stage slice directly from the adapter's
   exact stage-layer table and the manifest's explicit per-node transport hosts,
   then loads the selected transport module. It does not reuse a model family's
   balanced-partition or routed-layer cut rules. Hostnames are never derived
   from a prefix or decimal rank.
   Local rank and control-port identity are carried in the typed endpoint; the
   resident does not mutate process environment to configure the route. Rank
   plans are pointer-free value objects; endpoint builders materialize each edge,
   and an opened transport session owns copies of all endpoint text.
6. The resident allocates one stable input and output boundary arena per
   in-flight slot: mapped host memory for host RDMA or device memory for
   GPUDirect RDMA. No transport name maps to a simulation implementation.
7. The adapter receives one typed configuration object and creates the model
   driver through a versioned create request. The resident's execution stream is
   carried through firmware host services at initialization and must be the same
   stream named by every submitted frame. A model module does not create a
   private stream or select one through environment. Model geometry and pack
   paths are not recovered from process-global
   environment variables. The typed configuration includes the manifest's
   immutable runtime root; model configuration files name pack artifacts only
   by normalized root-relative paths. Absolute, empty, `.`/`..`, and trailing
   slash paths fail closed. Production adapter configuration cannot override a
   model's qualification state. Unqualified CUDA bring-up is confined to an
   explicitly named validation executable and cannot be selected by deployment.
8. A client must match adapter ID, model ID, revision, artifact digest, rank,
   and stage during the IPC handshake before work is accepted.
9. Shutdown first closes admission, then calls the adapter's mandatory
   `quiesce` hook to a bounded monotonic deadline. Only a quiescent adapter may
   be followed by resident-stream synchronization, transport close, adapter
   destroy/unload, boundary release, and stream destruction. A failed quiesce
   preserves live resources until process exit rather than risking use-after-
   free.

`runtime_limits.max_active_sequences`, `max_input_rows`, and
`resident_sequence_capacity` are separate, mandatory manifest settings. They
bound one dispatch's live lanes, one prefill packet's hidden-state rows, and the
persistent KV slot table respectively. Input rows and resident capacity must
both be at least as large as the live dispatch width. All resident hot-path
queues, message storage, boundary arenas, and model KV allocations are bounded
by those validated startup limits. A protocol error or disconnect is terminal
for that client connection; there is no automatic reconnect or alternate
execution path.

The pipeline client loads the coordinator's adapter and connects every declared
stage from that same manifest over an explicitly typed Unix or TCP control
endpoint. It does not accept a second endpoint list or capacity table. It
preflights all rank queues before fanout and uses two-phase admission. Every
resident first validates and reserves a bounded route without invoking CUDA or
hidden transport. The route atomically claims every persistent KV slot named by
its live lanes, so overlapping work cannot collide inside a driver after
commit. Only after all stages prepare successfully does the pipeline
commit downstream-to-upstream, so receives are armed before upstream sends. If
any stage rejects, all prepared routes are aborted and a later request can use
the released slots. It requires monotonically increasing global submission IDs,
strictly increasing per-connection message IDs, and nonzero distributed
generation fields. Partial execution is never used as a fallback.

`SparkModelBatchEngine` is the first model-neutral request consumer of that
pipeline. It allocates request token storage, lane maps, row maps, and every
in-flight transaction record once at startup. It packs multiple live requests,
chunks prompts at the manifest's row limit, alternates prefill and decode work,
and keeps independent persistent sequence slots. A model that requires explicit
KV release receives a distributed `RELEASE` transaction before the slot is
returned to the free list. Position-zero-rebind models can recycle the slot only
through a new request generation at position zero. There is no serial execution
mode or alternate backend inside the engine.

The batch engine accepts token IDs. Tokenization and chat-template policy are
application concerns and do not belong in a CUDA device driver. A family can
therefore change its tokenizer or API template without changing the resident,
pipeline protocol, transport, or driver ABI. Conversely, no application may
reach around the adapter and construct a family frame directly.

## Work contract

The scheduler submits model-neutral lanes, token rows, positions, sequence IDs,
and an optional bounded model extension. The adapter validates the submission,
translates it to the model frame, and reports a completion carrying the original
submission identity. Admission and completion are separate messages so a
synchronous driver callback cannot race the submit result.

The adapter's `validate_submission` entry point is a side-effect-free preflight.
It validates model extensions, row semantics, positions, and other model-owned
contracts before route reservation. Resident-owned hidden input/output pointers
are absent during preflight and are bound only after preparation. The adapter's
`submit` entry point may validate those bound pointers defensively, but it is
called only after the pipeline-wide commit.

The first `active_sequence_count` lanes are live; any additional lanes are
explicit padding. Every live lane must own at least one row, every row must name
a live lane, and the row sequence ID must match that lane. This is validated
before a model adapter sees the work.

Each live lane also names a persistent `resident_sequence_slot`. Resident slots
must be unique within a dispatch and remain independent of batch order. An
adapter declares exactly how a slot may be reused: either an explicit `RELEASE`
is required, or a changed sequence at position zero atomically resets and
rebinds the slot. DSV4 uses position-zero rebinding; the adapter translates each
batch-local row index to its lane's persistent slot before entering the driver.
Padding lanes carry the explicit no-slot sentinel.

Boundary pointers are process-local resident resources. They are neither
accepted from a client nor encoded in IPC. After receiving a submission, the
resident binds the route's owned input and output arenas before invoking the
adapter. A model adapter consumes those pointers but never allocates, registers,
selects, sends, or receives a transport buffer.

The first stage owns token embedding input. Intermediate stages own only hidden
transport input/output. The final stage owns emitted token IDs. Batched prefill
must return one output per live lane, selected from that lane's final row. The
pipeline coordinator stores the admitted live-lane count and accepts a successful
non-release completion only when the final stage returns exactly that many token
IDs; token output from any other stage, or from `RELEASE`, is a protocol error.

Every rank in a pipeline receives the same globally unique `submission_id` and
`dispatch_generation`, plus the same control, transaction, request, and step
generations. The opaque driver residency token also survives submission and
completion byte-for-byte. Those values survive IPC correlation;
the submission and dispatch generations are also the transport packet identity.
The resident does not advance from input wait or output wait, emit completion,
or recycle the slot until one exactly matching transport completion is polled.
Unknown, duplicate, stale, or mismatched completions fail the resident closed.
Transport operations have distinct ready and waiting states, so an accepted
asynchronous receive or send is posted exactly once. `RELEASE` is a control-plane
operation and never creates a zero-row hidden-state transfer.

## No fallback rule

Production transport modes are exactly `host-rdma` and `gpudirect-rdma`.
Unsupported adapter capabilities, model identities, revisions, artifacts,
transport modules, work kinds, or malformed IPC frames fail closed. Diagnostic
drivers and transports exist only as explicitly named test fixtures and are not
selectable through a production alias.

## Model adapter checklist

A new model adds a model-owned adapter shared object that provides:

- an immutable descriptor and exact stage-layer table;
- a structured adapter configuration schema;
- a typed driver node context;
- translation for every advertised work kind;
- exact per-submission completion ownership and final-stage token cardinality;
- a quiesce hook that closes admission and proves all model work complete;
- batched prefill and decode tests with multiple lanes;
- a process-level resident/client test using a fake driver and transport;
- live CUDA qualification from merged main before production status changes.

Capabilities that are not implemented are omitted. Prefetch, release, reset,
JIT KV, and speculation are independent capabilities, not compatibility stubs.

## GLM application boundary

The existing GLM service backend, rank daemon, resident daemon, work-control
packet, and gateway assembly predate this contract and remain the GLM
application path. Their configuration and dynamic-loader type remain explicitly
GLM-owned; they are not a template or compatibility layer for new model support.
New model families use the serving adapter, model resident, pipeline client, and
batch engine defined here. An HTTP application may be attached above the batch
engine, but cannot substitute a GLM driver or GLM work packet underneath it.
