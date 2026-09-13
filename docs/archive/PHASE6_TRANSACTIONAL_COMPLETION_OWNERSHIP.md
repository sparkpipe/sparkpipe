# Phase 6 — Transactional Completion Ownership

## Scope

Phase 6 carries the Phase 5 work-transaction foundation through the rank daemon,
resident-completion path, service backend, and final-event path. The objective is
to make request identity survive queueing, reconnect, asynchronous driver
completion, and completion backpressure without executing a packet twice or
silently losing its owner.

This phase is host-validated. It does not claim a system-wide all-rank commit,
CUDA compilation, Blackwell execution, or network qualification.

## Transaction identity

Every distributed work packet has one canonical identity:

```text
control generation
transaction id
dispatch generation
request generation
step generation
step chunk index/count
transaction phase
```

The packet fingerprint is retained with that identity. Reusing an identity for
different packet bytes is a validation failure. Replaying the same packet is
classified separately as an active or terminal replay.

The transaction ledger enforces:

```text
PREPARED -> ACCEPTED -> EXECUTING -> COMMITTED
                         |            |
                         +-> FAILED <-+
                         +-> CANCELLED
```

Terminal records are retained as bounded replay tombstones. A control-generation
change is accepted only when no transaction remains active.

## Acknowledged forwarding and reconnect replay

A sender no longer removes a work packet after the socket accepts its bytes. It
retains the exact packet until an acknowledgement matches all of:

```text
transaction identity
packet fingerprint
transaction ABI
acknowledgement descriptor size
```

After reconnect, the sender retransmits the retained bytes. The receiver's
transaction ledger recognizes the exact replay and does not execute the step a
second time. A different packet carrying the same identity is rejected.

The current forwarding window is deliberately stop-and-wait. This is a
correctness-first bring-up mode, not the final throughput protocol.

## Separate credit domains

The old overloaded credit meaning is split into four independent domains:

```text
transport-window credit
resident-reservation credit
execution credit
completion-ownership credit
```

A release without a matching acquisition is a validation failure. The backend
currently uses resident-reservation credit for CUDA-resident submission. The
other domains are defined and tested but still need complete integration into
the all-rank protocol.

## Asynchronous rank completion ownership

The rank daemon allocates one in-flight transaction record and one completion
mapping for each submitted lane before calling the model driver. The mapping
retains:

```text
transaction owner
request id and request generation
sequence id and sequence position
expected completion count
terminal failure state
```

Driver callbacks do not mutate transaction state or final-event queues. They
copy a bounded completion record under a mutex, signal the daemon wake pipe, and
return. The event-loop thread performs matching, state transitions, final-event
construction, and queue ownership.

This ordering handles a model driver that invokes its completion callback
synchronously from `submit`: the callback is queued, but it is not consumed
until the transaction has reached `EXECUTING`.

A multi-lane transaction commits only after every expected lane completion has
been consumed. The first failing lane marks the transaction terminal-failed;
remaining sibling completions are still drained before the in-flight owner is
released.

## Lossless final-event behavior

Each final event carries the complete transaction envelope and the exact lane
request generation. Final-event saturation returns backpressure before the
completion mapping or transaction count is changed. The completion remains at
the head of the bounded driver-completion queue until the final-event queue can
accept it.

A malformed completion that matches a live owner deterministically fails that
transaction. An unmatched or stale completion is rejected and cannot steal a
valid owner's mapping.

Completion-queue overflow is fail-stop rather than lossy: the daemon records the
overflow, wakes the event loop, and stops instead of dropping completion
identity.

## Backend ownership

The service backend now owns an exact heap copy of every queued packet and keeps
that copy until a matching acknowledgement arrives. It also:

- stamps canonical transaction identities on decode, prefill, verify, and
  release work;
- coalesces exact duplicate release records;
- rejects an identity reused for different bytes;
- retries capacity-rejected work without losing queue ownership;
- uses state-owned prefill storage rather than function-static storage;
- backpressures early-final saturation instead of evicting an older event;
- fails a live pending request when its matching final event is malformed;
- configures outgoing work sockets with `TCP_NODELAY` and keepalive.

## Callback-thread safety

The driver-completion queue is mutex protected. Wake-pipe signal and drop
statistics are updated and read with relaxed atomic operations because the write
side can run on model-driver callback threads while the status side runs on the
event-loop thread.

## Validation boundary

Phase 6 proves host-side ownership and replay invariants. It does not yet prove:

- reservation by every pipeline rank before execution;
- one global commit decision after every rank completes;
- rollback of already executed device state after a downstream failure;
- replay safety across process restart without a persisted restart epoch;
- CUDA callback ordering on Blackwell;
- socket/RDMA failure behavior on the physical ring;
- throughput of the eventual multi-packet acknowledgement window.
