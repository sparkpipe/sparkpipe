# SparkPipe platform PERT and delivery graph

The graph is dependency-driven. Durations are engineering estimates, not
deadlines or performance claims. Parallel work begins only when its input
contract is frozen and its write set is independent.

```mermaid
flowchart LR
    A0["A0 Architecture baseline\n0.5 d"]
    A1["A1 Main/unified reconciliation manifest\n0.5 d"]
    A2["A2 Qualification lattice and receipts\n0.5 d"]
    O0["O0 Retrying pair controller\n0.5 d"]
    O1["O1 Live dashboard and restart recovery\n0.5 d"]

    P0["P0 Daily comparable-SOTA ledger\ncontinuous"]
    P1["P1 Benchmark and +10% target protocol\n1 d"]
    P2["P2 Profile-to-work-order compiler\n1 d"]
    P3["P3 DSV4 Flash profile and speed loop\ncontinuous"]
    P4["P4 Qwen 27B profile and speed loop\ncontinuous"]
    P5["P5 GLM 5.2 correctness then speed\ncontinuous"]
    P6["P6 K3 correctness then speed\ncontinuous"]

    R0["R0 Recipe schema v1\n1 d"]
    R1["R1 Canonical hash and mnemonic catalog\n1 d"]
    R2["R2 Recipe compiler and capacity envelope\n3 d"]
    R3["R3 Artifact DAG and Ceph factory\n5 d"]

    H0["H0 Hardware capability schema\n1 d"]
    H1["H1 Hardware interface port\n2 d"]
    C0["C0 CUDA baseline adapter\n3 d"]
    A3["A3 ROCm/HIP substrate\n4 d"]
    A4["A4 DSV4 gfx950 one-layer gate\n5 d"]
    M0["M0 Metal capability probe\n2 d"]
    M1["M1 Metal execution substrate\n5 d"]
    M2["M2 Metal first island gate\n4 d"]

    T0["T0 Topology/capability graph\n2 d"]
    S0["S0 Compute-island offers and lease model\n2 d"]
    S1["S1 Fleet placement solver\n4 d"]
    S2["S2 Island active-set reconciler\n4 d"]
    S3["S3 Island-local batch/deadline scheduler\n5 d"]
    B0["B0 Global request broker\n4 d"]

    K0["K0 KV ownership and precision contract\n2 d"]
    K1["K1 Common KV convergence\n4 d"]
    K2["K2 Park/restore module operations\n4 d"]
    K3["K3 Pager, prefetch and anti-thrash\n5 d"]
    K4["K4 B1024 x 256K fleet qualification\n5 d"]

    G0["G0 Public API compatibility contract\n2 d"]
    G1["G1 Hardened gateway and streaming\n5 d"]
    G2["G2 Auth, quotas and cancellation\n4 d"]
    L0["L0 Immutable usage ledger\n4 d"]
    P0["P0 Provider enrollment and signed leases\n5 d"]
    P1["P1 Billing, reserve and payout sandbox\n6 d"]

    Q0["Q0 Qwen 27B recipe-to-build CUDA\n4 d"]
    D0["D0 DSV4 Flash production CUDA\n5 d"]
    D1["D1 DSV4 ROCm full model\n8 d"]
    X0["X0 GLM 5.2 correctness\n8 d"]
    X1["X1 K3 correctness\n8 d"]
    X2["X2 Big-model review-to-runtime program\n12 d"]

    Z0["Z0 Local authenticated beta\n5 d"]
    Z1["Z1 External-provider beta\n10 d"]
    Z2["Z2 Production marketplace qualification\n20 d+"]

    A0 --> A1
    A0 --> A2
    A0 --> O0 --> O1
    A0 --> P0 --> P1 --> P2
    P0 --> P3
    P0 --> P4
    P2 --> P3
    P2 --> P4
    P2 --> P5
    P2 --> P6
    A0 --> R0 --> R1 --> R2 --> R3
    A1 --> H1
    A2 --> R2
    A2 --> H1

    A0 --> H0 --> H1
    H1 --> C0
    H1 --> A3 --> A4
    H0 --> M0 --> M1 --> M2

    H0 --> T0
    R0 --> T0
    T0 --> S0 --> S1 --> S2 --> S3
    R2 --> S1

    R0 --> K0 --> K1 --> K2 --> K3 --> K4
    H1 --> K1
    S3 --> K3

    A0 --> G0 --> G1 --> G2
    S0 --> B0
    S1 --> B0
    G0 --> B0
    B0 --> G1
    S3 --> G1
    G2 --> L0
    S0 --> P0
    A2 --> P0
    L0 --> P1
    P0 --> P1

    C0 --> Q0
    R2 --> Q0
    C0 --> D0
    K3 --> D0
    A4 --> D1
    R2 --> D1
    C0 --> X0
    C0 --> X1
    R2 --> X0
    R2 --> X1
    X0 --> X2
    X1 --> X2

    P3 --> D0
    Q0 --> P4
    X0 --> P5
    X1 --> P6

    Q0 --> Z0
    D0 --> Z0
    G2 --> Z0
    K3 --> Z0
    S2 --> Z0
    B0 --> Z0
    Z0 --> Z1
    P0 --> Z1
    L0 --> Z1
    D1 --> Z1
    Z1 --> Z2
    P1 --> Z2
    K4 --> Z2
    X2 --> Z2
```

## Critical paths

### Local product path

`A0 -> R0 -> R1 -> R2 -> Q0 -> Z0`

This yields one recipe-selected, authenticated local service while the broader
model/backend matrix continues in parallel.

### Performance lead path

`A0 -> P0 -> P1 -> P2 -> per-model profile -> audited optimization -> live receipt`

This path runs continuously. A SOTA observation expires when its daily source
scan or comparability fields are stale. Working models enter the profile and
optimization loop immediately; non-working models remain on correctness gates
before performance work can claim progress.

### Hardware-neutral path

`A0 -> H0 -> H1 -> A3 -> A4 -> D1 -> Z1`

This is blocked by actual MI350P access at hardware gates, not by host-only
scaffolding.

### KV scale path

`A0 -> R0 -> K0 -> K1 -> K2 -> K3 -> K4 -> Z2`

B1024 and 256K are an offered-load and parked-session goal. A single device is
not expected to hold every lane active simultaneously.

### Marketplace path

`A0 -> H0 -> T0 -> S0 -> P0 -> Z1 -> Z2`, with request routing through
`G0 -> B0 -> G1` and the independent financial path
`G0 -> G1 -> G2 -> L0 -> P1 -> Z2`.

## First 24-hour cut

The first day targets contract and controller milestones, not the final graph:

| Lane | Deliverable | Exit evidence |
| --- | --- | --- |
| Architecture | A0, A1, A2 drafts | Reviewed docs, machine-readable reconciliation and task graph |
| Agent control | O0, O1 | One implementation/audit pair survives injected 429, empty response, process restart, and audit rejection |
| Performance | P0, P1, first P3 refresh | Daily primary-source ledger, field-level comparability decisions, 110% targets, and a no-drop DSV4 work-order queue |
| Recipes | R0 and R1 skeletons | Schema positives/negatives, canonical hash vectors, mnemonic collision tests |
| Hardware | H0 and H1 reviewable slices | Host compile, no vendor types in portable headers, preserved CUDA contract tests |
| Topology/scheduler | T0, S0, and B0 contracts | Island offer filtering, locality groups, lease expiry, no-double-sell, and global-to-local admission tests |
| KV | K0 contract and K2 ABI design | Capacity arithmetic, 1/N ownership assertions, no-mid-decode-paging tests |
| API | G0 contract tests and G1 skeleton | Standard payload/error/SSE fixtures; existing `model_api.c` remains private |
| Backends | A3 and M0 compile/probe scaffolds | Honest host/simulator receipts only; no device-readiness claim |

## Pairing rules for the DAG

- Every box in the graph expands to one or more task contracts.
- Each task has an implementer and an independent auditor.
- The auditor waits for an immutable patch hash, then starts in a fresh clone.
- A rejected patch returns to the same task with a new attempt; dependents stay
  blocked.
- Overlapping write sets create an additional scheduler lock even when the PERT
  graph has no semantic edge.
- Hardware tasks may complete host-only subtasks while their target gate stays
  blocked. Their state must say which gate is missing.

## Realistic program size

The complete graph is a multi-week engineering program even with substantial
parallelism because target-hardware, integration, correctness, load, security,
and zero-drift release gates are serial at key boundaries. Dozens of agents can
compress independent implementation and audit work; they cannot make a failed
numerical or hardware gate true by consensus.
