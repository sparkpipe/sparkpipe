# Repository Structure

| Path | Responsibility |
| --- | --- |
| `include/sparkpipe/` | Public runtime, model, transport, and package ABIs |
| `src/` | Core status, hashing, filesystem, JSON, loading, and compilation |
| `api/` | HTTP and compatibility request surfaces |
| `scheduler/` | Admission, priorities, deadlines, dynamic batching, and stage planning |
| `cache/` | KV storage, prefix sharing, retention, and tiering policy |
| `inference/kernels/` | Reusable CUDA mechanisms, formats, and launch primitives |
| `inference/llms/` | Checkpoint-specific device execution |
| `model-families/` | Generated model facts and host-side family contracts |
| `model_contracts/` | Exact checkpoint and execution contracts |
| `modules/` | Package-selected resident adapters, drivers, and firmware modules |
| `ring/` | Pipeline transport, TP collectives, RDMA, and TCP reference transport |
| `runtime/` | Resident lifecycle, workspace, launch planning, and tensor maps |
| `node/` | Generic resident process and batch client |
| `deployment/` | Immutable release assembly, installation, and activation |
| `qualification/` | Hardware, transport, numerical, and evaluation gates |
| `tools/` | Generators, packers, release tooling, and qualification runners |
| `tests/` | Host, contract, source, and integration tests |
| `text/` | Tokenizer and prompt-template primitives |

Common runtime code does not choose a model family, codec, topology, batch
width, or fallback implementation by name. A deployment package binds those
decisions through exact model, hardware, and release contracts.
