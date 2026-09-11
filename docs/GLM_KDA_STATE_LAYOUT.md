# GLM rank-local recurrent state

The KDA kernels compute `64 / tp_degree` heads. Their shared delta-rule kernel
already accepts a slot byte stride; convolution kernels address rank-local
channels. The GLM driver nevertheless allocated and reset all 64 heads per
resident slot. Its window constant includes Q, K and V, but allocation multiplied
that constant by three again.

Allocation, reset, layer binding and replay now use the rank-local state stride.
Each window pool gets one rank-local window. No kernel algorithm or scheduler
was added. The layer shape check requires the actual head count's state bytes.

For 34 KDA layers, per resident sequence (excluding KV/index and workspaces):

| Topology | Before | After |
| --- | ---: | ---: |
| TP1 | 155.125 MiB | 142.375 MiB |
| TP4 | 155.125 MiB | 35.59375 MiB |
| TP16 | 155.125 MiB | 8.8984375 MiB |

These are allocation calculations, not measured memory bandwidth or tok/s.
State is 64 heads × 128 × 128 × four bytes, divided by TP. Combined windows
are 3 × 64 heads × 128 × four history entries × two bytes, divided by TP.
This also bounds the meaningful bytes a full recurrent checkpoint must retain.

Validation: `python3 tests/test_glm5_next_stage_context.py` executes the real
allocator with host CUDA allocation stubs at TP1/4/16, three slots and three
KDA layers. It checks the allocation ledger, layer strides and disjoint window
pool offsets. `test_glm5_next_module_host_syntax.py` passes. GPU compilation,
merged-main B3 memory checking, cross-version token comparison and numerical
validation are still required. Full prefix restoration is a separate unfinished
integration in PR #861.

Reusable lesson: inspect the shared kernel's existing stride parameter before
adding a new abstraction. Allocate the state that the rank actually owns and
keep allocation, initialization, execution and replay on that same layout.
