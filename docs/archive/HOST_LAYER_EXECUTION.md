# Running a whole layer on a CPU: what blocks it, and what unblocks it

## Why this matters

Every per-kernel host harness passes, because every kernel is individually
correct. An external audit then found six P0s in K3, and its closing observation
was the useful one: they live in the paths those harnesses do not execute.

Three were dataflow —

- the routed-expert quantise read row `r` for packed row `r`, with no route map
- the shared expert wrote the buffer the routed branch had just written, and
  `LmGemmStore` assigns
- a norm strided by its slice width instead of its row width

— and no per-kernel test can see any of that, because each kernel was doing
exactly what it was told with the arguments it was given. `tests/test_layer_dataflow.py`
catches the first two by reading the source. The third needs execution.

## What blocks it

`g++` cannot parse `<<<grid, block, shared, stream>>>`. It is not a macro and
not a template; it is CUDA syntax that only `nvcc` accepts. Across the tree:

    layer.cuh files          114 launches
    project.cuh                2

So the shim approach that works for kernels — include the real header, give it
a grid, call it directly — cannot reach a layer, because the layer's body is
made of launches.

`nvcc` parses them and then requires a device at run time, so compiling the
harness with `nvcc` does not help either.

## What unblocks it

A launch macro. Every launch becomes

    LM_LAUNCH(KernelName<T...>, grid, block, shared, stream, args...)

which expands to `<<< >>>` under `__CUDACC__` and, in the host harness, to the
loop `tests/host_cuda/lm_host_cuda.cuh` already implements — set `blockIdx`,
call the function, repeat.

This is a mechanical change to 116 call sites and no change to any kernel. It is
worth doing for a second reason beyond testability: `tests/test_kernel_launches.py`
currently reads launches with a regex, which is why it can be defeated by
rewriting a call in a form the parser does not recognise. A macro makes the grid
and the argument list ordinary C++ that a compiler checks.

## What is already here

`tests/host_cuda/shim/runtime/gemm.cuh` records a GEMM rather than performing
one: the call, its shapes, and a value derived from the call index so a later
write landing on an earlier buffer is visible. It builds. `launch.h` copies the
real error codes rather than choosing its own, so a host failure and a device
failure are the same number.

The stubs that let the kernel tree parse for a host — `cuda_runtime.h`,
`cuda_pipeline.h`, the address-space casts — are in `tests/host_cuda/`. They
return zero rather than something plausible, because any of them being called on
the host is a bug in the harness and not something to emulate.

What is missing is only the macro, and the reason it is not in this commit is
that changing 116 call sites is a change that should be reviewed on its own
rather than buried in a harness.
