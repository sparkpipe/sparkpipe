#pragma once
// The CUDA wmma header, stubbed for the host harness. The nvcuda::wmma
// declarations live in the harness that includes this header before the
// kernel tree (see tests/host_cuda/spark_lm_batched_host.cu): only enough
// namespace for the tensor-tile templates in the shared kernel header to
// PARSE. Nothing may call them here - the device-only tile paths must never
// run under the one-thread harness, and the no-op stubs fail to link if one
// does.
