#pragma once
// The kernel tree includes cuda_runtime.h for the stream and error types; the
// shim already declares them. This exists so the include resolves without
// pulling the toolkit in, and defines nothing of its own - anything the tree
// actually needs from the runtime should fail loudly here rather than be
// invented.
#include "lm_host_cuda.cuh"
