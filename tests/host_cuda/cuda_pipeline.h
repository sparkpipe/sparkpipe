#pragma once
// TMA and the async-copy pipeline have no host meaning. The layer does not
// reach them - the recorder replaces the GEMM that would - so these are
// declarations that let the header parse, not emulations.
static inline void __pipeline_commit(void) {}
static inline void __pipeline_wait_prior(int) {}
