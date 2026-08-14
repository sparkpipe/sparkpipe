# DSV4 nonblocking graph semaphore qualification

This receipt qualifies the scheduling primitive used to keep a complete TP
decode graph armed across host-RDMA collectives. It does not claim inference
correctness or throughput.

The probe links the real DSV4 resident-decode archive, captures a graph that
writes a mapped producer word, waits on a mapped completion word, then runs a
kernel. A highest-priority nonblocking operation stream writes completion. The
graph completes and produces the expected value without a host callback or an
SM-spinning wait kernel.

The recorded GB10 run used CUDA 13.0, target `sm_121a`, and driver 580.159.03.
Its raw facts are in [`result.json`](result.json). The reproducible probe is
[`tools/hardware/spark_dsv4_graph_semaphore_characterize.cu`](../../../tools/hardware/spark_dsv4_graph_semaphore_characterize.cu).
