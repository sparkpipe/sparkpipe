# Device programs

A program is a step recorded once on a device queue and run again with one
call per step, through the device layer
([`include/sparkpipe/spark_device.h`](../include/sparkpipe/spark_device.h)).
Values that change from step to step, such as a row count or a ring slot, are
patched into places marked while recording. Every backend takes the same
calls, so an engine records and runs its steps the same way on each.

## Entry points

```c
SparkStatus SparkDeviceProgramRecord(SparkDeviceQueue *queue, const char *name);
SparkStatus SparkDeviceProgramPatch(SparkDeviceQueue *queue, const SparkDevicePatch *patch);
SparkStatus SparkDeviceProgramFinish(SparkDeviceQueue *queue, SparkDeviceProgram **program);
SparkStatus SparkDeviceProgramRun(SparkDeviceProgram *program, SparkDeviceQueue *queue, const uint64_t *values, uint32_t count);
SparkStatus SparkDeviceProgramGetInfo(const SparkDeviceProgram *program, SparkDeviceProgramInfo *info);
SparkStatus SparkDeviceProgramDestroy(SparkDeviceProgram *program);
uint64_t SparkDeviceTransformApply(const SparkDeviceTransform *transform, uint64_t value, uint32_t *fits);
```

**Recording.** `SparkDeviceProgramRecord` puts a queue in recording mode
until `SparkDeviceProgramFinish`. Each operation submitted to the queue in
that time is checked as it would be when run, then appended to the program
instead of running. The operations are kernel launches, copies, fills, host
callbacks, event records and event waits.
- A queue records one program at a time. A second `SparkDeviceProgramRecord`
  on it returns `SPARK_STATUS_BUSY`.
- While a queue records, the recording thread is refused with
  `SPARK_STATUS_BUSY` when it waits on the queue or queries it, calls
  `SparkDeviceSynchronize`, starts a capture on it, destroys it or runs a
  program on it.
- Another thread's wait on the recording queue, and its
  `SparkDeviceSynchronize`, wait until the recording ends, within their
  timeout, and its query answers `SPARK_STATUS_PENDING`. That is how an
  adopted queue's owner keeps other threads off a stream it is capturing.
- Operations on other queues run as usual and are not part of the program.
- An operation refused while recording breaks the program. Finish then
  refuses it with `SPARK_STATUS_INVALID_ARGUMENT`, so an incomplete program
  never runs. Finish always ends the recording.

**Patches.** `SparkDeviceProgramPatch` marks one place in the kernel launch
recorded just before it, and is refused after any other operation.
- `value` picks one of the run's values, 0 to 63
  (`SPARK_DEVICE_PROGRAM_MAX_PATCHES`). One value can feed several places.
  A program takes one more value than the highest index its patches use.
- `target` is the place:
  - `SPARK_DEVICE_PROGRAM_TARGET_ARGUMENT`: kernel parameter `argument`, at
    the offset and size the kernel's parameter table gives;
  - `SPARK_DEVICE_PROGRAM_TARGET_CONSTANT`: `bytes` bytes at `offset` in the
    launch's constants;
  - `SPARK_DEVICE_PROGRAM_TARGET_GRID_X`, `_Y` or `_Z`: the launch's group
    count on that axis.
- A place is 1, 2, 4 or 8 bytes inside the launch's constants. A group count
  is 4 bytes.
- `transform` turns the run's value into the place's value:
  `ceil((value * multiply + add) / divide) * scale + offset`, computed
  exactly in 128 bits. `divide` 0 leaves the value unchanged. A group count
  from a row count is `multiply` 1, `divide` the rows per group and `scale` 1.
  `SparkDeviceTransformApply` computes it the same way and clears `fits` when
  the result is negative or over 64 bits.

**Runs.** `SparkDeviceProgramRun` runs a finished program on any queue,
with exactly as many values as the program takes.
- Every place gets its transformed value. A value that does not fit its
  place refuses the whole run before anything is queued: a negative or
  over-wide result, or a group count of 0.
- Runs on one queue are ordered like any other operations on it.
- Event records and waits in a program act at their position in each run.
  After a run, a recorded event is in flight on the queue the program ran
  on, so other queues and the host wait on it as on any recorded event.

**Lifetimes.** A program keeps everything its recorded device-layer
operations use.
- From the moment an operation is recorded, freeing its buffers, destroying
  its kernel or destroying its event returns `SPARK_STATUS_BUSY` until the
  program is destroyed, or until a failed finish discards it.
- Memory that a launch outside the device layer reaches by address (CUDA's
  own launches, captured by the CUDA flavour) is the caller's to keep alive
  while the program lives.
- `SparkDeviceProgramDestroy` returns `SPARK_STATUS_BUSY` while a queued run
  has not completed.
- `SparkDeviceShutdown` refuses while any program is live.
- A host callback's context belongs to the caller and must outlive the
  program.

**Information.** `SparkDeviceProgramGetInfo` gives:
- `nodes`: the operations in the program. For the CUDA flavour it counts
  every node of the captured graph, including launches made outside the
  device layer.
- `kernels`: the kernel launches among them.
- `values` and `places`: the values a run takes and the places they fill.
- `digest`: a 64-bit FNV-1a digest of the operations and places. Two
  recordings of the same operations on the same buffers and kernels give
  the same digest.

## Backends

`SPARK_DEVICE_PROPERTY_PROGRAMS` is set by every backend that runs programs, and a caller selects programs by it:
where it is clear, the caller launches eagerly.

| Backend | Recording | Run |
| --- | --- | --- |
| CUDA | stream capture on the queue's stream, local to the recording thread, so launches made with CUDA's own calls are captured as well. A patch resolves to the last kernel node captured on the stream (`cudaStreamGetCaptureInfo`), and its parameter offset and size come from `cudaFuncGetParamInfo`. Event records and waits become external event nodes. Finish ends the capture, instantiates the graph and uploads it | values that changed since the last run are written to the kernel nodes they touch (`cudaGraphExecKernelNodeSetParams`), then one `cudaGraphLaunch` |
| host | the recorded operations | each operation replayed in order through the public calls, with patched constants in a scratch copy |
| Metal | refused with `SPARK_STATUS_UNSUPPORTED`: the replay has not run on Metal hardware, so Metal launches eagerly | not reached |
| nvrm | the recorded operations. Finish compiles them into an image: a persistent descriptor and constant bank 0 for every launch, and a prebuilt push segment that runs the steps in order ([nvrm programs](NVRM_BACKEND.md#programs)) | one submission: the queue's ordering acquire, the segment, and a tail that releases the queue's fence. Values, event payloads and host tickets are stored into the image first |

nvrm refuses copies and fills inside a program at finish, naming them: they go on the queue before or after the run.

CUDA refuses what a graph cannot hold, and the refusal breaks the recording:
- a patch on a kernel node that passes its arguments as one packed buffer
  instead of a parameter array, or that takes more than 255 parameters;
- a constant patch that does not cover exactly one parameter;
- a copy of pageable host memory to or from the device. The CUDA backend
  stages those through a buffer each queue owns, which a program run on
  another queue would share. Copies of pinned or registered memory are
  recorded.

## Tests

[`tests/test_device_contract.c`](../tests/test_device_contract.c) runs in
`make test` against the host backend. Its nvrm build runs in
`make test-nvrm-gpu` and its Metal build on a Mac with Metal, where it checks that recording is refused. It covers:

- refusals: a patch or a finish without a recording, a second recording on
  one queue, waits and synchronization while recording, a patch after a fill,
  and a recording with a refused operation refused at finish;
- other threads' waits: a queue wait and a device synchronize started on
  other threads during a recording return only after it finishes, and a
  query answers pending;
- a program of a fill, a copy, a host callback, an event record and a wait
  on it, run three times and compared byte for byte. Recording runs nothing.
  On nvrm, finish refuses that program, naming copies and fills, and a
  program of the host callback, the event record and the wait runs three
  times instead;
- buffers and events kept while a recording or a program uses them;
- transforms: no transform, a ceiling division, an affine form, all five
  terms together, and a negative intermediate that does not fit;
- a kernel program with constant and group-count patches, run with three
  value sets and compared byte for byte with the same launches made eagerly,
  and the kernel kept while the program lives.

[`tests/cuda/device_program_cuda.cu`](../tests/cuda/device_program_cuda.cu),
run by
[`tests/test_device_program_cuda.py`](../tests/test_device_program_cuda.py)
on a CUDA host, records two kernels launched with CUDA's own launch syntax
and a device-layer copy:

- six places for two values: both kernels' group counts and counts from a
  row count, and both seeds, one named by parameter index and one by its
  place in the parameters;
- refused patches break their recordings: a parameter the kernel does not
  have, a constant patch that is not one whole parameter, and a patch after
  a copy;
- five runs, alternating between two streams, each compared byte for byte
  with the same launches made eagerly;
- destroying a program with a run queued behind a held stream returns
  `SPARK_STATUS_BUSY`, and succeeds once the run is done;
- a program that waits for another stream's slow write and records its own
  event, which the other stream waits for before copying the program's
  output: four rounds, each compared with values computed on the host.
