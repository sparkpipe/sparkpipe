import argparse
import hashlib
import json
import pathlib
import subprocess


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Installed CUPTI sample differs at required anchor: {old[:80]!r}")
    return text.replace(old, new, 1)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Build an isolated adaptation of the installed NVIDIA CUPTI injection sample; never run CUDA.")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--cuda-root", type=pathlib.Path)
    parser.add_argument("--output-directory", type=pathlib.Path)
    args = parser.parse_args()
    if not args.compile:
        parser.print_help()
        return 0
    if args.cuda_root is None or args.output_directory is None:
        parser.error("--compile requires --cuda-root and --output-directory")
    cuda = args.cuda_root.resolve(strict=True)
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    samples = cuda / "extras/CUPTI/samples"
    original_source = samples / "cupti_trace_injection/cupti_trace_injection.cpp"
    original_helper = samples / "common/helper_cupti_activity.h"
    source = original_source.read_text()
    helper = original_helper.read_text()
    for kind in ("MEMCPY2", "MEMORY2", "NAME", "MARKER", "MARKER_DATA"):
        source = replace_once(source, f"    SELECT_ACTIVITY(injectionGlobals.profileMode, CUPTI_ACTIVITY_KIND_{kind});\n", "")
    source = replace_once(source, "#include <mutex>\n", "#include <mutex>\n#include <time.h>\n")
    source = replace_once(source, "    InitCuptiTrace(pUserData, (void *)InjectionCallbackHandler, stdout);", """    const char *tracePath = getenv("SPARK_CUPTI_TRACE_FILE");
    if (tracePath == NULL || tracePath[0] != '/')
    {
        fprintf(stderr,"SPARK_CUPTI_TRACE_FILE must be an absolute output path\\n");
        exit(EXIT_FAILURE);
    }
    FILE *trace = fopen(tracePath,"wx");
    if (trace == NULL)
    {
        perror("CUPTI trace open");
        exit(EXIT_FAILURE);
    }
    setvbuf(trace,NULL,_IOLBF,0);
    InitCuptiTrace(pUserData, (void *)InjectionCallbackHandler, trace);
    CUPTI_API_CALL(cuptiActivityFlushPeriod(1000u));
    uint64_t timestamp = 0;
    struct timespec monotonicTime,realtime;
    if (clock_gettime(CLOCK_MONOTONIC,&monotonicTime) != 0 ||
        clock_gettime(CLOCK_REALTIME,&realtime) != 0)
    {
        perror("CUPTI clock anchor");
        exit(EXIT_FAILURE);
    }
    CUPTI_API_CALL(cuptiGetTimestamp(&timestamp));
    fprintf(trace,"TRACE_ANCHOR pid=%ld cupti_ns=%llu monotonic_ns=%llu realtime_ns=%llu flush_period_ms=1000\\n",
        (long)getpid(),(unsigned long long)timestamp,
        (unsigned long long)monotonicTime.tv_sec*1000000000ull+monotonicTime.tv_nsec,
        (unsigned long long)realtime.tv_sec*1000000000ull+realtime.tv_nsec);""")
    source = replace_once(source, "        CUPTI_API_CALL_VERBOSE(cuptiActivityFlushAll(1));", """        CUPTI_API_CALL_VERBOSE(cuptiActivityFlushAll(1));
        FILE *trace = globals.pOutputFile;
        flockfile(trace);
        fprintf(trace,"TRACE_EXIT buffers_requested=%llu buffers_completed=%llu\\n",
            (unsigned long long)globals.buffersRequested,
            (unsigned long long)globals.buffersCompleted);
        fflush(trace);
        funlockfile(trace);""")
    helper = replace_once(helper, "        PrintActivityBuffer(pBuffer, validSize, pOutputFile, globals.pUserData);", """        flockfile(pOutputFile);
        PrintActivityBuffer(pBuffer, validSize, pOutputFile, globals.pUserData);
        funlockfile(pOutputFile);""")
    helper = replace_once(helper, "    globals.buffersCompleted++;\n    free(pBuffer);", """    size_t dropped = 0;
    CUPTI_API_CALL(cuptiActivityGetNumDroppedRecords(context,streamId,&dropped));
    FILE *trace = globals.pOutputFile;
    flockfile(trace);
    fprintf(trace,"TRACE_BUFFER context=%p stream=%u valid_bytes=%zu capacity=%zu dropped_records=%zu\\n",
        (void *)context,streamId,validSize,size,dropped);
    fflush(trace);
    funlockfile(trace);
    globals.buffersCompleted++;
    free(pBuffer);""")
    source_path = output / "cupti_trace_injection.cpp"
    helper_path = output / "helper_cupti_activity.h"
    library = output / "libspark_cupti_trace.so"
    source_path.write_text(source)
    helper_path.write_text(helper)
    command = ["c++", "-std=c++17", "-O2", "-fPIC", "-shared", "-pthread",
               f"-I{cuda / 'include'}", f"-I{output}", f"-I{samples / 'common'}",
               str(source_path), f"-L{cuda / 'lib64'}", f"-Wl,-rpath,{cuda / 'lib64'}",
               "-lcuda", "-lcupti", "-o", str(library)]
    with (output / "build.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    receipt = {"compiled_only": True, "command": command,
               "sources": {str(path): digest(path) for path in
                           (original_source, original_helper, source_path, helper_path)},
               "library": str(library), "library_sha256": digest(library),
               "trace_environment": {"CUDA_INJECTION64_PATH": str(library),
                                     "SPARK_CUPTI_TRACE_FILE": "/absolute/unique/rank0.cupti.log"},
               "activities": ["DRIVER", "RUNTIME", "OVERHEAD", "CONCURRENT_KERNEL", "MEMSET", "MEMCPY"],
               "flush_period_ms": 1000,
               "final_flush": "official sample atexit, profiler-stop and device-reset hooks"}
    (output / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
