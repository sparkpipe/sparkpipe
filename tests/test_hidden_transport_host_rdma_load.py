"""Build the host RDMA hidden transport on an sm_121 device and load it through the interface check residentd applies to a host-rdma deployment."""
import pathlib
import shutil
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
REQUIRED_COMPUTE_CAPABILITY = "12.1"
PROBE = r'''
#include <stdio.h>
#include "sparkpipe/spark_hidden_transport.h"
int main(int argc, char **argv)
{
    SparkHiddenTransportDynamicLibrary library;
    SparkStatus status = SparkHiddenTransportLoadInterfaceFromSharedObject(argv[1], SPARK_HIDDEN_TRANSPORT_REQUIRED_SPARK_HOST_RDMA_CAPS, &library);
    if (argc != 2 || status != SPARK_STATUS_OK)
    {
        printf("FAIL host RDMA transport %s refused by the host-rdma contract: status=%d\n", argc == 2 ? argv[1] : "?", (int)status);
        return 1;
    }
    if (library.transport_interface.cancel == 0)
    {
        printf("FAIL host RDMA transport exports no cancel\n");
        return 1;
    }
    SparkHiddenTransportUnloadInterface(&library);
    printf("PASS host RDMA transport loads under the host-rdma contract with every required operation\n");
    return 0;
}
'''


def device_capabilities():
    if shutil.which("nvidia-smi") is None:
        return []
    result = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True, text=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()] if result.returncode == 0 else []


capabilities = device_capabilities()
if REQUIRED_COMPUTE_CAPABILITY not in capabilities:
    print(f"SKIP test_hidden_transport_host_rdma_load (needs an sm_121 device; this host reports {capabilities or 'none'})")
    raise SystemExit(0)
subprocess.run(["make", "-j8", "hidden_transport_spark_host_rdma_verbs", "build/libsparkpipe_model_common.a", "build/libsparkpipe_core.a"],
               cwd=ROOT, check=True, capture_output=True)
with tempfile.TemporaryDirectory(prefix="host-rdma-load-") as directory:
    source = pathlib.Path(directory) / "probe.c"
    binary = pathlib.Path(directory) / "probe"
    source.write_text(PROBE)
    subprocess.run(["cc", "-std=c11", "-I", str(ROOT / "include"), str(source), str(ROOT / "build/libsparkpipe_model_common.a"),
                    str(ROOT / "build/libsparkpipe_core.a"), "-ldl", "-lpthread", "-o", str(binary)], check=True)
    subprocess.run([str(binary), str(ROOT / "build/libhidden_transport_spark_host_rdma_verbs.so")], check=True, timeout=120)
