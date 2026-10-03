#!/usr/bin/env python3
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

with tempfile.TemporaryDirectory(prefix="weightd-mesh-src-") as build:
    for source in ("node/weightd.c", "node/weightd_mesh.c"):
        subprocess.run(
            ["cc", "-std=c11", "-Wall", "-Wextra", "-Werror",
             "-D_GNU_SOURCE", "-pthread", "-c",
             "-Itests/ibv_stub", "-Itests/cuda_stub", "-Iinclude", source,
             "-o", str(Path(build) / (Path(source).stem + ".o"))],
            cwd=ROOT,
            check=True,
        )

with tempfile.TemporaryDirectory(prefix="weightd-startup-") as build:
    fixture = Path(build) / "startup.c"
    binary = Path(build) / "startup"
    fixture.write_text(r'''
#define socket TestSocket
#define setsockopt TestSetSockOpt
#define bind TestBind
#define listen TestListen
#define close TestClose
#define pthread_create TestThreadCreate
#include "node/weightd.c"
int TestSocket(int domain,int type,int protocol)
{ (void)domain; (void)type; (void)protocol; return 42; }
int TestSetSockOpt(int fd,int level,int name,const void *value,socklen_t bytes)
{ (void)fd; (void)level; (void)name; (void)value; (void)bytes; return 0; }
int TestBind(int fd,const struct sockaddr *address,socklen_t bytes)
{ (void)fd; (void)address; (void)bytes; return 0; }
int TestListen(int fd,int backlog)
{ (void)fd; (void)backlog; return 0; }
int TestClose(int fd) { (void)fd; return 0; }
int TestThreadCreate(pthread_t *thread,const pthread_attr_t *attributes,
    void *(*start)(void *),void *argument)
{ (void)thread; (void)attributes; (void)start; (void)argument; return EAGAIN; }
SparkStatus SparkWeightdMeshInit(uint32_t rank,const char *interface_name,
    uint32_t sgid,const char *directory,uint32_t mask,
    const char *pair_interface_name,uint32_t pair_sgid,uint32_t traffic_class)
{
    (void)rank; (void)interface_name; (void)sgid; (void)directory; (void)mask;
    (void)pair_interface_name; (void)pair_sgid; (void)traffic_class;
    return getenv("TEST_MESH_INIT_OK") != 0 ? SPARK_STATUS_BUSY : SPARK_STATUS_IO_ERROR;
}
void SparkWeightdMeshDoorbellLoop(int32_t cpu) { (void)cpu; }
SparkStatus SparkWeightdServerCreate(const SparkWeightdServerConfig *config,
    SparkWeightdServer **server)
{ (void)config; *server = (SparkWeightdServer *)(uintptr_t)1u; return SPARK_STATUS_OK; }
SparkStatus SparkWeightdServerRun(SparkWeightdServer *server,const volatile sig_atomic_t *stop)
{ (void)server; (void)stop; puts("SERVER-RUN"); return SPARK_STATUS_OK; }
void SparkWeightdServerDestroy(SparkWeightdServer *server)
{ (void)server; puts("SERVER-DESTROY"); }
uint32_t SparkWeightdServerArenaCount(const SparkWeightdServer *server)
{ (void)server; return 0u; }
uint64_t SparkWeightdServerResidentBytes(const SparkWeightdServer *server)
{ (void)server; return 0u; }
const char *SparkStatusToString(SparkStatus status)
{ (void)status; return "test-status"; }
''', encoding="utf-8")
    subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-D_GNU_SOURCE",
                    "-pthread", "-I.", "-Iinclude", str(fixture), "-o", str(binary)],
                   cwd=ROOT, check=True)
    arguments = [str(binary), "--mesh-rank", "0", "--mesh-interface", "test0",
                 "--mesh-sgid-index", "3", "--mesh-rank-mask", "0xf"]
    for environment, diagnostic in (({}, "init=test-status"),
                                    ({"TEST_MESH_INIT_OK": "1"}, "thread create failed")):
        result = subprocess.run(arguments, env=environment, capture_output=True, text=True)
        assert result.returncode == 1, result
        assert diagnostic in result.stderr, result
        assert "ready" not in result.stdout and "SERVER-RUN" not in result.stdout, result
        assert result.stdout.count("SERVER-DESTROY") == 1, result
    for flag in range(1, len(arguments), 2):
        partial = arguments[:flag] + arguments[flag + 2:]
        result = subprocess.run(partial, env={"TEST_MESH_INIT_OK": "1"}, capture_output=True, text=True)
        assert result.returncode == 2 and "SERVER-RUN" not in result.stdout, (arguments[flag], result)
    outside = arguments[:2] + ["4"] + arguments[3:]
    result = subprocess.run(outside, env={"TEST_MESH_INIT_OK": "1"}, capture_output=True, text=True)
    assert result.returncode == 2 and "SERVER-RUN" not in result.stdout, result
    for value in ("", "x", "-1", "1x", "100000"):
        result = subprocess.run(arguments, env={"SPARK_WEIGHTD_MESH_DOORBELL_CPU": value}, capture_output=True, text=True)
        assert result.returncode == 2 and "bad SPARK_WEIGHTD_MESH_DOORBELL_CPU" in result.stderr, (value, result)
    result = subprocess.run(arguments, env={"SPARK_WEIGHTD_MESH_DOORBELL_CPU": "0", "TEST_MESH_INIT_OK": "1"}, capture_output=True, text=True)
    assert result.returncode == 1 and "thread create failed" in result.stderr, result
    result = subprocess.run([str(binary)], env={}, capture_output=True, text=True)
    assert result.returncode == 0 and "SERVER-RUN" in result.stdout, result
