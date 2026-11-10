#pragma once

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

ssize_t SparkIoSend(int fd, const void *data, size_t bytes);
SparkStatus SparkIoReadAll(int fd, void *data, size_t bytes);
SparkStatus SparkIoWriteAll(int fd, const void *data, size_t bytes);
SparkStatus SparkIoRecvAll(int fd, void *data, size_t bytes);
SparkStatus SparkIoSendAll(int fd, const void *data, size_t bytes);
SparkStatus SparkIoRecvAllBefore(int fd, void *data, size_t bytes, uint64_t deadline_ns);
SparkStatus SparkIoSendAllBefore(int fd, const void *data, size_t bytes, uint64_t deadline_ns);

#ifdef __cplusplus
}
#endif
