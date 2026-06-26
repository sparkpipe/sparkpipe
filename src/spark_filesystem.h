#ifndef SPARKPIPE_SPARK_FILESYSTEM_H
#define SPARKPIPE_SPARK_FILESYSTEM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "sparkpipe/spark_status.h"

#define SPARK_INTERNAL_PATH_BYTES 4096u

void SparkSetError(char *error_buffer, uint32_t error_buffer_bytes, const char *format, ...);
SparkStatus SparkReadEntireFile(const char *path, char **data, size_t *data_bytes);
SparkStatus SparkWriteEntireFile(const char *path, const void *data, size_t data_bytes);
SparkStatus SparkWriteEntireFileAtomically(const char *path, const void *data, size_t data_bytes);
SparkStatus SparkCopyFile(const char *source_path, const char *destination_path);
SparkStatus SparkCreateDirectories(const char *path);
SparkStatus SparkRemoveDirectoryTree(const char *path);
SparkStatus SparkJoinPath(const char *left, const char *right, char *path, uint32_t path_bytes);
bool SparkPathExists(const char *path);
SparkStatus SparkRunProcess(const char *executable, char *const arguments[], int *exit_code);
SparkStatus SparkCopyString(char *destination, uint32_t destination_bytes, const char *source);
bool SparkCIdentifierIsValid(const char *identifier, bool allow_empty);

#endif
