#define _POSIX_C_SOURCE 200809L

#include "spark_filesystem.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define SPARK_FILE_COPY_BUFFER_BYTES 65536u

void SparkSetError(char *error_buffer, uint32_t error_buffer_bytes, const char *format, ...)
{
    va_list arguments;

    if (error_buffer == 0 || error_buffer_bytes == 0u || format == 0)
    {
        return;
    }

    va_start(arguments, format);
    vsnprintf(error_buffer, error_buffer_bytes, format, arguments);
    va_end(arguments);
}

SparkStatus SparkReadEntireFile(const char *path, char **data, size_t *data_bytes)
{
    FILE *file;
    long file_bytes;
    char *buffer;
    size_t bytes_read;

    if (path == 0 || data == 0 || data_bytes == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *data = 0;
    *data_bytes = 0u;

    file = fopen(path, "rb");
    if (file == 0)
    {
        return SPARK_STATUS_IO_ERROR;
    }
    if (fseek(file, 0, SEEK_END) != 0)
    {
        fclose(file);
        return SPARK_STATUS_IO_ERROR;
    }
    file_bytes = ftell(file);
    if (file_bytes < 0 || fseek(file, 0, SEEK_SET) != 0)
    {
        fclose(file);
        return SPARK_STATUS_IO_ERROR;
    }

    buffer = (char *)malloc((size_t)file_bytes + 1u);
    if (buffer == 0)
    {
        fclose(file);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    bytes_read = fread(buffer, 1u, (size_t)file_bytes, file);
    if (bytes_read != (size_t)file_bytes || ferror(file) != 0)
    {
        free(buffer);
        fclose(file);
        return SPARK_STATUS_IO_ERROR;
    }
    if (fclose(file) != 0)
    {
        free(buffer);
        return SPARK_STATUS_IO_ERROR;
    }

    buffer[bytes_read] = '\0';
    *data = buffer;
    *data_bytes = bytes_read;
    return SPARK_STATUS_OK;
}

SparkStatus SparkWriteEntireFile(const char *path, const void *data, size_t data_bytes)
{
    FILE *file;
    size_t bytes_written;
    SparkStatus status;

    if (path == 0 || (data == 0 && data_bytes != 0u))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    file = fopen(path, "wb");
    if (file == 0)
    {
        return SPARK_STATUS_IO_ERROR;
    }
    status = SPARK_STATUS_OK;
    bytes_written = fwrite(data, 1u, data_bytes, file);
    if (bytes_written != data_bytes)
    {
        status = SPARK_STATUS_IO_ERROR;
    }
    if (fflush(file) != 0)
    {
        status = SPARK_STATUS_IO_ERROR;
    }
    if (fclose(file) != 0)
    {
        status = SPARK_STATUS_IO_ERROR;
    }
    return status;
}

SparkStatus SparkWriteEntireFileAtomically(const char *path, const void *data, size_t data_bytes)
{
    char temporary_path[SPARK_INTERNAL_PATH_BYTES];
    SparkStatus status;

    if (path == 0 || (data == 0 && data_bytes != 0u))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (snprintf(temporary_path, sizeof(temporary_path), "%s.tmp.%ld", path, (long)getpid()) >= (int)sizeof(temporary_path))
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    status = SparkWriteEntireFile(temporary_path, data, data_bytes);
    if (status != SPARK_STATUS_OK)
    {
        unlink(temporary_path);
        return status;
    }
    if (rename(temporary_path, path) != 0)
    {
        unlink(temporary_path);
        return SPARK_STATUS_IO_ERROR;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkCopyFile(const char *source_path, const char *destination_path)
{
    FILE *source;
    FILE *destination;
    uint8_t buffer[SPARK_FILE_COPY_BUFFER_BYTES];
    size_t bytes_read;
    SparkStatus status;

    if (source_path == 0 || destination_path == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    source = fopen(source_path, "rb");
    if (source == 0)
    {
        return SPARK_STATUS_IO_ERROR;
    }
    destination = fopen(destination_path, "wb");
    if (destination == 0)
    {
        fclose(source);
        return SPARK_STATUS_IO_ERROR;
    }

    status = SPARK_STATUS_OK;
    while ((bytes_read = fread(buffer, 1u, sizeof(buffer), source)) != 0u)
    {
        if (fwrite(buffer, 1u, bytes_read, destination) != bytes_read)
        {
            status = SPARK_STATUS_IO_ERROR;
            break;
        }
    }
    if (ferror(source) != 0)
    {
        status = SPARK_STATUS_IO_ERROR;
    }
    if (fflush(destination) != 0)
    {
        status = SPARK_STATUS_IO_ERROR;
    }
    if (fclose(source) != 0)
    {
        status = SPARK_STATUS_IO_ERROR;
    }
    if (fclose(destination) != 0)
    {
        status = SPARK_STATUS_IO_ERROR;
    }
    if (status != SPARK_STATUS_OK)
    {
        unlink(destination_path);
    }
    return status;
}

SparkStatus SparkCreateDirectories(const char *path)
{
    char mutable_path[SPARK_INTERNAL_PATH_BYTES];
    size_t path_length;
    size_t character_index;

    if (path == 0 || path[0] == '\0')
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    path_length = strlen(path);
    if (path_length >= sizeof(mutable_path))
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    memcpy(mutable_path, path, path_length + 1u);

    for (character_index = 1u; character_index < path_length; ++character_index)
    {
        if (mutable_path[character_index] == '/')
        {
            mutable_path[character_index] = '\0';
            if (mkdir(mutable_path, 0775) != 0 && errno != EEXIST)
            {
                return SPARK_STATUS_IO_ERROR;
            }
            mutable_path[character_index] = '/';
        }
    }
    if (mkdir(mutable_path, 0775) != 0 && errno != EEXIST)
    {
        return SPARK_STATUS_IO_ERROR;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkRemoveDirectoryTree(const char *path)
{
    struct stat file_status;

    if (path == 0 || path[0] == '\0' || strcmp(path, "/") == 0 ||
        strcmp(path, ".") == 0 || strcmp(path, "..") == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (lstat(path, &file_status) != 0)
    {
        return errno == ENOENT ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
    }
    if (!S_ISDIR(file_status.st_mode))
    {
        return unlink(path) == 0 ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
    }

    {
        DIR *directory;
        struct dirent *directory_entry;
        SparkStatus status;

        directory = opendir(path);
        if (directory == 0)
        {
            return SPARK_STATUS_IO_ERROR;
        }
        status = SPARK_STATUS_OK;
        errno = 0;
        while ((directory_entry = readdir(directory)) != 0)
        {
            char child_path[SPARK_INTERNAL_PATH_BYTES];

            if (strcmp(directory_entry->d_name, ".") == 0 ||
                strcmp(directory_entry->d_name, "..") == 0)
            {
                continue;
            }
            status = SparkJoinPath(path, directory_entry->d_name, child_path, sizeof(child_path));
            if (status != SPARK_STATUS_OK)
            {
                break;
            }
            status = SparkRemoveDirectoryTree(child_path);
            if (status != SPARK_STATUS_OK)
            {
                break;
            }
            errno = 0;
        }
        if (status == SPARK_STATUS_OK && errno != 0)
        {
            status = SPARK_STATUS_IO_ERROR;
        }
        if (closedir(directory) != 0 && status == SPARK_STATUS_OK)
        {
            status = SPARK_STATUS_IO_ERROR;
        }
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }

    return rmdir(path) == 0 ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
}

SparkStatus SparkJoinPath(const char *left, const char *right, char *path, uint32_t path_bytes)
{
    int formatted_bytes;
    const char *separator;

    if (left == 0 || right == 0 || path == 0 || path_bytes == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    separator = left[0] != '\0' && left[strlen(left) - 1u] == '/' ? "" : "/";
    formatted_bytes = snprintf(path, path_bytes, "%s%s%s", left, separator, right);
    if (formatted_bytes < 0 || (uint32_t)formatted_bytes >= path_bytes)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    return SPARK_STATUS_OK;
}

bool SparkPathExists(const char *path)
{
    struct stat file_status;

    return path != 0 && stat(path, &file_status) == 0;
}

SparkStatus SparkRunProcess(const char *executable, char *const arguments[], int *exit_code)
{
    pid_t child_process;
    int wait_status;

    if (executable == 0 || arguments == 0 || exit_code == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *exit_code = -1;

    child_process = fork();
    if (child_process < 0)
    {
        return SPARK_STATUS_IO_ERROR;
    }
    if (child_process == 0)
    {
        execvp(executable, arguments);
        _exit(127);
    }
    if (waitpid(child_process, &wait_status, 0) < 0)
    {
        return SPARK_STATUS_IO_ERROR;
    }
    if (WIFEXITED(wait_status))
    {
        *exit_code = WEXITSTATUS(wait_status);
    }
    else if (WIFSIGNALED(wait_status))
    {
        *exit_code = 128 + WTERMSIG(wait_status);
    }
    else
    {
        *exit_code = 255;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkCopyString(char *destination, uint32_t destination_bytes, const char *source)
{
    size_t source_bytes;

    if (destination == 0 || destination_bytes == 0u || source == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    source_bytes = strlen(source);
    if (source_bytes >= destination_bytes)
    {
        destination[0] = '\0';
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    memcpy(destination, source, source_bytes + 1u);
    return SPARK_STATUS_OK;
}

bool SparkCIdentifierIsValid(const char *identifier, bool allow_empty)
{
    size_t character_index;

    if (identifier == 0)
    {
        return false;
    }
    if (identifier[0] == '\0')
    {
        return allow_empty;
    }
    if (!((identifier[0] >= 'A' && identifier[0] <= 'Z') ||
          (identifier[0] >= 'a' && identifier[0] <= 'z') ||
          identifier[0] == '_'))
    {
        return false;
    }
    for (character_index = 1u; identifier[character_index] != '\0'; ++character_index)
    {
        char character;

        character = identifier[character_index];
        if (!((character >= 'A' && character <= 'Z') ||
              (character >= 'a' && character <= 'z') ||
              (character >= '0' && character <= '9') ||
              character == '_'))
        {
            return false;
        }
    }
    return true;
}
