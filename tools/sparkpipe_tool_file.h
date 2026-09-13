#ifndef SPARKPIPE_TOOL_FILE_H
#define SPARKPIPE_TOOL_FILE_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Read a whole file into a NUL-terminated buffer the caller frees, or return
// null. Four tools carried their own copy of this; it is one thing and it now
// lives in one place. Header-only so no tool needs new build wiring.
static char *SparkToolReadWholeFile(const char *path, uint32_t *text_bytes_out)
{
	FILE *file;
	long file_bytes;
	char *text;
	size_t read_bytes;

	if (path == 0 || text_bytes_out == 0)
	{
		return 0;
	}
	*text_bytes_out = 0u;
	file = fopen(path, "rb");
	if (file == 0)
	{
		return 0;
	}
	if (fseek(file, 0, SEEK_END) != 0 ||
		(file_bytes = ftell(file)) < 0 ||
		(uint64_t)file_bytes > 0xffffffffull ||
		fseek(file, 0, SEEK_SET) != 0)
	{
		fclose(file);
		return 0;
	}
	text = (char *)malloc((size_t)file_bytes + 1u);
	if (text == 0)
	{
		fclose(file);
		return 0;
	}
	read_bytes = fread(text, 1u, (size_t)file_bytes, file);
	fclose(file);
	if (read_bytes != (size_t)file_bytes)
	{
		free(text);
		return 0;
	}
	text[file_bytes] = '\0';
	*text_bytes_out = (uint32_t)file_bytes;
	return text;
}

#endif
