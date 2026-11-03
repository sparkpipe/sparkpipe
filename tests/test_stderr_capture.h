#pragma once

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct TestStderrCapture
{
	int saved;
	char path[64];
} TestStderrCapture;

static void TestStderrCaptureBegin(TestStderrCapture *capture)
{
	int file;
	strcpy(capture->path,"/tmp/sparkpipe-stderr-XXXXXX");
	file = mkstemp(capture->path);
	assert(file >= 0);
	(void)fflush(stderr);
	capture->saved = dup(2);
	assert(capture->saved >= 0 && dup2(file,2) == 2);
	(void)close(file);
}

static char *TestStderrCaptureEnd(TestStderrCapture *capture)
{
	FILE *file;
	char *text;
	long length;
	(void)fflush(stderr);
	assert(dup2(capture->saved,2) == 2);
	(void)close(capture->saved);
	file = fopen(capture->path,"rb");
	assert(file != 0 && fseek(file,0,SEEK_END) == 0);
	length = ftell(file);
	assert(length >= 0 && fseek(file,0,SEEK_SET) == 0);
	text = (char *)malloc((size_t)length + 1u);
	assert(text != 0 && fread(text,1u,(size_t)length,file) == (size_t)length);
	text[length] = '\0';
	(void)fclose(file);
	(void)unlink(capture->path);
	(void)fputs(text,stderr);
	return(text);
}
