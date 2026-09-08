#define _POSIX_C_SOURCE 200809L
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include "weightd_spawn.h"

static int32_t SparkWeightdReadDigest(FILE *file,char digest[65])
{
	uint32_t i;
	int32_t tail;
	if ( fread(digest,1u,64u,file) != 64u )
		return(-1);
	for (i=0u; i<64u; i++)
		if ( (digest[i] < '0' || digest[i] > '9') && (digest[i] < 'a' || digest[i] > 'f') )
			return(-2);
	tail = fgetc(file);
	if ( tail != EOF && tail != ' ' && tail != '\t' && tail != '\n' )
		return(-3);
	if ( ferror(file) != 0 )
		return(-4);
	digest[64] = '\0';
	return(0);
}

static int32_t SparkWeightdFindDigest(DIR *directory,const char *root,char digest[65])
{
	struct dirent *entry;
	FILE *file;
	char path[1024];
	uint32_t found = 0u;
	int32_t bytes,status;
	errno = 0;
	while ( (entry = readdir(directory)) != 0 )
	{
		bytes = (int32_t)strlen(entry->d_name);
		if ( bytes <= 7 || strcmp(entry->d_name + bytes - 7,".sha256") != 0 )
			continue;
		if ( found != 0u )
			return(-5);
		bytes = snprintf(path,sizeof(path),"%s/packs/%s",root,entry->d_name);
		if ( bytes < 0 || (uint32_t)bytes >= sizeof(path) )
			return(-6);
		file = fopen(path,"rb");
		if ( file == 0 )
			return(-7);
		status = SparkWeightdReadDigest(file,digest);
		(void)fclose(file);
		if ( status != 0 )
			return(status);
		found = 1u;
		errno = 0;
	}
	if ( errno != 0 )
		return(-8);
	return(found != 0u ? 0 : -9);
}

static int32_t SparkWeightdResolveDigest(const char *root,char digest[65])
{
	DIR *directory;
	char path[1024];
	int32_t bytes,status;
	bytes = snprintf(path,sizeof(path),"%s/packs",root);
	if ( bytes < 0 || (uint32_t)bytes >= sizeof(path) )
		return(-10);
	directory = opendir(path);
	if ( directory == 0 )
		return(-11);
	status = SparkWeightdFindDigest(directory,root,digest);
	(void)closedir(directory);
	return(status);
}

int32_t SparkModelResidentdPrepareWeightd(const char *root,const char *socket_path)
{
	struct sockaddr_un address;
	char digest[65];
	const char *setting;
	int32_t fd,status;
	if ( root == 0 || socket_path == 0 || root[0] == '\0' || socket_path[0] == '\0' )
		return(-12);
	if ( strlen(socket_path) >= sizeof(address.sun_path) )
		return(-13);
	setting = getenv("SPARK_WEIGHTD_ATTACH");
	if ( setting != 0 && strcmp(setting,"0") == 0 )
		return(-14);
	status = SparkWeightdResolveDigest(root,digest);
	if ( status != 0 )
		return(status);
	fd = socket(AF_UNIX,SOCK_STREAM,0);
	if ( fd < 0 )
		return(-15);
	if ( fcntl(fd,F_SETFL,O_NONBLOCK) != 0 )
	{
		(void)close(fd);
		return(-19);
	}
	memset(&address,0,sizeof(address));
	address.sun_family = AF_UNIX;
	memcpy(address.sun_path,socket_path,strlen(socket_path) + 1u);
	status = connect(fd,(struct sockaddr *)&address,sizeof(address));
	(void)close(fd);
	if ( status != 0 )
		return(-16);
	if ( setenv("SPARK_WEIGHTD_SOCKET",socket_path,1) != 0 )
		return(-17);
	if ( setenv("SPARK_WEIGHTD_PACK_SHA256",digest,1) != 0 )
		return(-18);
	return(0);
}
