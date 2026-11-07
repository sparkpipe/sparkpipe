#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <fcntl.h>
#include <ftw.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "sparkpipe/spark_kv_snapshot.h"
#include "sparkpipe/spark_working_state.h"

static uint32_t checks,failures;
#define CHECK(condition) do { checks++; if ( !(condition) ) { failures++; fprintf(stderr,"FAIL line=%d %s\n",__LINE__,#condition); } } while (0)

static void WriteText(const char *path,const char *text)
{
	FILE *file = fopen(path,"w");
	CHECK(file != 0);
	if ( file == 0 )
		return;
	CHECK(fputs(text,file) >= 0);
	CHECK(fclose(file) == 0);
}

static void WriteAged(const char *path,const char *text,uint64_t age_seconds)
{
	struct timespec times[2];
	WriteText(path,text);
	times[0].tv_sec = times[1].tv_sec = time(0) - (time_t)age_seconds;
	times[0].tv_nsec = times[1].tv_nsec = 0;
	CHECK(utimensat(AT_FDCWD,path,times,AT_SYMLINK_NOFOLLOW) == 0);
}

static void Age(const char *path,uint64_t age_seconds)
{
	struct timespec times[2];
	times[0].tv_sec = times[1].tv_sec = time(0) - (time_t)age_seconds;
	times[0].tv_nsec = times[1].tv_nsec = 0;
	CHECK(utimensat(AT_FDCWD,path,times,AT_SYMLINK_NOFOLLOW) == 0);
}

static uint32_t Exists(const char *path)
{
	struct stat info;
	return(lstat(path,&info) == 0 ? 1u : 0u);
}

static void Join(char *path,size_t capacity,const char *left,const char *right)
{
	CHECK((size_t)snprintf(path,capacity,"%s/%s",left,right) < capacity);
}

static int RemoveEntry(const char *path,const struct stat *info,int flag,struct FTW *walk)
{
	(void)info;
	(void)flag;
	(void)walk;
	return(remove(path));
}

static void TestParse(void)
{
	static const char good[] = "sparkpipe-working-state 2\nwiped 1764550800\nheld_until 1764554400\nstorage_wiped 1764540000\n";
	static const char *bad[] = {
		"",
		"sparkpipe-working-state 1\nwiped 1\nheld_until 2\n",
		"sparkpipe-working-state 1\nwiped 1\nheld_until 2\nstorage_wiped 1\n",
		"sparkpipe-working-state 2\nwiped 1\nheld_until 2\n",
		"sparkpipe-working-state 2\nheld_until 2\nwiped 1\nstorage_wiped 1\n",
		"sparkpipe-working-state 2\nwiped 1\nstorage_wiped 1\nheld_until 2\n",
		"sparkpipe-working-state 2\nwiped 1\n",
		"sparkpipe-working-state 2\nwiped 1\nheld_until 2\nstorage_wiped 1",
		"sparkpipe-working-state 2\nwiped 1\nheld_until 2\nstorage_wiped 1\nextra 3\n",
		"sparkpipe-working-state 2\nwiped 01\nheld_until 2\nstorage_wiped 1\n",
		"sparkpipe-working-state 2\nwiped -1\nheld_until 2\nstorage_wiped 0\n",
		"sparkpipe-working-state 2\nwiped  1\nheld_until 2\nstorage_wiped 1\n",
		"sparkpipe-working-state 2\nwiped 1x\nheld_until 2\nstorage_wiped 1\n",
		"sparkpipe-working-state 2\nwiped 1099511627777\nheld_until 2\nstorage_wiped 1\n",
		"sparkpipe-working-state 2\nwiped 99999999999999999999999\nheld_until 2\nstorage_wiped 1\n",
		"sparkpipe-working-state 2\nwiped 5\nheld_until 2\nstorage_wiped 6\n",
	};
	SparkWorkingStateDirective directive;
	uint32_t index;
	CHECK(SparkWorkingStateDirectiveParse(good,(uint32_t)strlen(good),&directive) == SPARK_STATUS_OK &&
		directive.wiped_seconds == UINT64_C(1764550800) && directive.held_until_seconds == UINT64_C(1764554400) && directive.storage_wiped_seconds == UINT64_C(1764540000));
	CHECK(SparkWorkingStateDirectiveParse("sparkpipe-working-state 2\nwiped 0\nheld_until 0\nstorage_wiped 0\n",63u,&directive) == SPARK_STATUS_OK &&
		directive.wiped_seconds == 0u && directive.held_until_seconds == 0u && directive.storage_wiped_seconds == 0u);
	for (index=0u; index<sizeof(bad)/sizeof(bad[0]); index++)
	{
		directive.wiped_seconds = directive.held_until_seconds = directive.storage_wiped_seconds = 7u;
		CHECK(SparkWorkingStateDirectiveParse(bad[index],(uint32_t)strlen(bad[index]),&directive) == SPARK_STATUS_PARSE_ERROR &&
			directive.wiped_seconds == 0u && directive.held_until_seconds == 0u && directive.storage_wiped_seconds == 0u);
	}
	CHECK(SparkWorkingStateDirectiveParse(good,(uint32_t)strlen(good),0) == SPARK_STATUS_INVALID_ARGUMENT);
	printf("working state: a directive parses only in its exact four-line form with storage_wiped no later than wiped, and anything else is refused as a parse error with a zero directive\n");
}

static void TestRead(const char *root)
{
	SparkWorkingStateDirective directive;
	char path[1024],home[1024];
	Join(path,sizeof(path),root,"directive");
	directive.wiped_seconds = 9u;
	CHECK(SparkWorkingStateDirectiveRead(path,&directive) == SPARK_STATUS_OK && directive.wiped_seconds == 0u && directive.held_until_seconds == 0u);
	WriteText(path,"sparkpipe-working-state 2\nwiped 100\nheld_until 200\nstorage_wiped 50\n");
	CHECK(SparkWorkingStateDirectiveRead(path,&directive) == SPARK_STATUS_OK && directive.wiped_seconds == 100u && directive.held_until_seconds == 200u &&
		directive.storage_wiped_seconds == 50u);
	WriteText(path,"sparkpipe-working-state 2\nwiped 100\n");
	CHECK(SparkWorkingStateDirectiveRead(path,&directive) == SPARK_STATUS_PARSE_ERROR && directive.wiped_seconds == 0u);
	CHECK(SparkWorkingStateDirectiveRead("relative/directive",&directive) == SPARK_STATUS_INVALID_ARGUMENT);
	CHECK(SparkWorkingStateDirectivePath("/home/spark0",path,sizeof(path)) == SPARK_STATUS_OK &&
		strcmp(path,"/home/spark0/sparkdata/working-state/directive") == 0);
	CHECK(SparkWorkingStateDirectivePath("/home/spark0",path,10u) == SPARK_STATUS_CAPACITY_EXCEEDED);
	CHECK(SparkWorkingStateDirectivePath("relative",path,sizeof(path)) == SPARK_STATUS_INVALID_ARGUMENT);
	CHECK(SparkWorkingStateOwnerRoot(home,sizeof(home)) == SPARK_STATUS_OK && home[0] == '/');
	printf("working state: a missing directive reads as wiped 0 and held_until 0, a malformed one is refused, and the path sits under the owner's home\n");
}

static void TestRule(void)
{
	const uint64_t day = SPARK_KV_RETENTION_SECONDS,now = UINT64_C(2000000000);
	SparkWorkingStateDirective none = {0u,0u,0u},wiped = {now - 600u,0u,now - 600u},held = {0u,now - 3600u,0u},both = {now - 600u,now + 1800u,now - 600u};
	SparkWorkingStateDirective cache_only = {now - 600u,0u,now - 7200u};
	CHECK(day == UINT64_C(86400));
	CHECK(SparkWorkingStateExpired(&none,now - day + 1u,now) == 0u && SparkWorkingStateExpired(&none,now - day,now) == 1u);
	CHECK(SparkWorkingStateExpired(0,now - day + 1u,now) == 0u && SparkWorkingStateExpired(0,now - day,now) == 1u);
	CHECK(SparkWorkingStateExpired(&wiped,now - 601u,now) == 1u && SparkWorkingStateExpired(&wiped,now - 600u,now) == 0u && SparkWorkingStateExpired(&wiped,now,now) == 0u);
	CHECK(SparkWorkingStateExpired(&held,now - 3u * day,now) == 0u && SparkWorkingStateExpired(&held,now - 3u * day,now + day - 3601u) == 0u &&
		SparkWorkingStateExpired(&held,now - 3u * day,now + day - 3600u) == 1u);
	CHECK(SparkWorkingStateExpired(&held,now - 60u,now + day - 3600u) == 0u && SparkWorkingStateExpired(&held,now - 60u,now + day - 60u) == 1u);
	CHECK(SparkWorkingStateExpired(&both,now - 601u,now) == 1u && SparkWorkingStateExpired(&both,now - 599u,now + day) == 0u &&
		SparkWorkingStateExpired(&both,now - 599u,now + day + 1800u) == 1u);
	CHECK(SparkWorkingStateExpired(&cache_only,now - 601u,now) == 1u && SparkWorkingStateStorageExpired(&cache_only,now - 601u,now) == 0u &&
		SparkWorkingStateStorageExpired(&cache_only,now - 7201u,now) == 1u && SparkWorkingStateStorageExpired(&cache_only,now - 7200u,now + day - 7201u) == 0u &&
		SparkWorkingStateStorageExpired(&cache_only,now - 7200u,now + day - 7200u) == 1u);
	CHECK(SparkWorkingStateStorageExpired(&both,now - 601u,now) == 1u && SparkWorkingStateStorageExpired(&both,now - 599u,now + day + 1800u) == 1u &&
		SparkWorkingStateStorageExpired(0,now - day + 1u,now) == 0u);
	printf("working state: state last used before the wipe is expired at once; otherwise it lives 24 hours past the later of its last use and held_until; storage follows storage_wiped, so a cache-only deletion keeps it\n");
}

static void MakeArea(const char *area,uint64_t stamp_age)
{
	char path[1024],nested[1024],deep[1024];
	CHECK(mkdir(area,0700) == 0);
	Join(path,sizeof(path),area,"notes.txt");
	WriteText(path,"repository notes");
	Join(nested,sizeof(nested),area,"repo");
	CHECK(mkdir(nested,0700) == 0);
	Join(deep,sizeof(deep),nested,"main.c");
	WriteText(deep,"int main(void) { return 0; }");
	Join(path,sizeof(path),area,SPARK_WORKING_STATE_LAST_USE_NAME);
	WriteAged(path,"",stamp_age);
}

static void TestScratch(const char *root)
{
	const uint64_t day = SPARK_KV_RETENTION_SECONDS;
	SparkWorkingStateDirective directive = {0u,0u,0u};
	SparkWorkingStatePurgeResult result;
	char base[1024],area[1024],mirror[1024],mirrored[1024],outside[1024],link[1024],file[1024],secret[1024];
	uint64_t now = (uint64_t)time(0);
	Join(base,sizeof(base),root,"sparkdata");
	CHECK(mkdir(base,0700) == 0);
	Join(base,sizeof(base),root,"sparkdata/working-state");
	CHECK(mkdir(base,0700) == 0);
	Join(area,sizeof(area),base,"scratch");
	CHECK(SparkWorkingStatePurgeScratch(root,&directive,now,&result) == SPARK_STATUS_OK && result.areas == 0u && result.emptied_areas == 0u);
	MakeArea(area,7200u);
	Join(outside,sizeof(outside),root,"outside");
	CHECK(mkdir(outside,0700) == 0);
	Join(secret,sizeof(secret),outside,"keep.txt");
	WriteText(secret,"not scratch");
	Join(link,sizeof(link),area,"link");
	CHECK(symlink(outside,link) == 0);
	CHECK(SparkWorkingStatePurgeScratch(root,&directive,now,&result) == SPARK_STATUS_OK && result.areas == 1u && result.emptied_areas == 0u);
	Join(file,sizeof(file),area,"repo/main.c");
	CHECK(Exists(file) == 1u);
	directive.wiped_seconds = now - 3600u;
	CHECK(SparkWorkingStatePurgeScratch(root,&directive,now,&result) == SPARK_STATUS_OK && result.areas == 1u && result.emptied_areas == 0u && Exists(file) == 1u);
	directive.storage_wiped_seconds = now - 3600u;
	CHECK(SparkWorkingStatePurgeScratch(root,&directive,now,&result) == SPARK_STATUS_OK && result.areas == 1u && result.emptied_areas == 1u &&
		result.failed_unlinks == 0u && result.deleted_entries == 5u);
	CHECK(Exists(area) == 1u && Exists(file) == 0u && Exists(link) == 0u && Exists(secret) == 1u);
	CHECK(rmdir(area) == 0);
	MakeArea(area,day + 600u);
	directive.wiped_seconds = 0u;
	directive.storage_wiped_seconds = 0u;
	directive.held_until_seconds = now - 1800u;
	CHECK(SparkWorkingStatePurgeScratch(root,&directive,now,&result) == SPARK_STATUS_OK && result.emptied_areas == 0u && Exists(file) == 1u);
	CHECK(SparkWorkingStatePurgeScratch(root,&directive,now + day - 1800u,&result) == SPARK_STATUS_OK && result.emptied_areas == 1u && Exists(file) == 0u);
	CHECK(rmdir(area) == 0);
	MakeArea(area,0u);
	Join(file,sizeof(file),area,SPARK_WORKING_STATE_LAST_USE_NAME);
	CHECK(unlink(file) == 0);
	Age(area,day + 60u);
	directive.held_until_seconds = 0u;
	CHECK(SparkWorkingStatePurgeScratch(root,&directive,now,&result) == SPARK_STATUS_OK && result.emptied_areas == 1u);
	Join(mirror,sizeof(mirror),base,"mirror");
	CHECK(mkdir(mirror,0700) == 0);
	Join(mirrored,sizeof(mirrored),mirror,"1");
	CHECK(mkdir(mirrored,0700) == 0);
	Join(mirrored,sizeof(mirrored),mirror,"1/sparkdata");
	CHECK(mkdir(mirrored,0700) == 0);
	Join(mirrored,sizeof(mirrored),mirror,"1/sparkdata/working-state");
	CHECK(mkdir(mirrored,0700) == 0);
	Join(mirrored,sizeof(mirrored),mirror,"1/sparkdata/working-state/scratch");
	MakeArea(mirrored,600u);
	Join(file,sizeof(file),mirrored,"repo/main.c");
	CHECK(SparkWorkingStatePurgeScratch(root,&directive,now,&result) == SPARK_STATUS_OK && result.areas == 2u && result.emptied_areas == 0u && Exists(file) == 1u);
	directive.wiped_seconds = now + 1u;
	CHECK(SparkWorkingStatePurgeScratch(root,&directive,now,&result) == SPARK_STATUS_OK && result.areas == 2u && result.emptied_areas == 0u && Exists(file) == 1u);
	directive.storage_wiped_seconds = now + 1u;
	CHECK(SparkWorkingStatePurgeScratch(root,&directive,now,&result) == SPARK_STATUS_OK && result.areas == 2u && result.emptied_areas == 2u && Exists(file) == 0u);
	CHECK(SparkWorkingStatePurgeScratch("relative",&directive,now,&result) == SPARK_STATUS_INVALID_ARGUMENT);
	CHECK(SparkWorkingStatePurgeScratchArea(area,&directive,now,0) == SPARK_STATUS_INVALID_ARGUMENT);
	printf("working state: weightd empties a scratch area, its own or a partner's mirror, as a whole by its last-use stamp under the storage rule (storage_wiped, not wiped), never following a link out of it\n");
}

int main(void)
{
	char root[] = "/tmp/sparkpipe-working-state-XXXXXX";
	TestParse();
	TestRule();
	if ( mkdtemp(root) == 0 )
		return(2);
	TestRead(root);
	TestScratch(root);
	(void)nftw(root,RemoveEntry,16,FTW_DEPTH | FTW_PHYS);
	printf("test_working_state: %u checks, %u failures\n",checks,failures);
	return(failures == 0u ? 0 : 1);
}
