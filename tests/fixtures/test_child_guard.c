#include "test_child_guard.h"

#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#if defined(__linux__)
#include <sys/prctl.h>
#endif

static pid_t TestChildGuardOwner;
static uint32_t TestChildGuardAtexitArmed;
static volatile sig_atomic_t TestChildGuardCount;
static pid_t TestChildGuardPids[TEST_CHILD_GUARD_CAPACITY];
static const int32_t TestChildGuardSignals[] =
{
	SIGABRT,SIGSEGV,SIGBUS,SIGILL,SIGFPE,SIGTERM,SIGINT,SIGHUP,SIGQUIT,SIGALRM,SIGXCPU,SIGUSR1,SIGUSR2
};

static void TestChildGuardSleepMs(uint32_t milliseconds)
{
	struct timespec delay;
	delay.tv_sec = (time_t)(milliseconds / 1000u);
	delay.tv_nsec = (long)(milliseconds % 1000u) * 1000000L;
	(void)nanosleep(&delay,0);
}

static void TestChildGuardReapOne(pid_t pid)
{
	pid_t probe;
	uint32_t waited;
	int status;
	probe = waitpid(pid,&status,WNOHANG);
	if ( probe != 0 && probe != pid )
		return;
	(void)kill(-pid,SIGKILL);
	(void)kill(pid,SIGKILL);
	if ( probe == pid )
		return;
	for (waited=0u; waited<TEST_CHILD_GUARD_REAP_WAIT_MS; waited+=10u)
	{
		probe = waitpid(pid,&status,WNOHANG);
		if ( probe == pid || (probe < 0 && errno != EINTR) )
			return;
		TestChildGuardSleepMs(10u);
	}
}

void TestChildGuardKillAll(void)
{
	uint32_t index,count;
	pid_t pid;
	if ( TestChildGuardOwner == 0 || getpid() != TestChildGuardOwner )
		return;
	count = (uint32_t)TestChildGuardCount;
	for (index=0u; index<count; index++)
	{
		pid = TestChildGuardPids[index];
		TestChildGuardPids[index] = 0;
		if ( pid > 0 )
			TestChildGuardReapOne(pid);
	}
	TestChildGuardCount = 0;
}

static void TestChildGuardFatalSignal(int signal_number)
{
	int saved_errno;
	saved_errno = errno;
	TestChildGuardKillAll();
	errno = saved_errno;
	(void)signal(signal_number,SIG_DFL);
	(void)raise(signal_number);
}

static void TestChildGuardInstallSignals(void)
{
	struct sigaction current,handler;
	uint32_t index;
	memset(&handler,0,sizeof(handler));
	handler.sa_handler = TestChildGuardFatalSignal;
	handler.sa_flags = SA_NODEFER;
	(void)sigemptyset(&handler.sa_mask);
	for (index=0u; index<sizeof(TestChildGuardSignals)/sizeof(TestChildGuardSignals[0]); index++)
	{
		if ( sigaction(TestChildGuardSignals[index],0,&current) != 0 )
			continue;
		if ( current.sa_handler == SIG_DFL && (current.sa_flags & SA_SIGINFO) == 0 )
			(void)sigaction(TestChildGuardSignals[index],&handler,0);
	}
}

static unsigned long long TestChildGuardStartTicks(void)
{
#if defined(__linux__)
	char stat_text[1024];
	const char *cursor;
	unsigned long long ticks;
	uint32_t field;
	size_t length;
	FILE *file;
	file = fopen("/proc/self/stat","r");
	if ( file == 0 )
		return(0u);
	length = fread(stat_text,1u,sizeof(stat_text) - 1u,file);
	(void)fclose(file);
	stat_text[length] = '\0';
	cursor = strrchr(stat_text,')');
	if ( cursor == 0 )
		return(0u);
	for (field=2u; field<22u && cursor != 0; field++)
		cursor = strchr(cursor + 1,' ');
	if ( cursor == 0 || sscanf(cursor + 1,"%llu",&ticks) != 1 )
		return(0u);
	return(ticks);
#else
	return(0u);
#endif
}

static int32_t TestChildGuardArm(void)
{
	char owner[64];
	pid_t self;
	self = getpid();
	if ( TestChildGuardOwner == self )
		return(0);
	TestChildGuardOwner = self;
	TestChildGuardCount = 0;
	memset(TestChildGuardPids,0,sizeof(TestChildGuardPids));
	if ( snprintf(owner,sizeof(owner),"%ld.%llu",(long)self,TestChildGuardStartTicks()) <= 0 ||
		setenv(TEST_CHILD_GUARD_OWNER_VARIABLE,owner,1) != 0 )
		return(-1);
	if ( TestChildGuardAtexitArmed == 0u )
	{
		if ( atexit(TestChildGuardKillAll) != 0 )
			return(-1);
		TestChildGuardAtexitArmed = 1u;
	}
	TestChildGuardInstallSignals();
	return(0);
}

pid_t TestChildGuardFork(void)
{
	sigset_t blocked,previous;
	pid_t child,owner;
	uint32_t index;
	if ( TestChildGuardArm() != 0 )
		return(-1);
	if ( (uint32_t)TestChildGuardCount >= TEST_CHILD_GUARD_CAPACITY )
	{
		errno = EAGAIN;
		return(-1);
	}
	owner = getpid();
	(void)sigemptyset(&blocked);
	for (index=0u; index<sizeof(TestChildGuardSignals)/sizeof(TestChildGuardSignals[0]); index++)
		(void)sigaddset(&blocked,TestChildGuardSignals[index]);
	(void)sigprocmask(SIG_BLOCK,&blocked,&previous);
	child = fork();
	if ( child == 0 )
	{
		(void)setpgid(0,0);
#if defined(__linux__)
		if ( prctl(PR_SET_PDEATHSIG,SIGKILL) != 0 )
			_exit(125);
#endif
		if ( getppid() != owner )
			_exit(125);
		(void)sigprocmask(SIG_SETMASK,&previous,0);
		return(0);
	}
	if ( child > 0 )
	{
		(void)setpgid(child,child);
		TestChildGuardPids[TestChildGuardCount] = child;
		TestChildGuardCount = TestChildGuardCount + 1;
	}
	(void)sigprocmask(SIG_SETMASK,&previous,0);
	return(child);
}

uint32_t TestChildGuardTrackedCount(void)
{
	siginfo_t info;
	uint32_t index,count,live;
	live = 0u;
	count = (uint32_t)TestChildGuardCount;
	for (index=0u; index<count; index++)
	{
		if ( TestChildGuardPids[index] <= 0 )
			continue;
		memset(&info,0,sizeof(info));
		if ( waitid(P_PID,(id_t)TestChildGuardPids[index],&info,WEXITED | WNOHANG | WNOWAIT) == 0 && info.si_pid == 0 )
			live++;
	}
	return(live);
}
