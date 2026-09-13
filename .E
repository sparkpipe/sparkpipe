typedef signed char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long long int64_t;

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef int8_t int_least8_t;
typedef int16_t int_least16_t;
typedef int32_t int_least32_t;
typedef int64_t int_least64_t;
typedef uint8_t uint_least8_t;
typedef uint16_t uint_least16_t;
typedef uint32_t uint_least32_t;
typedef uint64_t uint_least64_t;
typedef int8_t int_fast8_t;
typedef int16_t int_fast16_t;
typedef int32_t int_fast32_t;
typedef int64_t int_fast64_t;
typedef uint8_t uint_fast8_t;
typedef uint16_t uint_fast16_t;
typedef uint32_t uint_fast32_t;
typedef uint64_t uint_fast64_t;
typedef signed char __int8_t;
typedef unsigned char __uint8_t;
typedef short __int16_t;
typedef unsigned short __uint16_t;
typedef int __int32_t;
typedef unsigned int __uint32_t;
typedef long long __int64_t;
typedef unsigned long long __uint64_t;
typedef long __darwin_intptr_t;
typedef unsigned int __darwin_natural_t;
typedef int __darwin_ct_rune_t;
typedef union {
 char __mbstate8[128];
 long long _mbstateL;
} __mbstate_t;
typedef __mbstate_t __darwin_mbstate_t;
typedef long int __darwin_ptrdiff_t;
typedef long unsigned int __darwin_size_t;
typedef __builtin_va_list __darwin_va_list;
typedef int __darwin_wchar_t;
typedef __darwin_wchar_t __darwin_rune_t;
typedef int __darwin_wint_t;
typedef unsigned long __darwin_clock_t;
typedef __uint32_t __darwin_socklen_t;
typedef long __darwin_ssize_t;
typedef long __darwin_time_t;
typedef __int64_t __darwin_blkcnt_t;
typedef __int32_t __darwin_blksize_t;
typedef __int32_t __darwin_dev_t;
typedef unsigned int __darwin_fsblkcnt_t;
typedef unsigned int __darwin_fsfilcnt_t;
typedef __uint32_t __darwin_gid_t;
typedef __uint32_t __darwin_id_t;
typedef __uint64_t __darwin_ino64_t;
typedef __darwin_ino64_t __darwin_ino_t;
typedef __darwin_natural_t __darwin_mach_port_name_t;
typedef __darwin_mach_port_name_t __darwin_mach_port_t;
typedef __uint16_t __darwin_mode_t;
typedef __int64_t __darwin_off_t;
typedef __int32_t __darwin_pid_t;
typedef __uint32_t __darwin_sigset_t;
typedef __int32_t __darwin_suseconds_t;
typedef __uint32_t __darwin_uid_t;
typedef __uint32_t __darwin_useconds_t;
typedef unsigned char __darwin_uuid_t[16];
typedef char __darwin_uuid_string_t[37];
struct __darwin_pthread_handler_rec {
 void (*__routine)(void *);
 void *__arg;
 struct __darwin_pthread_handler_rec *__next;
};
struct _opaque_pthread_attr_t {
 long __sig;
 char __opaque[56];
};
struct _opaque_pthread_cond_t {
 long __sig;
 char __opaque[40];
};
struct _opaque_pthread_condattr_t {
 long __sig;
 char __opaque[8];
};
struct _opaque_pthread_mutex_t {
 long __sig;
 char __opaque[56];
};
struct _opaque_pthread_mutexattr_t {
 long __sig;
 char __opaque[8];
};
struct _opaque_pthread_once_t {
 long __sig;
 char __opaque[8];
};
struct _opaque_pthread_rwlock_t {
 long __sig;
 char __opaque[192];
};
struct _opaque_pthread_rwlockattr_t {
 long __sig;
 char __opaque[16];
};
struct _opaque_pthread_t {
 long __sig;
 struct __darwin_pthread_handler_rec *__cleanup_stack;
 char __opaque[8176];
};
typedef struct _opaque_pthread_attr_t __darwin_pthread_attr_t;
typedef struct _opaque_pthread_cond_t __darwin_pthread_cond_t;
typedef struct _opaque_pthread_condattr_t __darwin_pthread_condattr_t;
typedef unsigned long __darwin_pthread_key_t;
typedef struct _opaque_pthread_mutex_t __darwin_pthread_mutex_t;
typedef struct _opaque_pthread_mutexattr_t __darwin_pthread_mutexattr_t;
typedef struct _opaque_pthread_once_t __darwin_pthread_once_t;
typedef struct _opaque_pthread_rwlock_t __darwin_pthread_rwlock_t;
typedef struct _opaque_pthread_rwlockattr_t __darwin_pthread_rwlockattr_t;
typedef struct _opaque_pthread_t *__darwin_pthread_t;
typedef __darwin_intptr_t intptr_t;
typedef unsigned long uintptr_t;
typedef long int intmax_t;
typedef long unsigned int uintmax_t;


typedef int __darwin_nl_item;
typedef int __darwin_wctrans_t;
typedef __uint32_t __darwin_wctype_t;
typedef unsigned char u_int8_t;
typedef unsigned short u_int16_t;
typedef unsigned int u_int32_t;
typedef unsigned long long u_int64_t;
typedef int64_t register_t;
typedef u_int64_t user_addr_t;
typedef u_int64_t user_size_t;
typedef int64_t user_ssize_t;
typedef int64_t user_long_t;
typedef u_int64_t user_ulong_t;
typedef int64_t user_time_t;
typedef int64_t user_off_t;
typedef u_int64_t syscall_arg_t;
typedef __darwin_va_list va_list;
typedef __darwin_size_t size_t;

int renameat(int, const char *, int, const char *) __attribute__((availability(macosx,introduced=10.10)));
int renamex_np(const char *, const char *, unsigned int) __attribute__((availability(macosx,introduced=10.12))) __attribute__((availability(ios,introduced=10.0))) __attribute__((availability(tvos,introduced=10.0))) __attribute__((availability(watchos,introduced=3.0)));
int renameatx_np(int, const char *, int, const char *, unsigned int) __attribute__((availability(macosx,introduced=10.12))) __attribute__((availability(ios,introduced=10.0))) __attribute__((availability(tvos,introduced=10.0))) __attribute__((availability(watchos,introduced=3.0)));
int printf(const char * restrict, ...) __attribute__((__format__ (__printf__, 1, 2)));

typedef __darwin_off_t fpos_t;
struct __sbuf {
 unsigned char *_base;
 int _size;
};
struct __sFILEX;
typedef struct __sFILE {
 unsigned char *_p;
 int _r;
 int _w;
 short _flags;
 short _file;
 struct __sbuf _bf;
 int _lbfsize;
 void *_cookie;
 int (* _Nullable _close)(void *);
 int (* _Nullable _read) (void *, char *, int);
 fpos_t (* _Nullable _seek) (void *, fpos_t, int);
 int (* _Nullable _write)(void *, const char *, int);
 struct __sbuf _ub;
 struct __sFILEX *_extra;
 int _ur;
 unsigned char _ubuf[3];
 unsigned char _nbuf[1];
 struct __sbuf _lb;
 int _blksize;
 fpos_t _offset;
} FILE;
extern FILE *__stdinp __attribute__((__swift_attr__("nonisolated(unsafe)")));
extern FILE *__stdoutp __attribute__((__swift_attr__("nonisolated(unsafe)")));
extern FILE *__stderrp __attribute__((__swift_attr__("nonisolated(unsafe)")));
void clearerr(FILE *);
int fclose(FILE *);
int feof(FILE *);
int ferror(FILE *);
int fflush(FILE *);
int fgetc(FILE *);
int fgetpos(FILE * restrict, fpos_t *);
char *fgets(char * restrict, int, FILE *);
FILE *fopen(const char * restrict __filename, const char * restrict __mode) __asm("_" "fopen" );
int fprintf(FILE * restrict, const char * restrict, ...) __attribute__((__format__ (__printf__, 2, 3)));
int fputc(int, FILE *);
int fputs(const char * restrict, FILE * restrict) __asm("_" "fputs" );
size_t fread(void * restrict __ptr, size_t __size, size_t __nitems, FILE * restrict __stream);
FILE *freopen(const char * restrict, const char * restrict,
     FILE * restrict) __asm("_" "freopen" );
int fscanf(FILE * restrict, const char * restrict, ...) __attribute__((__format__ (__scanf__, 2, 3)));
int fseek(FILE *, long, int);
int fsetpos(FILE *, const fpos_t *);
long ftell(FILE *);
size_t fwrite(const void * restrict __ptr, size_t __size, size_t __nitems, FILE * restrict __stream) __asm("_" "fwrite" );
int getc(FILE *);
int getchar(void);
__attribute__((__deprecated__("This function is provided for compatibility reasons only.  Due to security concerns inherent in the design of gets(3), it is highly recommended that you use fgets(3) instead.")))
char *gets(char *);
void perror(const char *) __attribute__((__cold__));
int putc(int, FILE *);
int putchar(int);
int puts(const char *);
int remove(const char *);
int rename (const char *__old, const char *__new);
void rewind(FILE *);
int scanf(const char * restrict, ...) __attribute__((__format__ (__scanf__, 1, 2)));
void setbuf(FILE * restrict, char * restrict);
int setvbuf(FILE * restrict, char * restrict, int, size_t);
__attribute__((__availability__(swift, unavailable, message="Use snprintf instead.")))
__attribute__((__deprecated__("This function is provided for compatibility reasons only.  Due to security concerns inherent in the design of sprintf(3), it is highly recommended that you use snprintf(3) instead.")))
int sprintf(char * restrict, const char * restrict, ...) __attribute__((__format__ (__printf__, 2, 3)));
int sscanf(const char * restrict, const char * restrict, ...) __attribute__((__format__ (__scanf__, 2, 3)));
FILE *tmpfile(void);
__attribute__((__availability__(swift, unavailable, message="Use mkstemp(3) instead.")))
__attribute__((__deprecated__("This function is provided for compatibility reasons only.  Due to security concerns inherent in the design of tmpnam(3), it is highly recommended that you use mkstemp(3) instead.")))
char *tmpnam(char *);
int ungetc(int, FILE *);
int vfprintf(FILE * restrict, const char * restrict, va_list) __attribute__((__format__ (__printf__, 2, 0)));
int vprintf(const char * restrict, va_list) __attribute__((__format__ (__printf__, 1, 0)));
__attribute__((__availability__(swift, unavailable, message="Use vsnprintf instead.")))
__attribute__((__deprecated__("This function is provided for compatibility reasons only.  Due to security concerns inherent in the design of sprintf(3), it is highly recommended that you use vsnprintf(3) instead.")))
int vsprintf(char * restrict, const char * restrict, va_list) __attribute__((__format__ (__printf__, 2, 0)));
char *ctermid(char *);
FILE *fdopen(int, const char *) __asm("_" "fdopen" );
int fileno(FILE *);
int pclose(FILE *) __attribute__((__availability__(swift, unavailable, message="Use posix_spawn APIs or NSTask instead. (On iOS, process spawning is unavailable.)")));
FILE *popen(const char *, const char *) __asm("_" "popen" ) __attribute__((__availability__(swift, unavailable, message="Use posix_spawn APIs or NSTask instead. (On iOS, process spawning is unavailable.)")));
int __srget(FILE *);
int __svfscanf(FILE *, const char *, va_list) __attribute__((__format__ (__scanf__, 2, 0)));
int __swbuf(int, FILE *);
inline __attribute__ ((__always_inline__)) int __sputc(int _c, FILE *_p) {
 if (--_p->_w >= 0 || (_p->_w >= _p->_lbfsize && (char)_c != '\n'))
  return (*_p->_p++ = _c);
 else
  return (__swbuf(_c, _p));
}
void flockfile(FILE *);
int ftrylockfile(FILE *);
void funlockfile(FILE *);
int getc_unlocked(FILE *);
int getchar_unlocked(void);
int putc_unlocked(int, FILE *);
int putchar_unlocked(int);
int getw(FILE *);
int putw(int, FILE *);
__attribute__((__availability__(swift, unavailable, message="Use mkstemp(3) instead.")))
__attribute__((__deprecated__("This function is provided for compatibility reasons only.  Due to security concerns inherent in the design of tempnam(3), it is highly recommended that you use mkstemp(3) instead.")))
char *tempnam(const char *__dir, const char *__prefix) __asm("_" "tempnam" );
typedef __darwin_off_t off_t;
int fseeko(FILE * __stream, off_t __offset, int __whence);
off_t ftello(FILE * __stream);
int snprintf(char * restrict __str, size_t __size, const char * restrict __format, ...) __attribute__((__format__ (__printf__, 3, 4)));
int vfscanf(FILE * restrict __stream, const char * restrict __format, va_list) __attribute__((__format__ (__scanf__, 2, 0)));
int vscanf(const char * restrict __format, va_list) __attribute__((__format__ (__scanf__, 1, 0)));
int vsnprintf(char * restrict __str, size_t __size, const char * restrict __format, va_list) __attribute__((__format__ (__printf__, 3, 0)));
int vsscanf(const char * restrict __str, const char * restrict __format, va_list) __attribute__((__format__ (__scanf__, 2, 0)));
typedef __darwin_ssize_t ssize_t;
int dprintf(int, const char * restrict, ...) __attribute__((__format__ (__printf__, 2, 3))) __attribute__((availability(macosx,introduced=10.7)));
int vdprintf(int, const char * restrict, va_list) __attribute__((__format__ (__printf__, 2, 0))) __attribute__((availability(macosx,introduced=10.7)));
ssize_t getdelim(char ** restrict __linep, size_t * restrict __linecapp, int __delimiter, FILE * restrict __stream) __attribute__((availability(macosx,introduced=10.7)));
ssize_t getline(char ** restrict __linep, size_t * restrict __linecapp, FILE * restrict __stream) __attribute__((availability(macosx,introduced=10.7)));
FILE *fmemopen(void * restrict __buf, size_t __size, const char * restrict __mode) __attribute__((availability(macos,introduced=10.13))) __attribute__((availability(ios,introduced=11.0))) __attribute__((availability(tvos,introduced=11.0))) __attribute__((availability(watchos,introduced=4.0)));
FILE *open_memstream(char **__bufp, size_t *__sizep) __attribute__((availability(macos,introduced=10.13))) __attribute__((availability(ios,introduced=11.0))) __attribute__((availability(tvos,introduced=11.0))) __attribute__((availability(watchos,introduced=4.0)));
extern const int sys_nerr;
extern const char *const sys_errlist[];
int asprintf(char ** restrict, const char * restrict, ...) __attribute__((__format__ (__printf__, 2, 3)));
char *ctermid_r(char *);
char *fgetln(FILE *, size_t *);
const char *fmtcheck(const char *, const char *) __attribute__((format_arg(2)));
int fpurge(FILE *);
void setbuffer(FILE *, char *, int);
int setlinebuf(FILE *);
int vasprintf(char ** restrict, const char * restrict, va_list) __attribute__((__format__ (__printf__, 2, 0)));
FILE *funopen(const void *,
     int (* _Nullable)(void *, char *, int),
     int (* _Nullable)(void *, const char *, int),
     fpos_t (* _Nullable)(void *, fpos_t, int),
     int (* _Nullable)(void *));
extern int __sprintf_chk (char * restrict, int, size_t,
     const char * restrict, ...);
extern int __snprintf_chk (char * restrict, size_t, int, size_t,
      const char * restrict, ...);
extern int __vsprintf_chk (char * restrict, int, size_t,
      const char * restrict, va_list);
extern int __vsnprintf_chk (char * restrict, size_t, int, size_t,
       const char * restrict, va_list);
typedef enum SparkStatus
{
    SPARK_STATUS_OK = 0,
    SPARK_STATUS_INVALID_ARGUMENT,
    SPARK_STATUS_CAPACITY_EXCEEDED,
    SPARK_STATUS_NOT_FOUND,
    SPARK_STATUS_IO_ERROR,
    SPARK_STATUS_PARSE_ERROR,
    SPARK_STATUS_SCHEMA_ERROR,
    SPARK_STATUS_HASH_MISMATCH,
    SPARK_STATUS_MODULE_NOT_VALIDATED,
    SPARK_STATUS_VALIDATION_FAILED,
    SPARK_STATUS_ABI_MISMATCH,
    SPARK_STATUS_TARGET_MISMATCH,
    SPARK_STATUS_COMPILER_ERROR,
    SPARK_STATUS_DRIVER_LOAD_ERROR,
    SPARK_STATUS_ROUTE_NOT_FOUND,
    SPARK_STATUS_BUSY,
    SPARK_STATUS_DUPLICATE,
    SPARK_STATUS_INTERNAL_ERROR,
    SPARK_STATUS_PENDING,
    SPARK_STATUS_UNSUPPORTED
} SparkStatus;
const char *SparkStatusToString(SparkStatus status);
static inline SparkStatus SparkReportError(char *error_buffer, uint32_t error_buffer_bytes, SparkStatus status, const char *message)
{
 if ( error_buffer != 0 && error_buffer_bytes != 0u )
 {
  if ( message == 0 )
   error_buffer[0] = '\0';
  else
   (void)__builtin___snprintf_chk (error_buffer, error_buffer_bytes, 0, __builtin_object_size (error_buffer, 2 > 1 ? 1 : 0), "%s", message);
 }
 return status;
}
static inline uint32_t SparkCeilDivU32(uint32_t numerator, uint32_t denominator)
{
 return denominator == 0u ? 0u : (numerator + denominator - 1u) / denominator;
}
static inline uint64_t SparkCeilDivU64(uint64_t numerator, uint64_t denominator)
{
 return denominator == 0u ? 0u : (numerator + denominator - 1u) / denominator;
}
static inline uint64_t SparkHashBytes(uint64_t hash,const void *data,uint32_t data_bytes)
{
 const uint8_t *bytes = (const uint8_t *)data;
 uint32_t byte_index;
 for (byte_index = 0u; byte_index < data_bytes; ++byte_index)
 {
  hash ^= bytes[byte_index];
  hash *= 1099511628211u;
 }
 return hash;
}
static inline uint64_t SparkNormalizeRequestIdBase(
    uint64_t default_base,
    uint64_t request_id_base)
{
    if (request_id_base == 0u)
    {
        return default_base;
    }
    return request_id_base;
}
static inline uint32_t SparkRoundDownToMultipleU32(
    uint32_t value,
    uint32_t multiple)
{
    if (multiple == 0u)
    {
        return 0u;
    }
    return value - (value % multiple);
}
static inline uint32_t SparkNormalizeMaxContextTokens(
    uint32_t default_max,
    uint32_t max_context_tokens)
{
    if (max_context_tokens == 0u)
    {
        return default_max;
    }
    return max_context_tokens;
}
typedef struct SparkWorkTransactionIdentity
{
    uint64_t control_generation;
    uint64_t transaction_id;
    uint64_t dispatch_generation;
    uint64_t request_generation;
    uint64_t step_generation;
    uint32_t step_chunk_index;
    uint32_t step_chunk_count;
    union
    {
        uint32_t phase;
        uint32_t transaction_phase;
    };
    uint32_t reserved0;
} SparkWorkTransactionIdentity;
typedef struct SparkWorkTransactionAcknowledgement
{
    uint32_t magic;
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t status;
    uint32_t transaction_state;
    uint32_t reserved0;
    SparkWorkTransactionIdentity identity;
    uint64_t packet_fingerprint;
    uint64_t reserved1;
} SparkWorkTransactionAcknowledgement;
typedef struct SparkWorkTransactionEntry
{
    SparkWorkTransactionIdentity identity;
    union
    {
        uint64_t payload_fingerprint;
        uint64_t packet_hash;
    };
    uint64_t last_observed_epoch;
    uint32_t state;
    uint32_t terminal_status;
} SparkWorkTransactionEntry;
typedef struct SparkWorkTransactionLedger
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t entry_capacity;
    uint32_t entry_count;
    uint32_t active_entry_count;
    uint32_t terminal_entry_count;
    uint32_t tombstone_entry_count;
    uint32_t reserved0;
    uint64_t epoch;
    uint64_t active_control_generation;
    SparkWorkTransactionEntry *entries;
} SparkWorkTransactionLedger;
typedef struct SparkWorkTransactionCreditDomain
{
    uint32_t capacity;
    uint32_t in_use;
} SparkWorkTransactionCreditDomain;
typedef struct SparkWorkTransactionCreditLedger
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    SparkWorkTransactionCreditDomain domains[
        4u];
} SparkWorkTransactionCreditLedger;
uint32_t SparkWorkTransactionPhaseIsValid(
    uint32_t phase);
uint32_t SparkWorkTransactionStateIsTerminal(
    uint32_t state);
uint64_t SparkWorkTransactionFingerprintBytes(
    const void *data,
    uint32_t data_bytes);
uint32_t SparkWorkTransactionIdentitiesMatch(
    const SparkWorkTransactionIdentity *left,
    const SparkWorkTransactionIdentity *right);
SparkStatus SparkWorkTransactionValidateIdentity(
    const SparkWorkTransactionIdentity *identity);
SparkStatus SparkWorkTransactionInitializeLedger(
    SparkWorkTransactionLedger *ledger,
    SparkWorkTransactionEntry *entries,
    uint32_t entry_capacity);
SparkStatus SparkWorkTransactionObserve(
    SparkWorkTransactionLedger *ledger,
    const SparkWorkTransactionIdentity *identity,
    uint64_t payload_fingerprint,
    SparkWorkTransactionEntry **entry_out,
    uint32_t *observation_out);
SparkStatus SparkWorkTransactionTransition(
    SparkWorkTransactionLedger *ledger,
    const SparkWorkTransactionIdentity *identity,
    uint32_t target_state,
    SparkStatus terminal_status);
SparkStatus SparkWorkTransactionFind(
    SparkWorkTransactionLedger *ledger,
    const SparkWorkTransactionIdentity *identity,
    SparkWorkTransactionEntry **entry_out);
void SparkWorkTransactionInitializeAcknowledgement(
    SparkWorkTransactionAcknowledgement *acknowledgement,
    const SparkWorkTransactionIdentity *identity,
    uint64_t packet_fingerprint,
    uint32_t transaction_state,
    SparkStatus status);
SparkStatus SparkWorkTransactionValidateAcknowledgement(
    const SparkWorkTransactionAcknowledgement *acknowledgement,
    const SparkWorkTransactionIdentity *expected_identity,
    uint64_t expected_packet_fingerprint);
SparkStatus SparkWorkTransactionInitializeCreditLedger(
    SparkWorkTransactionCreditLedger *ledger,
    const uint32_t capacities[4u]);
SparkStatus SparkWorkTransactionAcquireCredits(
    SparkWorkTransactionCreditLedger *ledger,
    uint32_t domain,
    uint32_t credit_count);
SparkStatus SparkWorkTransactionReleaseCredits(
    SparkWorkTransactionCreditLedger *ledger,
    uint32_t domain,
    uint32_t credit_count);
uint32_t SparkWorkTransactionAvailableCredits(
    const SparkWorkTransactionCreditLedger *ledger,
    uint32_t domain);
typedef struct SparkRuntimeTransactionRequest
{
    uint32_t descriptor_bytes;
    uint32_t participant_count;
    SparkWorkTransactionIdentity identity;
    uint64_t request_id;
    uint64_t sequence_id;
    uint64_t sequence_position;
    const void *payload;
    uint32_t payload_bytes;
    uint32_t reserved0;
} SparkRuntimeTransactionRequest;
typedef SparkStatus (*SparkRuntimeParticipantPrepareFunction)(
    void *participant_context,
    const SparkRuntimeTransactionRequest *request);
typedef SparkStatus (*SparkRuntimeParticipantExecuteFunction)(
    void *participant_context,
    const SparkRuntimeTransactionRequest *request);
typedef void (*SparkRuntimeParticipantCommitFunction)(
    void *participant_context,
    const SparkRuntimeTransactionRequest *request);
typedef void (*SparkRuntimeParticipantCancelFunction)(
    void *participant_context,
    const SparkRuntimeTransactionRequest *request,
    SparkStatus reason);
typedef struct SparkRuntimeParticipant
{
    uint32_t descriptor_bytes;
    uint32_t participant_index;
    uint32_t flags;
    uint32_t reserved0;
    uint64_t restart_epoch;
    const char *participant_name;
    void *participant_context;
    SparkRuntimeParticipantPrepareFunction prepare;
    SparkRuntimeParticipantExecuteFunction execute;
    SparkRuntimeParticipantCommitFunction commit;
    SparkRuntimeParticipantCancelFunction cancel;
} SparkRuntimeParticipant;
typedef struct SparkRuntimeFinalEvent
{
    uint32_t magic;
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t state;
    uint64_t event_generation;
    SparkWorkTransactionIdentity identity;
    uint64_t payload_fingerprint;
    uint64_t request_id;
    uint64_t sequence_id;
    uint64_t sequence_position;
    SparkStatus status;
    uint32_t reserved0;
} SparkRuntimeFinalEvent;
typedef struct SparkRuntimeFinalEventQueue
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t capacity;
    uint32_t pending_count;
    uint64_t next_event_generation;
    SparkRuntimeFinalEvent *events;
} SparkRuntimeFinalEventQueue;
typedef struct SparkRuntimeTransmissionSlot
{
    uint32_t state;
    uint32_t slot_index;
    uint64_t slot_generation;
    SparkWorkTransactionIdentity identity;
    uint64_t payload_fingerprint;
    uint64_t last_send_epoch;
    uint32_t retry_count;
    uint32_t reserved0;
} SparkRuntimeTransmissionSlot;
typedef struct SparkRuntimeTransmissionWindow
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t capacity;
    uint32_t in_use_count;
    uint64_t next_slot_generation;
    uint64_t send_epoch;
    SparkRuntimeTransmissionSlot *slots;
} SparkRuntimeTransmissionWindow;
typedef struct SparkRuntimeController
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t participant_count;
    uint32_t reserved0;
    SparkRuntimeParticipant *participants;
    SparkWorkTransactionLedger transaction_ledger;
    SparkWorkTransactionCreditLedger credit_ledger;
    SparkRuntimeFinalEventQueue final_event_queue;
} SparkRuntimeController;
typedef struct SparkRuntimeTransactionResult
{
    uint32_t descriptor_bytes;
    uint32_t replayed;
    uint32_t committed_participant_count;
    uint32_t cancelled_participant_count;
    uint64_t payload_fingerprint;
    uint64_t final_event_generation;
    SparkStatus status;
    uint32_t reserved0;
} SparkRuntimeTransactionResult;
SparkStatus SparkRuntimeInitializeFinalEventQueue(
    SparkRuntimeFinalEventQueue *queue,
    SparkRuntimeFinalEvent *events,
    uint32_t capacity);
SparkStatus SparkRuntimePeekFinalEvent(
    const SparkRuntimeFinalEventQueue *queue,
    const SparkRuntimeFinalEvent **event_out);
SparkStatus SparkRuntimeAcknowledgeFinalEvent(
    SparkRuntimeFinalEventQueue *queue,
    const SparkWorkTransactionIdentity *identity,
    uint64_t event_generation,
    uint64_t payload_fingerprint);
SparkStatus SparkRuntimeAcknowledgeControllerFinalEvent(
    SparkRuntimeController *controller,
    const SparkWorkTransactionIdentity *identity,
    uint64_t event_generation,
    uint64_t payload_fingerprint);
SparkStatus SparkRuntimeInitializeTransmissionWindow(
    SparkRuntimeTransmissionWindow *window,
    SparkRuntimeTransmissionSlot *slots,
    uint32_t capacity);
SparkStatus SparkRuntimeReserveTransmissionSlot(
    SparkRuntimeTransmissionWindow *window,
    const SparkWorkTransactionIdentity *identity,
    uint64_t payload_fingerprint,
    uint32_t *slot_index_out,
    uint64_t *slot_generation_out);
SparkStatus SparkRuntimeMarkTransmissionSent(
    SparkRuntimeTransmissionWindow *window,
    uint32_t slot_index,
    uint64_t slot_generation);
SparkStatus SparkRuntimeAcknowledgeTransmission(
    SparkRuntimeTransmissionWindow *window,
    uint32_t slot_index,
    uint64_t slot_generation,
    const SparkWorkTransactionIdentity *identity,
    uint64_t payload_fingerprint);
SparkStatus SparkRuntimeNextReplayTransmission(
    SparkRuntimeTransmissionWindow *window,
    uint32_t *slot_index_out,
    uint64_t *slot_generation_out);
SparkStatus SparkRuntimeInitializeController(
    SparkRuntimeController *controller,
    SparkRuntimeParticipant *participants,
    uint32_t participant_count,
    SparkWorkTransactionEntry *transaction_entries,
    uint32_t transaction_capacity,
    SparkRuntimeFinalEvent *final_events,
    uint32_t final_event_capacity,
    const uint32_t credit_capacities[
        4u]);
SparkStatus SparkRuntimeValidateTransactionRequest(
    const SparkRuntimeController *controller,
    const SparkRuntimeTransactionRequest *request,
    uint64_t *payload_fingerprint_out);
SparkStatus SparkRuntimeRunTransaction(
    SparkRuntimeController *controller,
    const SparkRuntimeTransactionRequest *request,
    SparkRuntimeTransactionResult *result);

typedef long int ptrdiff_t;
typedef long unsigned int rsize_t;
typedef int wchar_t;
typedef long double max_align_t;
void *memchr(const void *__s, int __c, size_t __n);
int memcmp(const void *__s1, const void *__s2, size_t __n);
void *memcpy(void *__dst, const void *__src, size_t __n);
void *memmove(void *__dst, const void *__src, size_t __len);
void *memset(void *__b, int __c, size_t __len);
char *strcat(char *__s1, const char *__s2);
char *strchr(const char *__s, int __c);
int strcmp(const char *__s1, const char *__s2);
int strcoll(const char *__s1, const char *__s2);
char *strcpy(char *__dst, const char *__src);
size_t strcspn(const char *__s, const char *__charset);
char *strerror(int __errnum) __asm("_" "strerror" );
size_t strlen(const char *__s);
char *strncat(char *__s1, const char *__s2, size_t __n);
int strncmp(const char *__s1, const char *__s2, size_t __n);
char *strncpy(char *__dst, const char *__src, size_t __n);
char *strpbrk(const char *__s, const char *__charset);
char *strrchr(const char *__s, int __c);
size_t strspn(const char *__s, const char *__charset);
char *strstr(const char *__big, const char *__little);
char *strtok(char *__str, const char *__sep);
size_t strxfrm(char *__s1, const char *__s2, size_t __n);
char *strtok_r(char *__str, const char *__sep, char **__lasts);
int strerror_r(int __errnum, char *__strerrbuf, size_t __buflen);
char *strdup(const char *__s1);
void *memccpy(void *__dst, const void *__src, int __c, size_t __n);
char *stpcpy(char *__dst, const char *__src);
char *stpncpy(char *__dst, const char *__src, size_t __n) __attribute__((availability(macosx,introduced=10.7)));
char *strndup(const char *__s1, size_t __n) __attribute__((availability(macosx,introduced=10.7)));
size_t strnlen(const char *__s1, size_t __n) __attribute__((availability(macosx,introduced=10.7)));
char *strsignal(int __sig);
typedef int errno_t;
errno_t memset_s(void *__s, rsize_t __smax, int __c, rsize_t __n) __attribute__((availability(macosx,introduced=10.9)));
void *memmem(const void *__big, size_t __big_len, const void *__little, size_t __little_len) __attribute__((availability(macosx,introduced=10.7)));
void memset_pattern4(void *__b, const void *__pattern4, size_t __len) __attribute__((availability(macosx,introduced=10.5)));
void memset_pattern8(void *__b, const void *__pattern8, size_t __len) __attribute__((availability(macosx,introduced=10.5)));
void memset_pattern16(void *__b, const void *__pattern16, size_t __len) __attribute__((availability(macosx,introduced=10.5)));
char *strcasestr(const char *__big, const char *__little);
char *strnstr(const char *__big, const char *__little, size_t __len);
size_t strlcat(char *__dst, const char *__source, size_t __size);
size_t strlcpy(char *__dst, const char *__source, size_t __size);
void strmode(int __mode, char *__bp);
char *strsep(char **__stringp, const char *__delim);
void swab(const void * restrict, void * restrict, ssize_t);
__attribute__((availability(macosx,introduced=10.12.1))) __attribute__((availability(ios,introduced=10.1)))
__attribute__((availability(tvos,introduced=10.0.1))) __attribute__((availability(watchos,introduced=3.1)))
int timingsafe_bcmp(const void *__b1, const void *__b2, size_t __len);
__attribute__((availability(macosx,introduced=11.0))) __attribute__((availability(ios,introduced=14.0)))
__attribute__((availability(tvos,introduced=14.0))) __attribute__((availability(watchos,introduced=7.0)))
int strsignal_r(int __sig, char *__strsignalbuf, size_t __buflen);
int bcmp(const void *, const void *, size_t) ;
void bcopy(const void *, void *, size_t) ;
void bzero(void *, size_t) ;
char *index(const char *, int) ;
char *rindex(const char *, int) ;
int ffs(int);
int strcasecmp(const char *, const char *);
int strncasecmp(const char *, const char *, size_t);
int ffsl(long) __attribute__((availability(macosx,introduced=10.5)));
int ffsll(long long) __attribute__((availability(macosx,introduced=10.9)));
int fls(int) __attribute__((availability(macosx,introduced=10.5)));
int flsl(long) __attribute__((availability(macosx,introduced=10.5)));
int flsll(long long) __attribute__((availability(macosx,introduced=10.9)));

static SparkStatus SparkRuntimeValidateParticipant(
    const SparkRuntimeParticipant *participant,
    uint32_t expected_participant_index)
{
    if (participant == 0 ||
        participant->descriptor_bytes != ((uint32_t)sizeof(SparkRuntimeParticipant)) ||
        participant->participant_index != expected_participant_index ||
        (participant->flags & ((0x00000001U) | (0x00000002U) | (0x00000004U))) !=
            ((0x00000001U) | (0x00000002U) | (0x00000004U)) ||
        (participant->flags & ~((0x00000001U) | (0x00000002U) | (0x00000004U))) != 0u ||
        participant->restart_epoch == 0u ||
        participant->participant_name == 0 ||
        participant->participant_name[0] == '\0' ||
        participant->prepare == 0 ||
        participant->execute == 0 ||
        participant->commit == 0 ||
        participant->cancel == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}
static SparkStatus SparkRuntimeValidateFinalEventQueue(
    const SparkRuntimeFinalEventQueue *queue)
{
    if (queue == 0 ||
        queue->abi_version != 1u ||
        queue->descriptor_bytes != ((uint32_t)sizeof(SparkRuntimeFinalEventQueue)) ||
        queue->capacity == 0u ||
        queue->pending_count > queue->capacity ||
        queue->next_event_generation == 0u ||
        queue->events == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}
static SparkStatus SparkRuntimeValidateTransmissionWindow(
    const SparkRuntimeTransmissionWindow *window)
{
    if (window == 0 ||
        window->abi_version != 1u ||
        window->descriptor_bytes != ((uint32_t)sizeof(SparkRuntimeTransmissionWindow)) ||
        window->capacity == 0u ||
        window->in_use_count > window->capacity ||
        window->next_slot_generation == 0u ||
        window->slots == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}
static SparkRuntimeFinalEvent *SparkRuntimeFindReusableFinalEvent(
    SparkRuntimeFinalEventQueue *queue)
{
    SparkRuntimeFinalEvent *oldest_acknowledged;
    uint64_t oldest_generation;
    uint32_t event_index;
    oldest_acknowledged = 0;
    oldest_generation = 18446744073709551615ULL;
    for (event_index = 0u; event_index < queue->capacity; ++event_index)
    {
        SparkRuntimeFinalEvent *event;
        event = &queue->events[event_index];
        if (event->state == 0u)
        {
            return event;
        }
        if (event->state == 2u &&
            event->event_generation < oldest_generation)
        {
            oldest_acknowledged = event;
            oldest_generation = event->event_generation;
        }
    }
    return oldest_acknowledged;
}
static SparkStatus SparkRuntimeEnqueueFinalEvent(
    SparkRuntimeFinalEventQueue *queue,
    const SparkRuntimeTransactionRequest *request,
    uint64_t payload_fingerprint,
    SparkStatus status,
    uint64_t *event_generation_out)
{
    SparkRuntimeFinalEvent *event;
    if (event_generation_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *event_generation_out = 0u;
    if (SparkRuntimeValidateFinalEventQueue(queue) != SPARK_STATUS_OK ||
        request == 0 ||
        payload_fingerprint == 0u ||
        queue->pending_count >= queue->capacity)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    event = SparkRuntimeFindReusableFinalEvent(queue);
    if (event == 0)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    __builtin___memset_chk (event, 0,sizeof(*event), __builtin_object_size (event, 0));
    event->magic = (0x544E5645U);
    event->abi_version = 1u;
    event->descriptor_bytes = ((uint32_t)sizeof(SparkRuntimeFinalEvent));
    event->state = 1u;
    event->event_generation = queue->next_event_generation;
    queue->next_event_generation += 1u;
    if (queue->next_event_generation == 0u)
    {
        queue->next_event_generation = 1u;
    }
    event->identity = request->identity;
    event->payload_fingerprint = payload_fingerprint;
    event->request_id = request->request_id;
    event->sequence_id = request->sequence_id;
    event->sequence_position = request->sequence_position;
    event->status = status;
    queue->pending_count += 1u;
    *event_generation_out = event->event_generation;
    return SPARK_STATUS_OK;
}
static SparkStatus SparkRuntimeFindFinalEvent(
    const SparkRuntimeFinalEventQueue *queue,
    const SparkWorkTransactionIdentity *identity,
    uint64_t payload_fingerprint,
    const SparkRuntimeFinalEvent **event_out)
{
    uint32_t event_index;
    if (event_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *event_out = 0;
    if (SparkRuntimeValidateFinalEventQueue(queue) != SPARK_STATUS_OK ||
        SparkWorkTransactionValidateIdentity(identity) != SPARK_STATUS_OK ||
        payload_fingerprint == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    for (event_index = 0u; event_index < queue->capacity; ++event_index)
    {
        const SparkRuntimeFinalEvent *event;
        event = &queue->events[event_index];
        if (event->state != 0u &&
            SparkWorkTransactionIdentitiesMatch(
                &event->identity,
                identity) != 0u &&
            event->payload_fingerprint == payload_fingerprint)
        {
            *event_out = event;
            return SPARK_STATUS_OK;
        }
    }
    return SPARK_STATUS_NOT_FOUND;
}
static SparkStatus SparkRuntimeAcquireCredit(
    SparkRuntimeController *controller,
    uint32_t domain)
{
    SparkStatus status;
    status = SparkWorkTransactionAcquireCredits(
        &controller->credit_ledger,
        domain,
        1u);
    return status == SPARK_STATUS_CAPACITY_EXCEEDED ?
        SPARK_STATUS_BUSY : status;
}
static void SparkRuntimeReleaseCreditIfHeld(
    SparkRuntimeController *controller,
    uint32_t domain,
    uint32_t *held)
{
    if (*held != 0u)
    {
        (void)SparkWorkTransactionReleaseCredits(
            &controller->credit_ledger,
            domain,
            1u);
        *held = 0u;
    }
}
static void SparkRuntimeCancelPreparedParticipants(
    SparkRuntimeController *controller,
    const SparkRuntimeTransactionRequest *request,
    uint32_t prepared_participant_count,
    SparkStatus reason,
    SparkRuntimeTransactionResult *result)
{
    while (prepared_participant_count != 0u)
    {
        SparkRuntimeParticipant *participant;
        prepared_participant_count -= 1u;
        participant = &controller->participants[prepared_participant_count];
        participant->cancel(
            participant->participant_context,
            request,
            reason);
        result->cancelled_participant_count += 1u;
    }
}
SparkStatus SparkRuntimeInitializeFinalEventQueue(
    SparkRuntimeFinalEventQueue *queue,
    SparkRuntimeFinalEvent *events,
    uint32_t capacity)
{
    if (queue == 0 || events == 0 || capacity == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    __builtin___memset_chk (queue, 0,sizeof(*queue), __builtin_object_size (queue, 0));
    __builtin___memset_chk (events, 0,(size_t)capacity * sizeof(events[0u]), __builtin_object_size (events, 0));
    queue->abi_version = 1u;
    queue->descriptor_bytes = ((uint32_t)sizeof(SparkRuntimeFinalEventQueue));
    queue->capacity = capacity;
    queue->next_event_generation = 1u;
    queue->events = events;
    return SPARK_STATUS_OK;
}
SparkStatus SparkRuntimePeekFinalEvent(
    const SparkRuntimeFinalEventQueue *queue,
    const SparkRuntimeFinalEvent **event_out)
{
    const SparkRuntimeFinalEvent *oldest_event;
    uint64_t oldest_generation;
    uint32_t event_index;
    if (event_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *event_out = 0;
    if (SparkRuntimeValidateFinalEventQueue(queue) != SPARK_STATUS_OK)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    oldest_event = 0;
    oldest_generation = 18446744073709551615ULL;
    for (event_index = 0u; event_index < queue->capacity; ++event_index)
    {
        const SparkRuntimeFinalEvent *event;
        event = &queue->events[event_index];
        if (event->state == 1u &&
            event->event_generation < oldest_generation)
        {
            oldest_event = event;
            oldest_generation = event->event_generation;
        }
    }
    if (oldest_event == 0)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    *event_out = oldest_event;
    return SPARK_STATUS_OK;
}
SparkStatus SparkRuntimeAcknowledgeFinalEvent(
    SparkRuntimeFinalEventQueue *queue,
    const SparkWorkTransactionIdentity *identity,
    uint64_t event_generation,
    uint64_t payload_fingerprint)
{
    uint32_t event_index;
    if (SparkRuntimeValidateFinalEventQueue(queue) != SPARK_STATUS_OK ||
        SparkWorkTransactionValidateIdentity(identity) != SPARK_STATUS_OK ||
        event_generation == 0u ||
        payload_fingerprint == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    for (event_index = 0u; event_index < queue->capacity; ++event_index)
    {
        SparkRuntimeFinalEvent *event;
        event = &queue->events[event_index];
        if (event->event_generation == event_generation &&
            SparkWorkTransactionIdentitiesMatch(
                &event->identity,
                identity) != 0u)
        {
            if (event->payload_fingerprint != payload_fingerprint)
            {
                return SPARK_STATUS_VALIDATION_FAILED;
            }
            if (event->state == 2u)
            {
                return SPARK_STATUS_DUPLICATE;
            }
            if (event->state != 1u ||
                queue->pending_count == 0u)
            {
                return SPARK_STATUS_VALIDATION_FAILED;
            }
            event->state = 2u;
            queue->pending_count -= 1u;
            return SPARK_STATUS_OK;
        }
    }
    return SPARK_STATUS_NOT_FOUND;
}
SparkStatus SparkRuntimeAcknowledgeControllerFinalEvent(
    SparkRuntimeController *controller,
    const SparkWorkTransactionIdentity *identity,
    uint64_t event_generation,
    uint64_t payload_fingerprint)
{
    SparkStatus status;
    if (controller == 0 ||
        controller->abi_version != 1u ||
        controller->descriptor_bytes != ((uint32_t)sizeof(SparkRuntimeController)))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkRuntimeAcknowledgeFinalEvent(
        &controller->final_event_queue,
        identity,
        event_generation,
        payload_fingerprint);
    if (status == SPARK_STATUS_DUPLICATE)
    {
        return status;
    }
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkWorkTransactionReleaseCredits(
        &controller->credit_ledger,
        3u,
        1u);
}
SparkStatus SparkRuntimeInitializeTransmissionWindow(
    SparkRuntimeTransmissionWindow *window,
    SparkRuntimeTransmissionSlot *slots,
    uint32_t capacity)
{
    uint32_t slot_index;
    if (window == 0 || slots == 0 || capacity == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    __builtin___memset_chk (window, 0,sizeof(*window), __builtin_object_size (window, 0));
    __builtin___memset_chk (slots, 0,(size_t)capacity * sizeof(slots[0u]), __builtin_object_size (slots, 0));
    window->abi_version = 1u;
    window->descriptor_bytes = ((uint32_t)sizeof(SparkRuntimeTransmissionWindow));
    window->capacity = capacity;
    window->next_slot_generation = 1u;
    window->slots = slots;
    for (slot_index = 0u; slot_index < capacity; ++slot_index)
    {
        slots[slot_index].slot_index = slot_index;
    }
    return SPARK_STATUS_OK;
}
SparkStatus SparkRuntimeReserveTransmissionSlot(
    SparkRuntimeTransmissionWindow *window,
    const SparkWorkTransactionIdentity *identity,
    uint64_t payload_fingerprint,
    uint32_t *slot_index_out,
    uint64_t *slot_generation_out)
{
    uint32_t slot_index;
    if (slot_index_out == 0 || slot_generation_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *slot_index_out = 4294967295U;
    *slot_generation_out = 0u;
    if (SparkRuntimeValidateTransmissionWindow(window) != SPARK_STATUS_OK ||
        SparkWorkTransactionValidateIdentity(identity) != SPARK_STATUS_OK ||
        payload_fingerprint == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (window->in_use_count >= window->capacity)
    {
        return SPARK_STATUS_BUSY;
    }
    for (slot_index = 0u; slot_index < window->capacity; ++slot_index)
    {
        SparkRuntimeTransmissionSlot *slot;
        slot = &window->slots[slot_index];
        if (slot->state == 0u ||
            slot->state == 3u)
        {
            uint64_t generation;
            generation = window->next_slot_generation;
            window->next_slot_generation += 1u;
            if (window->next_slot_generation == 0u)
            {
                window->next_slot_generation = 1u;
            }
            __builtin___memset_chk (slot, 0,sizeof(*slot), __builtin_object_size (slot, 0));
            slot->state = 1u;
            slot->slot_index = slot_index;
            slot->slot_generation = generation;
            slot->identity = *identity;
            slot->payload_fingerprint = payload_fingerprint;
            window->in_use_count += 1u;
            *slot_index_out = slot_index;
            *slot_generation_out = generation;
            return SPARK_STATUS_OK;
        }
    }
    return SPARK_STATUS_INTERNAL_ERROR;
}
SparkStatus SparkRuntimeMarkTransmissionSent(
    SparkRuntimeTransmissionWindow *window,
    uint32_t slot_index,
    uint64_t slot_generation)
{
    SparkRuntimeTransmissionSlot *slot;
    if (SparkRuntimeValidateTransmissionWindow(window) != SPARK_STATUS_OK ||
        slot_index >= window->capacity ||
        slot_generation == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    slot = &window->slots[slot_index];
    if (slot->slot_generation != slot_generation)
    {
        return SPARK_STATUS_VALIDATION_FAILED;
    }
    if (slot->state == 2u)
    {
        return SPARK_STATUS_DUPLICATE;
    }
    if (slot->state != 1u)
    {
        return SPARK_STATUS_VALIDATION_FAILED;
    }
    window->send_epoch += 1u;
    if (window->send_epoch == 0u)
    {
        window->send_epoch = 1u;
    }
    slot->state = 2u;
    slot->last_send_epoch = window->send_epoch;
    return SPARK_STATUS_OK;
}
SparkStatus SparkRuntimeAcknowledgeTransmission(
    SparkRuntimeTransmissionWindow *window,
    uint32_t slot_index,
    uint64_t slot_generation,
    const SparkWorkTransactionIdentity *identity,
    uint64_t payload_fingerprint)
{
    SparkRuntimeTransmissionSlot *slot;
    if (SparkRuntimeValidateTransmissionWindow(window) != SPARK_STATUS_OK ||
        slot_index >= window->capacity ||
        slot_generation == 0u ||
        SparkWorkTransactionValidateIdentity(identity) != SPARK_STATUS_OK ||
        payload_fingerprint == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    slot = &window->slots[slot_index];
    if (slot->slot_generation != slot_generation ||
        SparkWorkTransactionIdentitiesMatch(&slot->identity,identity) == 0u ||
        slot->payload_fingerprint != payload_fingerprint)
    {
        return SPARK_STATUS_VALIDATION_FAILED;
    }
    if (slot->state == 3u)
    {
        return SPARK_STATUS_DUPLICATE;
    }
    if (slot->state != 2u ||
        window->in_use_count == 0u)
    {
        return SPARK_STATUS_VALIDATION_FAILED;
    }
    slot->state = 3u;
    window->in_use_count -= 1u;
    return SPARK_STATUS_OK;
}
SparkStatus SparkRuntimeNextReplayTransmission(
    SparkRuntimeTransmissionWindow *window,
    uint32_t *slot_index_out,
    uint64_t *slot_generation_out)
{
    SparkRuntimeTransmissionSlot *oldest_slot;
    uint64_t oldest_epoch;
    uint32_t slot_index;
    if (slot_index_out == 0 || slot_generation_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *slot_index_out = 4294967295U;
    *slot_generation_out = 0u;
    if (SparkRuntimeValidateTransmissionWindow(window) != SPARK_STATUS_OK)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    oldest_slot = 0;
    oldest_epoch = 18446744073709551615ULL;
    for (slot_index = 0u; slot_index < window->capacity; ++slot_index)
    {
        SparkRuntimeTransmissionSlot *slot;
        slot = &window->slots[slot_index];
        if (slot->state == 2u &&
            slot->last_send_epoch < oldest_epoch)
        {
            oldest_slot = slot;
            oldest_epoch = slot->last_send_epoch;
        }
    }
    if (oldest_slot == 0)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    window->send_epoch += 1u;
    if (window->send_epoch == 0u)
    {
        window->send_epoch = 1u;
    }
    oldest_slot->last_send_epoch = window->send_epoch;
    oldest_slot->retry_count += 1u;
    if (oldest_slot->retry_count == 0u)
    {
        oldest_slot->retry_count = 4294967295U;
    }
    *slot_index_out = oldest_slot->slot_index;
    *slot_generation_out = oldest_slot->slot_generation;
    return SPARK_STATUS_OK;
}
SparkStatus SparkRuntimeInitializeController(
    SparkRuntimeController *controller,
    SparkRuntimeParticipant *participants,
    uint32_t participant_count,
    SparkWorkTransactionEntry *transaction_entries,
    uint32_t transaction_capacity,
    SparkRuntimeFinalEvent *final_events,
    uint32_t final_event_capacity,
    const uint32_t credit_capacities[
        4u])
{
    SparkStatus status;
    uint32_t participant_index;
    if (controller == 0 || participants == 0 ||
        participant_count == 0u ||
        participant_count > 32u ||
        transaction_entries == 0 || transaction_capacity == 0u ||
        final_events == 0 || final_event_capacity == 0u ||
        credit_capacities == 0 ||
        credit_capacities[
            3u] !=
            final_event_capacity)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    for (participant_index = 0u;
         participant_index < participant_count;
         ++participant_index)
    {
        status = SparkRuntimeValidateParticipant(
            &participants[participant_index],
            participant_index);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    __builtin___memset_chk (controller, 0,sizeof(*controller), __builtin_object_size (controller, 0));
    controller->abi_version = 1u;
    controller->descriptor_bytes = ((uint32_t)sizeof(SparkRuntimeController));
    controller->participant_count = participant_count;
    controller->participants = participants;
    status = SparkWorkTransactionInitializeLedger(
        &controller->transaction_ledger,
        transaction_entries,
        transaction_capacity);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkWorkTransactionInitializeCreditLedger(
        &controller->credit_ledger,
        credit_capacities);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkRuntimeInitializeFinalEventQueue(
        &controller->final_event_queue,
        final_events,
        final_event_capacity);
}
SparkStatus SparkRuntimeValidateTransactionRequest(
    const SparkRuntimeController *controller,
    const SparkRuntimeTransactionRequest *request,
    uint64_t *payload_fingerprint_out)
{
    uint64_t payload_fingerprint;
    if (payload_fingerprint_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *payload_fingerprint_out = 0u;
    if (controller == 0 ||
        controller->abi_version != 1u ||
        controller->descriptor_bytes != ((uint32_t)sizeof(SparkRuntimeController)) ||
        controller->participant_count == 0u ||
        controller->participants == 0 ||
        request == 0 ||
        request->descriptor_bytes != ((uint32_t)sizeof(SparkRuntimeTransactionRequest)) ||
        request->participant_count != controller->participant_count ||
        SparkWorkTransactionValidateIdentity(&request->identity) !=
            SPARK_STATUS_OK ||
        request->request_id == 0u ||
        request->sequence_id == 0u ||
        request->payload == 0 ||
        request->payload_bytes == 0u ||
        request->reserved0 != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    {
        uint32_t participant_index;
        for (participant_index = 0u;
             participant_index < controller->participant_count;
             ++participant_index)
        {
            if (request->identity.control_generation <
                controller->participants[participant_index].restart_epoch)
            {
                return SPARK_STATUS_VALIDATION_FAILED;
            }
        }
    }
    payload_fingerprint = SparkWorkTransactionFingerprintBytes(
        request->payload,
        request->payload_bytes);
    if (payload_fingerprint == 0u)
    {
        return SPARK_STATUS_VALIDATION_FAILED;
    }
    *payload_fingerprint_out = payload_fingerprint;
    return SPARK_STATUS_OK;
}
SparkStatus SparkRuntimeRunTransaction(
    SparkRuntimeController *controller,
    const SparkRuntimeTransactionRequest *request,
    SparkRuntimeTransactionResult *result)
{
    SparkWorkTransactionEntry *transaction_entry;
    const SparkRuntimeFinalEvent *existing_event;
    uint64_t payload_fingerprint;
    uint32_t observation;
    uint32_t participant_index;
    uint32_t prepared_participant_count;
    uint32_t transport_credit_held;
    uint32_t reservation_credit_held;
    uint32_t execution_credit_held;
    uint32_t completion_credit_held;
    uint32_t terminal_state;
    SparkStatus status;
    if (result == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    __builtin___memset_chk (result, 0,sizeof(*result), __builtin_object_size (result, 0));
    result->descriptor_bytes = ((uint32_t)sizeof(SparkRuntimeTransactionResult));
    result->status = SPARK_STATUS_PENDING;
    status = SparkRuntimeValidateTransactionRequest(
        controller,
        request,
        &payload_fingerprint);
    if (status != SPARK_STATUS_OK)
    {
        result->status = status;
        return status;
    }
    result->payload_fingerprint = payload_fingerprint;
    status = SparkWorkTransactionObserve(
        &controller->transaction_ledger,
        &request->identity,
        payload_fingerprint,
        &transaction_entry,
        &observation);
    if (status != SPARK_STATUS_OK)
    {
        result->status = status;
        return status;
    }
    if (observation != 1u)
    {
        result->replayed = 1u;
        if (observation == 2u)
        {
            result->status = SPARK_STATUS_BUSY;
            return SPARK_STATUS_BUSY;
        }
        result->status = (SparkStatus)transaction_entry->terminal_status;
        if (SparkRuntimeFindFinalEvent(
                &controller->final_event_queue,
                &request->identity,
                payload_fingerprint,
                &existing_event) == SPARK_STATUS_OK)
        {
            result->final_event_generation =
                existing_event->event_generation;
        }
        return SPARK_STATUS_DUPLICATE;
    }
    prepared_participant_count = 0u;
    transport_credit_held = 0u;
    reservation_credit_held = 0u;
    execution_credit_held = 0u;
    completion_credit_held = 0u;
    terminal_state = 6u;
    status = SparkRuntimeAcquireCredit(
        controller,
        0u);
    if (status != SPARK_STATUS_OK)
    {
        terminal_state = 5u;
        goto fail_transaction;
    }
    transport_credit_held = 1u;
    status = SparkRuntimeAcquireCredit(
        controller,
        1u);
    if (status != SPARK_STATUS_OK)
    {
        terminal_state = 5u;
        goto fail_transaction;
    }
    reservation_credit_held = 1u;
    status = SparkRuntimeAcquireCredit(
        controller,
        3u);
    if (status != SPARK_STATUS_OK)
    {
        terminal_state = 5u;
        goto fail_transaction;
    }
    completion_credit_held = 1u;
    for (participant_index = 0u;
         participant_index < controller->participant_count;
         ++participant_index)
    {
        SparkRuntimeParticipant *participant;
        participant = &controller->participants[participant_index];
        status = participant->prepare(
            participant->participant_context,
            request);
        if (status != SPARK_STATUS_OK)
        {
            goto fail_transaction;
        }
        prepared_participant_count += 1u;
    }
    status = SparkWorkTransactionTransition(
        &controller->transaction_ledger,
        &request->identity,
        2u,
        SPARK_STATUS_PENDING);
    if (status != SPARK_STATUS_OK)
    {
        goto fail_transaction;
    }
    status = SparkRuntimeAcquireCredit(
        controller,
        2u);
    if (status != SPARK_STATUS_OK)
    {
        terminal_state = 5u;
        goto fail_transaction;
    }
    execution_credit_held = 1u;
    status = SparkWorkTransactionTransition(
        &controller->transaction_ledger,
        &request->identity,
        3u,
        SPARK_STATUS_PENDING);
    if (status != SPARK_STATUS_OK)
    {
        goto fail_transaction;
    }
    for (participant_index = 0u;
         participant_index < controller->participant_count;
         ++participant_index)
    {
        SparkRuntimeParticipant *participant;
        participant = &controller->participants[participant_index];
        status = participant->execute(
            participant->participant_context,
            request);
        if (status != SPARK_STATUS_OK)
        {
            goto fail_transaction;
        }
    }
    for (participant_index = 0u;
         participant_index < controller->participant_count;
         ++participant_index)
    {
        SparkRuntimeParticipant *participant;
        participant = &controller->participants[participant_index];
        participant->commit(
            participant->participant_context,
            request);
        result->committed_participant_count += 1u;
    }
    status = SparkWorkTransactionTransition(
        &controller->transaction_ledger,
        &request->identity,
        4u,
        SPARK_STATUS_OK);
    if (status != SPARK_STATUS_OK)
    {
        result->status = SPARK_STATUS_INTERNAL_ERROR;
        SparkRuntimeReleaseCreditIfHeld(
            controller,
            2u,
            &execution_credit_held);
        SparkRuntimeReleaseCreditIfHeld(
            controller,
            1u,
            &reservation_credit_held);
        SparkRuntimeReleaseCreditIfHeld(
            controller,
            0u,
            &transport_credit_held);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    status = SparkRuntimeEnqueueFinalEvent(
        &controller->final_event_queue,
        request,
        payload_fingerprint,
        SPARK_STATUS_OK,
        &result->final_event_generation);
    if (status != SPARK_STATUS_OK)
    {
        result->status = SPARK_STATUS_INTERNAL_ERROR;
        SparkRuntimeReleaseCreditIfHeld(
            controller,
            2u,
            &execution_credit_held);
        SparkRuntimeReleaseCreditIfHeld(
            controller,
            1u,
            &reservation_credit_held);
        SparkRuntimeReleaseCreditIfHeld(
            controller,
            0u,
            &transport_credit_held);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    SparkRuntimeReleaseCreditIfHeld(
        controller,
        2u,
        &execution_credit_held);
    SparkRuntimeReleaseCreditIfHeld(
        controller,
        1u,
        &reservation_credit_held);
    SparkRuntimeReleaseCreditIfHeld(
        controller,
        0u,
        &transport_credit_held);
    result->status = SPARK_STATUS_OK;
    return SPARK_STATUS_OK;
fail_transaction:
    SparkRuntimeCancelPreparedParticipants(
        controller,
        request,
        prepared_participant_count,
        status,
        result);
    if (SparkWorkTransactionTransition(
            &controller->transaction_ledger,
            &request->identity,
            terminal_state,
            status) != SPARK_STATUS_OK)
    {
        status = SPARK_STATUS_INTERNAL_ERROR;
    }
    SparkRuntimeReleaseCreditIfHeld(
        controller,
        2u,
        &execution_credit_held);
    SparkRuntimeReleaseCreditIfHeld(
        controller,
        3u,
        &completion_credit_held);
    SparkRuntimeReleaseCreditIfHeld(
        controller,
        1u,
        &reservation_credit_held);
    SparkRuntimeReleaseCreditIfHeld(
        controller,
        0u,
        &transport_credit_held);
    result->status = status;
    return status;
}
