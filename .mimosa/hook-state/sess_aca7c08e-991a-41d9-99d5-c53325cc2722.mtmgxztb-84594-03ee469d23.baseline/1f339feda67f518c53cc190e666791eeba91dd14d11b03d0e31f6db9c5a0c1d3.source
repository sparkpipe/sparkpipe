
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <limits.h>
#include <unistd.h>
#include <dirent.h>

#define SPARK_REGISTRAR_MAX_MEMBERS 64
#define SPARK_REGISTRAR_MAX_CONNS 160
#define SPARK_REGISTRAR_MSG_MAX 256
#define SPARK_REGISTRAR_MAX_STALE 16
#define SPARK_REGISTRAR_WIRE "SPGA"
#define SPARK_REGISTRAR_GO_WIRE "SPGG"

typedef struct {
    int fd;
    int peer;
    int is_client;
    int sent_announce;
    size_t len;
    char buf[SPARK_REGISTRAR_MSG_MAX];
    long long deadline_ms;
} conn;

typedef struct {
    pid_t pid;
    long long term_deadline_ms;
    int immune_logged;
    int immune;
} stale_entry;

static struct {
    int rank;
    int member_count;
    char hosts[SPARK_REGISTRAR_MAX_MEMBERS][256];
    int ports[SPARK_REGISTRAR_MAX_MEMBERS];
    uint64_t expected_mask;
    int is_expected[SPARK_REGISTRAR_MAX_MEMBERS];
    int leader_rank;
    int port_base;
    long long timeout_ms;
    long long announce_ms;
    long long term_wait_ms;
    char deployment_cwd[4096];
    char daemon_name[256];
} cfg;

static struct {
    uint64_t my_view;
    pid_t my_stale_pid;
    int my_stale_immune;
    uint64_t peer_view[SPARK_REGISTRAR_MAX_MEMBERS];
    int peer_view_seen[SPARK_REGISTRAR_MAX_MEMBERS];
    pid_t peer_stale[SPARK_REGISTRAR_MAX_MEMBERS];
    int peer_stale_immune[SPARK_REGISTRAR_MAX_MEMBERS];
    int peer_stale_seen[SPARK_REGISTRAR_MAX_MEMBERS];
    uint64_t unexpected_seen;
    int go_received;
    int go_from;
    uint64_t go_sent;
    int ready_logged;
    int listen_fd;
    conn conns[SPARK_REGISTRAR_MAX_CONNS];
    stale_entry stale[SPARK_REGISTRAR_MAX_STALE];
    int stale_count;
} st;

static long long now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

static long long g_log_t0 = -1;

static void log_prefix(void) {
    long long now = now_ms();
    if (g_log_t0 < 0) g_log_t0 = now;
    printf("[+%lld.%03lld] ", (now - g_log_t0) / 1000,
           (now - g_log_t0) % 1000);
}

static void log_line(const char *fmt, ...) {
    va_list ap;
    log_prefix();
    printf("registrar ");
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    printf("\n");
    fflush(stdout);
}

static void die_usage(const char *message) {
    fprintf(stderr, "registrar fatal: %s\n", message);
    exit(2);
}

static void format_set(uint64_t mask, char *out, size_t out_size) {
    size_t used = 0;
    used += (size_t)snprintf(out + used, out_size - used, "[");
    int first = 1;
    for (int r = 0; r < SPARK_REGISTRAR_MAX_MEMBERS; r++) {
        if (!(mask & ((uint64_t)1 << r))) continue;
        used += (size_t)snprintf(out + used, out_size - used, "%s%d",
                                 first ? "" : ",", r);
        first = 0;
        if (used + 16 >= out_size) break;
    }
    snprintf(out + used, out_size - used, "]");
}

static int parse_rank_list(const char *csv, uint64_t *mask, int *leader) {
    const char *p = csv;
    while (*p) {
        char *end = NULL;
        long value = strtol(p, &end, 10);
        if (end == p || value < 0 || value >= SPARK_REGISTRAR_MAX_MEMBERS) return -1;
        *mask |= (uint64_t)1 << value;
        if (*leader < 0 || value < *leader) *leader = (int)value;
        p = (*end == ',') ? end + 1 : end;
        if (*p && *end != ',') return -1;
    }
    return 0;
}

static void parse_args(int argc, char **argv) {
    const char *hosts_csv = NULL;
    const char *subset_csv = NULL;
    int rank_set = 0;
    cfg.port_base = 22480;
    cfg.timeout_ms = 120000;
    cfg.announce_ms = 250;
    cfg.term_wait_ms = 15000;
    snprintf(cfg.daemon_name, sizeof(cfg.daemon_name), "sparkpipe_model_residentd");

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--rank") && i + 1 < argc) {
            cfg.rank = atoi(argv[++i]);
            rank_set = 1;
        }
        else if (!strcmp(a, "--hosts") && i + 1 < argc) hosts_csv = argv[++i];
        else if (!strcmp(a, "--port-base") && i + 1 < argc) cfg.port_base = atoi(argv[++i]);
        else if (!strcmp(a, "--timeout-ms") && i + 1 < argc) cfg.timeout_ms = atoll(argv[++i]);
        else if (!strcmp(a, "--announce-ms") && i + 1 < argc) cfg.announce_ms = atoll(argv[++i]);
        else if (!strcmp(a, "--term-wait-ms") && i + 1 < argc) cfg.term_wait_ms = atoll(argv[++i]);
        else if (!strcmp(a, "--deployment-cwd") && i + 1 < argc)
            snprintf(cfg.deployment_cwd, sizeof(cfg.deployment_cwd), "%s", argv[++i]);
        else if (!strcmp(a, "--daemon-name") && i + 1 < argc)
            snprintf(cfg.daemon_name, sizeof(cfg.daemon_name), "%s", argv[++i]);
        else if (!strcmp(a, "--expect-subset") && i + 1 < argc) subset_csv = argv[++i];
        else die_usage(a);
    }
    if (!rank_set) die_usage("--rank is required");
    if (hosts_csv == NULL) die_usage("--hosts is required (the launch table)");
    if (cfg.rank < 0 || cfg.rank >= SPARK_REGISTRAR_MAX_MEMBERS)
        die_usage("--rank out of range");
    if (cfg.port_base <= 0 || cfg.timeout_ms <= 0 || cfg.announce_ms <= 0 ||
        cfg.term_wait_ms < 0)
        die_usage("numeric flag out of range");

    char csv_copy[4096];
    snprintf(csv_copy, sizeof(csv_copy), "%s", hosts_csv);
    char *save = NULL;
    for (char *tok = strtok_r(csv_copy, ",", &save); tok != NULL;
         tok = strtok_r(NULL, ",", &save)) {
        if (cfg.member_count >= SPARK_REGISTRAR_MAX_MEMBERS)
            die_usage("too many hosts (max 64)");
        snprintf(cfg.hosts[cfg.member_count], sizeof(cfg.hosts[0]), "%s", tok);
        cfg.ports[cfg.member_count] = cfg.port_base + cfg.member_count;
        cfg.member_count++;
    }
    if (cfg.member_count == 0) die_usage("--hosts parsed to zero members");
    if (cfg.rank >= cfg.member_count) die_usage("--rank outside the launch table");

    cfg.expected_mask = 0;
    cfg.leader_rank = -1;
    if (subset_csv != NULL) {
        if (parse_rank_list(subset_csv, &cfg.expected_mask, &cfg.leader_rank) != 0)
            die_usage("--expect-subset malformed");
        if (!(cfg.expected_mask & ((uint64_t)1 << cfg.rank)))
            die_usage("--rank not in --expect-subset");
    } else {
        for (int r = 0; r < cfg.member_count; r++)
            cfg.expected_mask |= (uint64_t)1 << r;
        cfg.leader_rank = 0;
    }
    for (int r = 0; r < cfg.member_count; r++)
        cfg.is_expected[r] = (cfg.expected_mask >> r) & 1u;
}

static int set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) return -1;
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

static void open_listen(void) {
    st.listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (st.listen_fd < 0) { log_line("fatal socket errno=%d", errno); exit(2); }
    int one = 1;
    setsockopt(st.listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons((uint16_t)cfg.ports[cfg.rank]);
    if (bind(st.listen_fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        log_line("fatal bind port=%d errno=%d", cfg.ports[cfg.rank], errno);
        exit(2);
    }
    if (listen(st.listen_fd, 64) != 0) { perror("listen"); exit(2); }
    set_nonblocking(st.listen_fd);
}

static conn *conn_slot(void) {
    for (int i = 0; i < SPARK_REGISTRAR_MAX_CONNS; i++)
        if (st.conns[i].fd < 0) return &st.conns[i];
    return NULL;
}

static void conn_close(conn *c) {
    if (c->fd >= 0) close(c->fd);
    c->fd = -1;
    c->peer = -1;
    c->is_client = 0;
    c->sent_announce = 0;
    c->len = 0;
    c->deadline_ms = 0;
}

static void compose_announce(char *out, size_t out_size) {
    snprintf(out, out_size, "%s %d %016llx %ld %d\n", SPARK_REGISTRAR_WIRE,
             cfg.rank, (unsigned long long)st.my_view,
             (long)st.my_stale_pid, st.my_stale_immune);
}

static void merge_announce(int rank, uint64_t view, pid_t stale_pid,
                           int immune, int from_client_conn) {
    (void)from_client_conn;
    if (rank < 0 || rank >= cfg.member_count) return;
    if (!cfg.is_expected[rank]) {
        if (!(st.unexpected_seen & ((uint64_t)1 << rank))) {
            st.unexpected_seen |= (uint64_t)1 << rank;
            log_line("unexpected_announce rank=%d from_rank=%d not_in_expected=1",
                     cfg.rank, rank);
        }
        return;
    }
    if (rank != cfg.rank) {
        uint64_t old_view = st.peer_view[rank];
        pid_t old_pid = st.peer_stale[rank];
        int old_imm = st.peer_stale_immune[rank];
        int old_seen = st.peer_view_seen[rank];
        st.peer_view[rank] = view;
        st.peer_view_seen[rank] = 1;
        st.peer_stale[rank] = stale_pid;
        st.peer_stale_immune[rank] = immune;
        st.peer_stale_seen[rank] = 1;
        if (!old_seen) {
            char text[512];
            format_set(view, text, sizeof(text));
            log_line("peer_seen rank=%d peer=%d view=%s stale=%ld", cfg.rank,
                     rank, text, (long)stale_pid);
        } else if (old_view != view) {
            char text[512];
            format_set(view, text, sizeof(text));
            log_line("peer_view rank=%d peer=%d now=%s", cfg.rank, rank, text);
        }
        if (old_seen && (old_pid != stale_pid || old_imm != immune))
            log_line("peer_stale rank=%d peer=%d pid=%ld immune=%d", cfg.rank,
                     rank, (long)stale_pid, immune);
        if (!(st.my_view & ((uint64_t)1 << rank))) {
            st.my_view |= (uint64_t)1 << rank;
            char text[512];
            format_set(st.my_view, text, sizeof(text));
            log_line("view rank=%d now=%s", cfg.rank, text);
        }
    }
}

static void parse_line(const char *line) {
    unsigned long long view = 0;
    int rank = -1, immune = 0;
    long pid = 0;
    if (sscanf(line, SPARK_REGISTRAR_WIRE " %d %llx %ld %d", &rank, &view,
               &pid, &immune) == 4) {
        merge_announce(rank, (uint64_t)view, (pid_t)pid, immune, 0);
        return;
    }
    int from = -1;
    if (sscanf(line, SPARK_REGISTRAR_GO_WIRE " %d", &from) == 1) {
        if (!st.go_received) {
            st.go_received = 1;
            st.go_from = from;
        }
    }
}

static int dial_peer(int peer_rank, long long deadline_ms, conn *c) {
    struct addrinfo hints, *res = NULL;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    char port_text[16];
    snprintf(port_text, sizeof(port_text), "%d", cfg.ports[peer_rank]);
    if (getaddrinfo(cfg.hosts[peer_rank], port_text, &hints, &res) != 0 ||
        res == NULL)
        return -1;
    int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd < 0) { freeaddrinfo(res); return -1; }
    set_nonblocking(fd);
    int rc = connect(fd, res->ai_addr, res->ai_addrlen);
    freeaddrinfo(res);
    if (rc != 0 && errno != EINPROGRESS) { close(fd); return -1; }
    c->fd = fd;
    c->peer = peer_rank;
    c->is_client = 1;
    c->sent_announce = 0;
    c->len = 0;
    c->deadline_ms = deadline_ms;
    return 0;
}

static int proc_alive_and_matching(pid_t pid) {
    char path[64], link[PATH_MAX], state = 'R';
    snprintf(path, sizeof(path), "/proc/%ld/stat", (long)pid);
    FILE *f = fopen(path, "r");
    if (f == NULL) return 0;
    char line[1024];
    if (fgets(line, sizeof(line), f) != NULL) {
        char *close_paren = strrchr(line, ')');
        if (close_paren != NULL && close_paren + 2 < line + sizeof(line))
            state = close_paren[2];
    }
    fclose(f);
    if (state == 'Z') return 0;
    snprintf(path, sizeof(path), "/proc/%ld/exe", (long)pid);
    ssize_t n = readlink(path, link, sizeof(link) - 1);
    if (n < 0) return 0;
    link[n] = '\0';
    const char *base = strrchr(link, '/');
    base = base ? base + 1 : link;
    if (strcmp(base, cfg.daemon_name) != 0) return 0;
    if (cfg.deployment_cwd[0] == '\0') return 0;
    snprintf(path, sizeof(path), "/proc/%ld/cwd", (long)pid);
    n = readlink(path, link, sizeof(link) - 1);
    if (n < 0) return 0;
    link[n] = '\0';
    return strcmp(link, cfg.deployment_cwd) == 0;
}

static void stale_scan(long long now) {
    if (cfg.deployment_cwd[0] == '\0') return;
    pid_t found[SPARK_REGISTRAR_MAX_STALE];
    int found_count = 0;
    DIR *dir = opendir("/proc");
    if (dir == NULL) return;
    struct dirent *de;
    while ((de = readdir(dir)) != NULL) {
        if (de->d_name[0] < '0' || de->d_name[0] > '9') continue;
        pid_t pid = (pid_t)atoi(de->d_name);
        if (pid <= 0 || pid == getpid()) continue;
        if (!proc_alive_and_matching(pid)) continue;
        if (found_count < SPARK_REGISTRAR_MAX_STALE) found[found_count++] = pid;
    }
    closedir(dir);

    for (int i = 0; i < found_count; i++) {
        int known = 0;
        for (int j = 0; j < st.stale_count; j++)
            if (st.stale[j].pid == found[i]) { known = 1; break; }
        if (known) continue;
        if (st.stale_count >= SPARK_REGISTRAR_MAX_STALE) break;
        stale_entry *e = &st.stale[st.stale_count++];
        e->pid = found[i];
        e->immune = 0;
        e->immune_logged = 0;
        e->term_deadline_ms = now + cfg.term_wait_ms;
        log_line("stale rank=%d pid=%ld cwd=%s daemon=%s", cfg.rank,
                 (long)e->pid, cfg.deployment_cwd, cfg.daemon_name);
        int was_killed = (kill(e->pid, SIGTERM) == 0);
        log_line("stale_term rank=%d pid=%ld sent=%d", cfg.rank, (long)e->pid,
                 was_killed);
    }

    for (int j = st.stale_count - 1; j >= 0; j--) {
        if (!proc_alive_and_matching(st.stale[j].pid)) {
            log_line("stale_clear rank=%d pid=%ld", cfg.rank,
                     (long)st.stale[j].pid);
            memmove(&st.stale[j], &st.stale[j + 1],
                    sizeof(stale_entry) * (size_t)(st.stale_count - j - 1));
            st.stale_count--;
        } else if (now >= st.stale[j].term_deadline_ms) {
            st.stale[j].immune = 1;
            if (!st.stale[j].immune_logged) {
                st.stale[j].immune_logged = 1;
                log_line("stale_immune rank=%d pid=%ld term_wait_ms=%lld "
                         "action=operator_required", cfg.rank,
                         (long)st.stale[j].pid, cfg.term_wait_ms);
            }
        }
    }
    st.my_stale_pid = st.stale_count > 0 ? st.stale[0].pid : 0;
    st.my_stale_immune = 0;
    for (int j = 0; j < st.stale_count; j++)
        if (st.stale[j].immune) st.my_stale_immune = 1;
}

static int three_levels_met(void) {
    if (st.my_view != cfg.expected_mask) return 0;
    if (st.my_stale_pid != 0) return 0;
    for (int r = 0; r < cfg.member_count; r++) {
        if (!cfg.is_expected[r] || r == cfg.rank) continue;
        if (!st.peer_view_seen[r] || st.peer_view[r] != cfg.expected_mask)
            return 0;
        if (!st.peer_stale_seen[r] || st.peer_stale[r] != 0) return 0;
    }
    return 1;
}

static int send_go_once(int peer_rank, long long deadline_ms) {
    if (st.go_sent & ((uint64_t)1 << peer_rank)) return 0;
    st.go_sent |= (uint64_t)1 << peer_rank;
    conn *c = conn_slot();
    if (c == NULL) return -1;
    if (dial_peer(peer_rank, deadline_ms, c) != 0) return -1;
    char msg[SPARK_REGISTRAR_MSG_MAX];
    snprintf(msg, sizeof(msg), "%s %d\n", SPARK_REGISTRAR_GO_WIRE, cfg.rank);
    size_t len = strlen(msg);
    struct pollfd pfd = { .fd = c->fd, .events = POLLOUT, .revents = 0 };
    if (poll(&pfd, 1, 300) > 0 && (pfd.revents & POLLOUT)) {
        ssize_t n = write(c->fd, msg, len);
        (void)n;
    }
    conn_close(c);
    return 0;
}

static void go_relay(long long now) {
    int sent = 0, skipped_self = 0;
    for (int r = 0; r < cfg.member_count; r++) {
        if (!cfg.is_expected[r] || r == cfg.rank) { skipped_self++; continue; }
        if (send_go_once(r, now + 300) == 0) sent++;
    }
    log_line("relay_go rank=%d sent=%d members=%d", cfg.rank, sent,
             cfg.member_count);
    (void)skipped_self;
}

static void fail_loud(long long elapsed, int go_wait) {
    int expected_count = 0, view_count = 0, stale_clear = 0;
    for (int r = 0; r < cfg.member_count; r++) {
        if (!cfg.is_expected[r]) continue;
        expected_count++;
        if (st.my_view & ((uint64_t)1 << r)) view_count++;
    }
    for (int r = 0; r < cfg.member_count; r++)
        if (cfg.is_expected[r] && r != cfg.rank && st.peer_stale_seen[r] &&
            st.peer_stale[r] == 0)
            stale_clear++;

    log_prefix();
    printf("REGISTRAR FAIL rank=%d host=%s after_ms=%lld view=%d/%d "
           "peer_views=%d/%d stale_clear=%d/%d go_wait=%d\n",
           cfg.rank, cfg.hosts[cfg.rank], elapsed, view_count,
           expected_count, view_count, expected_count,
           st.my_stale_pid == 0 ? stale_clear + 1 : stale_clear,
           expected_count, go_wait);
    fflush(stdout);

    char line[2048];
    int used = snprintf(line, sizeof(line), "FAIL_DIFF rank=%d ", cfg.rank);
    char set_text[512];

    uint64_t missing = cfg.expected_mask & ~st.my_view;
    if (missing) {
        format_set(missing, set_text, sizeof(set_text));
        used += snprintf(line + used, sizeof(line) - (size_t)used,
                         "MISSING: %s ", set_text);
    }
    int partial = 0;
    used += snprintf(line + used, sizeof(line) - (size_t)used,
                     "PARTIAL-VIEW:");
    for (int r = 0; r < cfg.member_count; r++) {
        if (!cfg.is_expected[r] || r == cfg.rank) continue;
        if (!st.peer_view_seen[r]) continue;
        if (st.peer_view[r] == cfg.expected_mask) continue;
        format_set(st.peer_view[r], set_text, sizeof(set_text));
        used += snprintf(line + used, sizeof(line) - (size_t)used,
                         " %d->%s", r, set_text);
        partial = 1;
    }
    if (!partial)
        used += snprintf(line + used, sizeof(line) - (size_t)used, " none");
    int stale_listed = 0;
    used += snprintf(line + used, sizeof(line) - (size_t)used, " STALE:");
    if (st.my_stale_pid != 0 && !st.my_stale_immune) {
        used += snprintf(line + used, sizeof(line) - (size_t)used, " %d->%ld",
                         cfg.rank, (long)st.my_stale_pid);
        stale_listed = 1;
    }
    for (int r = 0; r < cfg.member_count; r++) {
        if (!cfg.is_expected[r] || r == cfg.rank) continue;
        if (!st.peer_stale_seen[r] || st.peer_stale[r] == 0) continue;
        if (st.peer_stale_immune[r]) continue;
        used += snprintf(line + used, sizeof(line) - (size_t)used, " %d->%ld",
                         r, (long)st.peer_stale[r]);
        stale_listed = 1;
    }
    if (!stale_listed)
        used += snprintf(line + used, sizeof(line) - (size_t)used, " none");
    used += snprintf(line + used, sizeof(line) - (size_t)used, " STALE-IMMUNE:");
    int immune_listed = 0;
    if (st.my_stale_pid != 0 && st.my_stale_immune) {
        used += snprintf(line + used, sizeof(line) - (size_t)used, " %d->%ld",
                         cfg.rank, (long)st.my_stale_pid);
        immune_listed = 1;
    }
    for (int r = 0; r < cfg.member_count; r++) {
        if (!cfg.is_expected[r] || r == cfg.rank) continue;
        if (!st.peer_stale_seen[r] || !st.peer_stale_immune[r]) continue;
        used += snprintf(line + used, sizeof(line) - (size_t)used, " %d->%ld",
                         r, (long)st.peer_stale[r]);
        immune_listed = 1;
    }
    if (!immune_listed)
        snprintf(line + used, sizeof(line) - (size_t)used, " none");
    log_prefix();
    printf("REGISTRAR %s\n", line);
    fflush(stdout);
}

int main(int argc, char **argv) {
    signal(SIGPIPE, SIG_IGN);
    parse_args(argc, argv);
    memset(&st, 0, sizeof(st));
    st.listen_fd = -1;
    for (int i = 0; i < SPARK_REGISTRAR_MAX_CONNS; i++) st.conns[i].fd = -1;

    open_listen();
    int cleanslate_enabled = (cfg.deployment_cwd[0] != '\0');
    if (cleanslate_enabled) {
        DIR *procd = opendir("/proc");
        if (procd == NULL) {
            log_line("fatal rank=%d /proc_unavailable cleanslate cannot be "
                     "verified; refusing to run", cfg.rank);
            exit(2);
        }
        closedir(procd);
    }
    st.my_view = (uint64_t)1 << cfg.rank;
    char members_text[1024];
    format_set(cfg.expected_mask, members_text, sizeof(members_text));
    log_line("start rank=%d host=%s port=%d leader=%d expected=%s "
             "deployment_cwd=%s daemon=%s cleanslate=%s announce_ms=%lld "
             "timeout_ms=%lld term_wait_ms=%lld",
             cfg.rank, cfg.hosts[cfg.rank], cfg.ports[cfg.rank],
             cfg.leader_rank, members_text, cfg.deployment_cwd,
             cfg.daemon_name, cleanslate_enabled ? "enabled" : "disabled",
             cfg.announce_ms, cfg.timeout_ms, cfg.term_wait_ms);

    long long t0 = now_ms();
    long long next_announce = 0;
    long long last_stale_scan = -1000000;

    for (;;) {
        long long now = now_ms();
        long long elapsed = now - t0;

        if (now - last_stale_scan >= cfg.announce_ms) {
            stale_scan(now);
            last_stale_scan = now;
        }
        if (now >= next_announce) {
            next_announce = now + cfg.announce_ms;
            for (int r = 0; r < cfg.member_count; r++) {
                if (!cfg.is_expected[r] || r == cfg.rank) continue;
                int live = 0;
                for (int i = 0; i < SPARK_REGISTRAR_MAX_CONNS; i++)
                    if (st.conns[i].fd >= 0 && st.conns[i].is_client &&
                        st.conns[i].peer == r)
                        live = 1;
                if (live) continue;
                conn *c = conn_slot();
                if (c == NULL) continue;
                if (dial_peer(r, now + 750, c) != 0) continue;
            }
        }

        if (!st.go_received && three_levels_met()) {
            if (cfg.rank == cfg.leader_rank) {
                char text[512];
                format_set(cfg.expected_mask, text, sizeof(text));
                log_line("GO rank=%d leader=%d levels=3/3 cleanslate=%s "
                         "expected=%s", cfg.rank, cfg.leader_rank,
                         cfg.deployment_cwd[0] ? "enabled" : "disabled", text);
                go_relay(now);
                log_line("exit rank=%d status=0", cfg.rank);
                return 0;
            }
            if (!st.ready_logged) {
                st.ready_logged = 1;
                log_line("ready rank=%d levels=3/3 cleanslate=%s waiting_go "
                         "leader=%d", cfg.rank,
                         cfg.deployment_cwd[0] ? "enabled" : "disabled",
                         cfg.leader_rank);
            }
        }
        if (st.go_received) {
            log_line("GO rank=%d by=%d", cfg.rank, st.go_from);
            go_relay(now);
            log_line("exit rank=%d status=0", cfg.rank);
            return 0;
        }

        if (elapsed >= cfg.timeout_ms) {
            fail_loud(elapsed, st.ready_logged);
            log_line("exit rank=%d status=1", cfg.rank);
            return 1;
        }

        struct pollfd pfds[SPARK_REGISTRAR_MAX_CONNS + 1];
        conn *mapping[SPARK_REGISTRAR_MAX_CONNS + 1];
        int n = 0;
        pfds[n].fd = st.listen_fd;
        pfds[n].events = POLLIN;
        pfds[n].revents = 0;
        n++;
        for (int i = 0; i < SPARK_REGISTRAR_MAX_CONNS; i++) {
            conn *c = &st.conns[i];
            if (c->fd < 0) continue;
            if (now >= c->deadline_ms) { conn_close(c); continue; }
            short events = POLLIN;
            if (c->is_client && !c->sent_announce) events = POLLOUT;
            pfds[n].fd = c->fd;
            pfds[n].events = (short)(events | POLLHUP | POLLERR | POLLNVAL);
            pfds[n].revents = 0;
            mapping[n] = c;
            n++;
        }

        long long wait = next_announce - now;
        if (wait > 50) wait = 50;
        if (wait < 0) wait = 0;
        int rc = poll(pfds, (nfds_t)n, (int)wait);
        if (rc < 0 && errno != EINTR) { log_line("poll_error errno=%d", errno); }

        for (int i = 1; i < n; i++) {
            conn *c = mapping[i];
            if (c->fd < 0) continue;
            short re = pfds[i].revents;
            if (re & (POLLERR | POLLNVAL)) { conn_close(c); continue; }
            if (c->is_client) {
                if ((re & POLLOUT) && !c->sent_announce) {
                    int err = 0;
                    socklen_t elen = sizeof(err);
                    getsockopt(c->fd, SOL_SOCKET, SO_ERROR, &err, &elen);
                    if (err != 0) { conn_close(c); continue; }
                    char msg[SPARK_REGISTRAR_MSG_MAX];
                    compose_announce(msg, sizeof(msg));
                    ssize_t w = write(c->fd, msg, strlen(msg));
                    if (w < 0) { conn_close(c); continue; }
                    c->sent_announce = 1;
                    c->deadline_ms = now_ms() + 750;
                } else if ((re & POLLIN) && c->sent_announce) {
                    char buf[SPARK_REGISTRAR_MSG_MAX];
                    ssize_t r = read(c->fd, buf, sizeof(buf) - 1);
                    if (r > 0) {
                        buf[r] = '\0';
                        char *nl = strchr(buf, '\n');
                        if (nl != NULL) *nl = '\0';
                        parse_line(buf);
                    }
                    conn_close(c);
                } else if (re & (POLLHUP)) {
                    conn_close(c);
                }
            } else {
                if (re & POLLIN) {
                    char chunk[SPARK_REGISTRAR_MSG_MAX];
                    ssize_t r = read(c->fd, chunk, sizeof(chunk) - 1);
                    if (r <= 0) { conn_close(c); continue; }
                    ssize_t space = (ssize_t)sizeof(c->buf) - (ssize_t)c->len - 1;
                    if (r > space) r = space;
                    memcpy(c->buf + c->len, chunk, (size_t)r);
                    c->len += (size_t)r;
                    c->buf[c->len] = '\0';
                    char *nl = strchr(c->buf, '\n');
                    if (nl != NULL) {
                        *nl = '\0';
                        parse_line(c->buf);
                        char reply[SPARK_REGISTRAR_MSG_MAX];
                        compose_announce(reply, sizeof(reply));
                        ssize_t w = write(c->fd, reply, strlen(reply));
                        (void)w;
                        conn_close(c);
                    } else if (c->len >= sizeof(c->buf) - 1) {
                        conn_close(c);
                    }
                } else if (re & POLLHUP) {
                    conn_close(c);
                }
            }
        }

        if (pfds[0].revents & POLLIN) {
            for (;;) {
                int fd = accept(st.listen_fd, NULL, NULL);
                if (fd < 0) break;
                set_nonblocking(fd);
                int one = 1;
                setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
                conn *c = conn_slot();
                if (c == NULL) { close(fd); continue; }
                c->fd = fd;
                c->peer = -1;
                c->is_client = 0;
                c->sent_announce = 0;
                c->len = 0;
                c->deadline_ms = now_ms() + 1000;
            }
        }
    }
}
