#!/usr/bin/env python3
"""Copy files and directories between any fleet machines over the fastest path.

  sparkcp SRC... DST

Each SRC and DST is a local path or NODE:PATH. NODE is spark0-sparkf,
lenovo7-lenovo9 or hub, and a relative remote path starts at that node's home.
Destinations follow cp -r. Run it from the Mac, the hub or any node.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import hmac
import json
import mmap
import os
import re
import secrets
import shlex
import socket
import stat
import struct
import subprocess
import sys
import threading
import time
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple


CHUNK_BYTES = 16 * 1024 * 1024
DIRECT_MIN_BYTES = 64 * 1024 * 1024
ALIGN_BYTES = 4096
STREAMS = 8
NICE = 10
CONNECT_TIMEOUT_S = 1.5
HANDSHAKE_TIMEOUT_S = 5.0
STALL_TIMEOUT_S = 60.0
SSH_TIMEOUT_S = 4
PROGRESS_INTERVAL_S = 1.0
TOKEN_BYTES = 32
GB = 1e9
LOOPBACK = "127.0.0.1"
CISCO_PREFIX = "10.10.201."
PATHS = (
    ("cisco",CISCO_PREFIX),
    ("crs804","10.10.100."),
    ("mgmt","10.20.0."),
    ("lan","192.168.50."),
    ("tailscale","100."),
)
ALIASES = {"rtx5090":"hub"}
LENGTH = struct.Struct("<I")
FRAME = struct.Struct("<IQI32s")


class CopyError(Exception):
    pass


class Node(NamedTuple):
    user: str
    cisco: str


class Endpoint(NamedTuple):
    node: str
    path: str


def fleet_nodes() -> Dict[str,Node]:
    nodes = {"hub":Node("spec",CISCO_PREFIX + "250")}
    for rank,digit in enumerate("0123456789abcdef"):
        nodes["spark" + digit] = Node("spark" + digit,CISCO_PREFIX + str(rank))
    for index in (7,8,9):
        nodes["lenovo%d" % index] = Node("spark%d" % index,CISCO_PREFIX + str(240 + index))
    return(nodes)


def align_up(value: int) -> int:
    return((value + ALIGN_BYTES - 1) & ~(ALIGN_BYTES - 1))


def gb_text(value: float) -> str:
    return("%.1f GB" % (value / GB))


def local_addresses() -> List[str]:
    for cmd in (["ip","-4","-o","addr","show"],["ifconfig"]):
        try:
            output = subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,timeout=5).stdout
        except (OSError,subprocess.SubprocessError):
            continue
        found = re.findall(r"inet (\d+\.\d+\.\d+\.\d+)",output)
        if found:
            return(found)
    return([])


def ranked_addresses(addresses: List[str]) -> List[Tuple[str,str]]:
    ranked = [("loopback",address) for address in addresses if address == LOOPBACK]
    for label,prefix in PATHS:
        ranked.extend((label,address) for address in addresses if address.startswith(prefix))
    return(ranked)


def send_json(sock: socket.socket, value: object) -> None:
    data = json.dumps(value).encode()
    sock.sendall(LENGTH.pack(len(data)) + data)


def recv_exact(sock: socket.socket, view: memoryview) -> None:
    done = 0
    while done < len(view):
        got = sock.recv_into(view[done:])
        if got == 0:
            raise CopyError("connection closed mid-transfer")
        done += got


def read_json(sock: socket.socket) -> Dict[str,object]:
    head = bytearray(LENGTH.size)
    recv_exact(sock,memoryview(head))
    body = bytearray(LENGTH.unpack(bytes(head))[0])
    recv_exact(sock,memoryview(body))
    return(json.loads(bytes(body)))


def read_into(fd: int, view: memoryview, offset: int) -> int:
    if hasattr(os,"preadv"):
        return(os.preadv(fd,[view],offset))
    data = os.pread(fd,len(view),offset)
    view[:len(data)] = data
    return(len(data))


def open_direct(path: str, flags: int, mode: int = 0o600) -> Tuple[int,bool]:
    direct = getattr(os,"O_DIRECT",0)
    if direct:
        try:
            return(os.open(path,flags | direct,mode),True)
        except OSError:
            pass
    return(os.open(path,flags,mode),False)


def preallocate(fd: int, size: int) -> None:
    if hasattr(os,"posix_fallocate"):
        try:
            os.posix_fallocate(fd,0,size)
            return
        except OSError as exc:
            if exc.errno == errno.ENOSPC:
                raise
    os.ftruncate(fd,size)


def scan_entry(rel: str, path: str, info: os.stat_result) -> Dict[str,object]:
    entry = {"rel":rel,"mode":stat.S_IMODE(info.st_mode),"mtime_ns":info.st_mtime_ns,"size":0,"target":""}
    if stat.S_ISDIR(info.st_mode):
        entry["kind"] = "d"
    elif stat.S_ISLNK(info.st_mode):
        entry["kind"] = "l"
        entry["target"] = os.readlink(path)
    elif stat.S_ISREG(info.st_mode):
        entry["kind"] = "f"
        entry["size"] = info.st_size
    else:
        raise CopyError("not a file, directory or link: %s" % path)
    return(entry)


def scan(sources: List[str]) -> Tuple[List[Dict[str,object]],List[str]]:
    entries: List[Dict[str,object]] = []
    paths: List[str] = []
    for source in sources:
        root = os.path.abspath(os.path.expanduser(source))
        base = os.path.basename(root)
        if base == "":
            raise CopyError("cannot copy a filesystem root")
        info = os.stat(root)
        entries.append(scan_entry(base,root,info))
        paths.append(root)
        if not stat.S_ISDIR(info.st_mode):
            continue
        for folder,dirs,files in os.walk(root):
            dirs.sort()
            for name in dirs + sorted(files):
                path = os.path.join(folder,name)
                entries.append(scan_entry(os.path.join(base,os.path.relpath(path,root)),path,os.lstat(path)))
                paths.append(path)
    return(entries,paths)


def checked_rel(rel: str) -> str:
    parts = rel.split("/")
    if rel.startswith("/") or any(part in ("","..",".") for part in parts):
        raise CopyError("unsafe path in manifest: %r" % rel)
    return(rel)


def plan_chunks(need: List[int], entries: List[Dict[str,object]]) -> Iterator[Tuple[int,int,int]]:
    for index in need:
        size = int(entries[index]["size"])
        for offset in range(0,size,CHUNK_BYTES):
            yield(index,offset,min(CHUNK_BYTES,size - offset))


class Receiver:
    def __init__(self, dest: str, entries: List[Dict[str,object]], token: str):
        self.entries = entries
        self.suffix = token[:8]
        self.lock = threading.Lock()
        self.pending: Dict[int,Dict[str,object]] = {}
        self.dirs: List[Tuple[str,Dict[str,object]]] = []
        self.dest = os.path.abspath(os.path.expanduser(dest))
        roots = [entry for entry in entries if "/" not in str(entry["rel"])]
        self.into = len(roots) > 1 or dest.endswith("/") or os.path.isdir(self.dest)

    def target(self, rel: str) -> str:
        checked_rel(rel)
        if self.into:
            return(os.path.join(self.dest,rel))
        parts = rel.split("/")
        return(os.path.join(self.dest,*parts[1:]))

    def unchanged(self, path: str, entry: Dict[str,object]) -> bool:
        try:
            info = os.lstat(path)
        except OSError:
            return(False)
        return(stat.S_ISREG(info.st_mode) and info.st_size == entry["size"] and info.st_mtime_ns == entry["mtime_ns"])

    def check_space(self, paths: List[str]) -> None:
        needed = 0
        for index,path in zip(self.pending,paths):
            needed += int(self.entries[index]["size"])
            if os.path.lexists(path):
                needed -= os.lstat(path).st_size
        probe = self.dest if os.path.isdir(self.dest) else os.path.dirname(self.dest)
        info = os.statvfs(probe)
        free = info.f_bavail * info.f_frsize
        if needed > free:
            raise CopyError("%s needed but only %s free on %s" % (gb_text(needed),gb_text(free),probe))

    def plan(self) -> List[int]:
        os.makedirs(self.dest if self.into else os.path.dirname(self.dest),exist_ok=True)
        need: List[int] = []
        paths: List[str] = []
        for index,entry in enumerate(self.entries):
            path = self.target(str(entry["rel"]))
            if entry["kind"] == "d":
                os.makedirs(path,exist_ok=True)
                self.dirs.append((path,entry))
            elif entry["kind"] == "l":
                os.makedirs(os.path.dirname(path),exist_ok=True)
                if os.path.lexists(path):
                    os.unlink(path)
                os.symlink(str(entry["target"]),path)
            elif not self.unchanged(path,entry):
                os.makedirs(os.path.dirname(path),exist_ok=True)
                temp = os.path.join(os.path.dirname(path),".sparkcp-%s-%d" % (self.suffix,index))
                self.pending[index] = {"path":path,"temp":temp,"fd":-1,"direct":False,"remaining":int(entry["size"])}
                need.append(index)
                paths.append(path)
        self.check_space(paths)
        for index in [index for index in need if self.entries[index]["size"] == 0]:
            self.open_file(index)
            self.finalize(index)
        return(need)

    def open_file(self, index: int) -> Dict[str,object]:
        with self.lock:
            state = self.pending.get(index)
            if state is None:
                raise CopyError("data for a file that is not pending: %d" % index)
            if state["fd"] < 0:
                flags = os.O_CREAT | os.O_WRONLY | os.O_TRUNC
                size = int(self.entries[index]["size"])
                if size >= DIRECT_MIN_BYTES:
                    state["fd"],state["direct"] = open_direct(str(state["temp"]),flags)
                    preallocate(int(state["fd"]),size)
                else:
                    state["fd"] = os.open(str(state["temp"]),flags,0o600)
                    os.ftruncate(int(state["fd"]),size)
            return(state)

    def write(self, index: int, offset: int, view: memoryview, length: int) -> None:
        state = self.open_file(index)
        if offset + length > int(self.entries[index]["size"]):
            raise CopyError("data past the end of %s" % state["path"])
        size = align_up(length) if state["direct"] else length
        view[length:size] = bytes(size - length)
        done = 0
        while done < size:
            done += os.pwrite(int(state["fd"]),view[done:size],offset + done)
        with self.lock:
            state["remaining"] = int(state["remaining"]) - length
            finished = state["remaining"] == 0
        if finished:
            self.finalize(index)

    def finalize(self, index: int) -> None:
        with self.lock:
            state = self.pending[index]
        entry = self.entries[index]
        fd = int(state["fd"])
        try:
            if state["direct"]:
                os.ftruncate(fd,int(entry["size"]))
                os.fsync(fd)
            os.fchmod(fd,int(entry["mode"]))
            os.utime(fd,ns=(int(entry["mtime_ns"]),int(entry["mtime_ns"])))
        finally:
            os.close(fd)
            state["fd"] = -1
        os.replace(str(state["temp"]),str(state["path"]))
        with self.lock:
            del self.pending[index]

    def finish(self) -> None:
        if self.pending:
            raise CopyError("%d files did not arrive complete" % len(self.pending))
        for path,entry in reversed(self.dirs):
            os.chmod(path,int(entry["mode"]))
            os.utime(path,ns=(int(entry["mtime_ns"]),int(entry["mtime_ns"])))

    def abort(self) -> None:
        for state in list(self.pending.values()):
            if int(state["fd"]) >= 0:
                try:
                    os.close(int(state["fd"]))
                except OSError:
                    pass
            try:
                os.unlink(str(state["temp"]))
            except OSError:
                pass
        self.pending.clear()


def receive_stream(sock: socket.socket, receiver: Receiver) -> None:
    view = memoryview(mmap.mmap(-1,CHUNK_BYTES + ALIGN_BYTES))
    head = bytearray(FRAME.size)
    head_view = memoryview(head)
    while True:
        first = sock.recv_into(head_view)
        if first == 0:
            return
        recv_exact(sock,head_view[first:])
        index,offset,length,digest = FRAME.unpack(bytes(head))
        if length > CHUNK_BYTES:
            raise CopyError("oversized chunk: %d" % length)
        recv_exact(sock,view[:length])
        if not hmac.compare_digest(hashlib.sha256(view[:length]).digest(),digest):
            raise CopyError("checksum mismatch at offset %d of file %d" % (offset,index))
        receiver.write(index,offset,view,length)


def send_stream(sock: socket.socket, chunks: Iterator[Tuple[int,int,int]], paths: List[str], entries: List[Dict[str,object]], lock: threading.Lock, stats: Dict[str,int]) -> None:
    view = memoryview(mmap.mmap(-1,CHUNK_BYTES + ALIGN_BYTES))
    current = -1
    fd = -1
    direct = False
    try:
        while True:
            with lock:
                item = next(chunks,None)
            if item is None:
                return
            index,offset,length = item
            if index != current:
                if fd >= 0:
                    os.close(fd)
                fd = -1
                if int(entries[index]["size"]) >= DIRECT_MIN_BYTES:
                    fd,direct = open_direct(paths[index],os.O_RDONLY)
                else:
                    fd,direct = os.open(paths[index],os.O_RDONLY),False
                current = index
            got = 0
            while got < length:
                request = align_up(length) if direct else length
                count = read_into(fd,view[got:request],offset + got)
                if count == 0:
                    raise CopyError("%s shrank during the copy" % paths[index])
                got += count
            sock.sendall(FRAME.pack(index,offset,length,hashlib.sha256(view[:length]).digest()))
            sock.sendall(view[:length])
            with lock:
                stats["sent"] += length
    finally:
        if fd >= 0:
            os.close(fd)


def run_workers(worker, sockets: List[socket.socket], *args) -> None:
    errors: List[BaseException] = []

    def guarded(sock: socket.socket) -> None:
        try:
            worker(sock,*args)
        except (OSError,CopyError) as exc:
            errors.append(exc)
            for other in sockets:
                try:
                    other.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    threads = [threading.Thread(target=guarded,args=(sock,),daemon=True) for sock in sockets]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]


def emit(value: Dict[str,object]) -> None:
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()


def accept_verified(listener: socket.socket, token: str) -> socket.socket:
    expected = token.encode()
    while True:
        sock,_ = listener.accept()
        sock.settimeout(HANDSHAKE_TIMEOUT_S)
        try:
            presented = bytearray(len(expected))
            recv_exact(sock,memoryview(presented))
            if hmac.compare_digest(bytes(presented),expected):
                sock.settimeout(STALL_TIMEOUT_S)
                return(sock)
        except (OSError,CopyError):
            pass
        sock.close()


def run_receive(job: Dict[str,object], owned: Dict[str,Receiver]) -> None:
    token = str(job["token"])
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as listener:
        listener.bind(("",0))
        listener.listen(STREAMS + 2)
        emit({"port":listener.getsockname()[1],"addresses":local_addresses()})
        control = accept_verified(listener,token)
        manifest = read_json(control)
        receiver = Receiver(str(job["dest"]),list(manifest["entries"]),token)
        owned["receiver"] = receiver
        try:
            need = receiver.plan()
        except (OSError,CopyError) as exc:
            send_json(control,{"error":str(exc)})
            raise
        streams = STREAMS if receiver.pending else 0
        send_json(control,{"need":need,"streams":streams})
        sockets = [accept_verified(listener,token) for _ in range(streams)]
    try:
        run_workers(receive_stream,sockets,receiver)
        read_json(control)
        receiver.finish()
    except (OSError,CopyError) as exc:
        send_json(control,{"error":str(exc)})
        raise
    send_json(control,{"ok":True})


def connect_verified(address: str, port: int, token: str, timeout_s: float) -> socket.socket:
    sock = socket.create_connection((address,port),timeout=timeout_s)
    sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
    sock.settimeout(STALL_TIMEOUT_S)
    sock.sendall(token.encode())
    return(sock)


def connect_first(addresses: List[str], port: int, token: str) -> Tuple[socket.socket,str,str]:
    for label,address in ranked_addresses(addresses):
        try:
            return(connect_verified(address,port,token,CONNECT_TIMEOUT_S),label,address)
        except OSError:
            continue
    raise CopyError("no reachable path to the destination: %s" % ", ".join(addresses))


def receiver_reason(control: socket.socket) -> str:
    control.settimeout(HANDSHAKE_TIMEOUT_S)
    try:
        return(str(read_json(control).get("error","")))
    except (OSError,CopyError,ValueError):
        return("")


def report_progress(stats: Dict[str,int], total: int, done: threading.Event) -> None:
    start = time.monotonic()
    while not done.wait(PROGRESS_INTERVAL_S):
        emit({"sent":stats["sent"],"total":total,"seconds":time.monotonic() - start})


def run_send(job: Dict[str,object]) -> None:
    token = str(job["token"])
    entries,paths = scan(list(job["sources"]))
    port = int(job["port"])
    control,label,address = connect_first(list(job["addresses"]),port,token)
    send_json(control,{"entries":entries})
    reply = read_json(control)
    if "error" in reply:
        raise CopyError(str(reply["error"]))
    need = list(reply["need"])
    total = sum(int(entries[index]["size"]) for index in need)
    stats = {"sent":0}
    lock = threading.Lock()
    done = threading.Event()
    sockets = [connect_verified(address,port,token,HANDSHAKE_TIMEOUT_S) for _ in range(int(reply["streams"]))]
    threading.Thread(target=report_progress,args=(stats,total,done),daemon=True).start()
    start = time.monotonic()
    try:
        run_workers(send_stream,sockets,plan_chunks(need,entries),paths,entries,lock,stats)
    except (OSError,CopyError) as exc:
        raise CopyError(receiver_reason(control) or str(exc))
    finally:
        done.set()
    for sock in sockets:
        sock.shutdown(socket.SHUT_WR)
    send_json(control,{"done":True})
    reply = read_json(control)
    if "error" in reply:
        raise CopyError(str(reply["error"]))
    unchanged = sum(1 for entry in entries if entry["kind"] == "f") - len(need)
    emit({"finished":True,"path":label,"address":address,"bytes":total,"files":len(need),"unchanged":unchanged,"seconds":time.monotonic() - start})


def agent_main() -> int:
    job = json.loads(sys.stdin.buffer.readline())
    owned: Dict[str,Receiver] = {}

    def abort() -> None:
        if "receiver" in owned:
            owned["receiver"].abort()

    def watch_controller() -> None:
        sys.stdin.buffer.read()
        abort()
        os._exit(1)

    threading.Thread(target=watch_controller,daemon=True).start()
    os.nice(NICE)
    try:
        if job["role"] == "receive":
            run_receive(job,owned)
        else:
            run_send(job)
    except (OSError,CopyError,ValueError,KeyError) as exc:
        abort()
        emit({"error":str(exc)})
        return(1)
    return(0)


def this_node(nodes: Dict[str,Node]) -> str:
    name = socket.gethostname().split(".")[0].lower()
    return(ALIASES.get(name,name) if ALIASES.get(name,name) in nodes else "")


def parse_endpoint(text: str, nodes: Dict[str,Node], local: str) -> Endpoint:
    name,colon,path = text.partition(":")
    node = ALIASES.get(name.lower(),name.lower())
    if colon and node in nodes:
        return(Endpoint("" if node == local else node,path or "."))
    if colon and "/" not in name and not os.path.lexists(text):
        raise CopyError("unknown node %r, expected one of: %s" % (name," ".join(sorted(nodes))))
    return(Endpoint("",text))


def ssh_targets(node: str, nodes: Dict[str,Node], on_cisco: bool) -> List[str]:
    targets = ["%s@%s" % (nodes[node].user,nodes[node].cisco)] if on_cisco else []
    targets.append("rtx5090" if node == "hub" else node)
    return(targets)


class Agent:
    def __init__(self, node: str, nodes: Dict[str,Node], on_cisco: bool, source: bytes):
        self.node = node or "local"
        self.source = source
        self.bootstrap = "import sys;exec(compile(sys.stdin.buffer.read(%d),'sparkcp','exec'))" % len(source)
        if node:
            self.commands = [self.ssh_command(target) for target in ssh_targets(node,nodes,on_cisco)]
        else:
            self.commands = [[sys.executable,"-u","-c",self.bootstrap,"agent"]]
        self.process: Optional[subprocess.Popen] = None

    def ssh_command(self, target: str) -> List[str]:
        options = ["-T","-o","BatchMode=yes","-o","ConnectTimeout=%d" % SSH_TIMEOUT_S,"-o","StrictHostKeyChecking=accept-new","-o","LogLevel=ERROR"]
        return(["ssh",*options,target,"python3 -u -c %s agent" % shlex.quote(self.bootstrap)])

    def start(self, job: Dict[str,object]) -> Dict[str,object]:
        failure = ""
        payload = self.source + json.dumps(job).encode() + b"\n"
        for command in self.commands:
            self.process = subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            try:
                self.process.stdin.write(payload)
                self.process.stdin.flush()
            except OSError:
                pass
            first = self.read()
            if first is not None:
                return(first)
            failure = self.process.stderr.read().decode(errors="replace").strip()
            self.process.wait()
        raise CopyError("cannot start on %s: %s" % (self.node,failure))

    def read(self) -> Optional[Dict[str,object]]:
        line = self.process.stdout.readline()
        return(json.loads(line) if line else None)

    def stop(self) -> None:
        if self.process is None:
            return
        try:
            self.process.stdin.close()
        except OSError:
            pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process.stdout.close()
        self.process.stderr.close()


def show_progress(update: Dict[str,object]) -> None:
    if sys.stderr.isatty():
        rate = float(update["sent"]) / max(float(update["seconds"]),0.001)
        sys.stderr.write("\r%s / %s  %.2f GB/s   " % (gb_text(float(update["sent"])),gb_text(float(update["total"])),rate / GB))
        sys.stderr.flush()


def summary_text(result: Dict[str,object]) -> str:
    size = float(result["bytes"])
    seconds = max(float(result["seconds"]),0.001)
    files = int(result["files"])
    text = "sparkcp: %d file%s, %s in %.1f s, %.2f GB/s over %s" % (files,"" if files == 1 else "s",gb_text(size),seconds,size / seconds / GB,result["path"])
    if int(result["unchanged"]) > 0:
        text += ", %d already up to date" % int(result["unchanged"])
    return(text)


def copy(sources: List[Endpoint], dest: Endpoint, nodes: Dict[str,Node], on_cisco: bool, source: bytes) -> Dict[str,object]:
    origin = {endpoint.node for endpoint in sources}
    if len(origin) != 1:
        raise CopyError("all sources must be on one node")
    node = origin.pop()
    token = secrets.token_hex(TOKEN_BYTES // 2)
    receiver = Agent(dest.node,nodes,on_cisco,source)
    sender = Agent(node,nodes,on_cisco,source)
    try:
        offer = receiver.start({"role":"receive","token":token,"dest":dest.path})
        if "error" in offer:
            raise CopyError("%s: %s" % (receiver.node,offer["error"]))
        addresses = [LOOPBACK] if node == dest.node else list(offer["addresses"])
        job = {"role":"send","token":token,"sources":[endpoint.path for endpoint in sources],"addresses":addresses,"port":offer["port"]}
        update = sender.start(job)
        while update is not None:
            if "error" in update:
                raise CopyError("%s: %s" % (sender.node,update["error"]))
            if "finished" in update:
                return(update)
            show_progress(update)
            update = sender.read()
        raise CopyError("%s: the copy stopped without a result" % sender.node)
    finally:
        sender.stop()
        receiver.stop()


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="sparkcp",description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths",nargs="+",metavar="SRC... DST")
    args = parser.parse_args(argv)
    if len(args.paths) < 2:
        parser.error("need at least one source and a destination")
    return(args)


def main(argv: List[str]) -> int:
    if argv == ["agent"]:
        return(agent_main())
    args = parse_args(argv)
    nodes = fleet_nodes()
    local = this_node(nodes)
    try:
        endpoints = [parse_endpoint(text,nodes,local) for text in args.paths]
        with open(os.path.realpath(__file__),"rb") as fp:
            source = fp.read()
        on_cisco = any(address.startswith(CISCO_PREFIX) for address in local_addresses())
        result = copy(endpoints[:-1],endpoints[-1],nodes,on_cisco,source)
    except (OSError,CopyError) as exc:
        print("sparkcp: %s" % exc,file=sys.stderr)
        return(1)
    if sys.stderr.isatty():
        sys.stderr.write("\n")
    print(summary_text(result))
    return(0)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
