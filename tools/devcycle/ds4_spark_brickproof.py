#!/usr/bin/env python3
"""Apply and audit the canonical Spark anti-brick boot and OOM policy."""

from __future__ import annotations

import argparse
import base64
import binascii
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile


SELF_SOURCE = "tools/devcycle/ds4_spark_brickproof.py"
DEFAULT_NODES = tuple(f"spark{rank:x}" for rank in range(16))
DEFAULT_SPARKPIPE_REPO = Path.home() / "sparkpipe"
DEFAULT_RECOVERY_IDENTITY = Path.home() / ".ssh" / "sparkpipe_fleet_root"
REMOTE_BOOTSTRAP = ("import json,sys,types;d=json.load(sys.stdin);m=types.ModuleType('ds4_spark_brickproof');sys.modules[m.__name__]=m;"
    "exec(compile(d['script'],'ds4_spark_brickproof.py','exec'),m.__dict__);print(json.dumps(getattr(m,'remote_'+sys.argv[1])(d['payload']),sort_keys=True))")
MANAGEMENT_ADDRESS_BASE = 10
MANAGEMENT_GATEWAY = "10.20.0.1"
MANAGEMENT_INTERFACE = "enP7s7"
MANAGEMENT_NETMASK = "255.255.255.0"
MANAGEMENT_PREFIX = "10.20.0"
LAN_ADDRESS_BASE = 128
LAN_PREFIX = "192.168.50"
NODE_LAYOUT_SOURCE = "tools/spark_node_layout.json"
MAX_SPARK_RANK = 15
NODE_RANK_PATH = Path("/etc/ds4-node-rank")
SWITCHED_ADDRESS_BASE = 10
SWITCHED_PREFIX = "10.10.100"
SWITCHED_INTERFACE = "enp1s0f1np1"
SWITCHED_SPEED_MBIT = "100000"
DIRECT_INTERFACE = "enp1s0f0np0"
DIRECT_SPEED_MBIT = "200000"
MESH_LINK_MAP = "0-1:200,2-3:200,4-5:200,6-7:200,8-9:200,10-11:200,12-13:200,14-15:200"
HOSTS_BLOCK_BEGIN = "# SparkPipe fleet aliases BEGIN"
HOSTS_BLOCK_END = "# SparkPipe fleet aliases END"
LEGACY_CEPH_WARM_STORAGE_MARKER = Path("/etc/sparkpipe/enable-ceph-warm-storage")
HUB_GATEWAY_RANK = 15
HUB_NOTIFY = "192.168.50.4:5514"
NETWORK_MANAGER_RUNTIME = "/usr/local/sbin/ds4-switched-fabric-apply"
DMA_GUARD_RUNTIME = "/usr/local/sbin/spark-dma-guard"
CEPH_UID = 167
CEPH_STATE_ROOT = Path("/var/lib/ceph")
CEPH_LOG_ROOT = Path("/var/log/ceph")
CEPH_DAEMON_PREFIXES = ("mon.","mgr.","osd.","mds.","crash.")
CEPH_REQUIRED_DAEMON_PREFIXES = ("mon.","mgr.","osd.")
CEPH_MASK_UNITS = (
    "ceph-crash.service",
    "ceph-fuse@.service",
    "ceph-mds.target",
    "ceph-mds@.service",
    "ceph-mgr.target",
    "ceph-mgr@.service",
    "ceph-mon.target",
    "ceph-mon.service",
    "ceph-mon@.service",
    "ceph-osd.target",
    "ceph-osd@.service",
    "ceph-radosgw.target",
    "ceph-radosgw@.service",
    "ceph-volume-systemd@.service",
    "ceph-volume@.service",
)
OBSOLETE_CEPH_STARTUP_UNITS = (
    "ds4-optional-storage.service",
    "ds4-optional-storage.timer",
)
LEGACY_HOSTS_BLOCKS = {
    "# DS4 Spark aliases BEGIN":"# DS4 Spark aliases END",
    "# DS4 switched Spark aliases BEGIN":"# DS4 switched Spark aliases END",
    HOSTS_BLOCK_BEGIN:HOSTS_BLOCK_END,
}
REQUIRED_KERNEL_MODULES = ("sch_fq_codel",)
FLEET_TOOL_PACKAGES = ("iperf3","fio","ninja-build","gh","exfatprogs","python3.12-dev","libnng-dev","libmbedtls-dev","libexpat1-dev")
OPTIONAL_BOOT_UNITS = (
    "rbdmap.service","nvmf-autoconnect.service","open-iscsi.service","iscsid.service","srp_daemon.service","srptools.service",
    "cloud-init.service","cloud-init-local.service","cloud-config.service","cloud-final.service","pollinate.service",
    "nvidia-spark-run-apt-upgrade-once.service","systemd-networkd-wait-online.service","plymouth-quit-wait.service",
)
FABRIC_INTERFACES = ("enp1s0f0np0","enp1s0f1np1","enP2p1s0f0np0","enP2p1s0f1np1")
UPLINK_PROFILE = "ds4-uplink-wired"
UPLINK_GATEWAY = "192.168.50.1"
UPLINK_METRIC = "10"
MIN_MEMTOTAL_KB = 127000000
NTP_SERVERS = ("192.168.50.4","10.10.250.2")
CEPH_CLUSTER_FSID = "cd8715a1-47b9-4e4a-8d3c-2cddb02077d1"
FIREWALL_RULESET = Path("/etc/spark-firewall.nft")
DMA_GUARD_STATE = Path("/var/lib/spark-dma-guard/boots.json")
RECOVERY_SETTINGS = {"kernel.panic":"60","kernel.softlockup_panic":"1","kernel.hung_task_panic":"1","vm.swappiness":"10"}
KERNEL_MODULES_LOAD = "".join(f"{module}\n" for module in REQUIRED_KERNEL_MODULES)
ASSET_SOURCES = {
    "/usr/local/bin/sparkpipe_fsck_health.sh": ("tools/devcycle/sparkpipe_fsck_health.sh",0o755),
    "/etc/systemd/system/sparkpipe-fsck-health.service": ("tools/devcycle/sparkpipe-fsck-health.service",0o644),
}
BOOT_TIMEOUT_SOURCES = {
    "rbdmap.service": "rbdmap.service.conf",
    "nvmf-autoconnect.service": "nvmf-autoconnect.service.conf",
    "open-iscsi.service": "open-iscsi.service.conf",
    "pollinate.service": "pollinate.service.conf",
}
ASSET_SOURCES.update({
    "/etc/logrotate.d/00-sparkpipe-cephadm": ("tools/devcycle/fleet/00-sparkpipe-cephadm",0o644),
    "/etc/systemd/system/ds4-switched-fabric.service": ("tools/devcycle/fleet/ds4-switched-fabric.service",0o644),
    "/etc/systemd/system/ds4-switched-fabric.timer": ("tools/devcycle/fleet/ds4-switched-fabric.timer",0o644),
    "/etc/systemd/system/ds4-direct-pair-fabric.service": ("tools/devcycle/fleet/ds4-direct-pair-fabric.service",0o644),
    "/etc/systemd/system/ds4-direct-pair-fabric.timer": ("tools/devcycle/fleet/ds4-direct-pair-fabric.timer",0o644),
    "/usr/local/sbin/ds4-switched-fabric-apply": ("tools/devcycle/fleet/ds4_switched_fabric_apply.sh",0o755),
    "/usr/local/sbin/ds4-direct-pair-fabric-apply": ("tools/devcycle/fleet/ds4_direct_pair_fabric_apply.sh",0o755),
    "/etc/systemd/system/sparkpipe-fsck-health.timer": ("tools/devcycle/fleet/sparkpipe-fsck-health.timer",0o644),
    "/etc/systemd/system/spark-firewall.service.d/10-ds4-timeout.conf": ("tools/devcycle/fleet/spark-firewall-timeout.conf",0o644),
    "/etc/systemd/system/spark-firewall.service": ("tools/devcycle/fleet/spark-firewall.service",0o644),
    DMA_GUARD_RUNTIME: ("tools/devcycle/spark_dma_guard.py",0o755),
    "/usr/local/sbin/spark-osd-guard": ("tools/devcycle/fleet/spark_osd_guard.sh",0o755),
    "/etc/systemd/system/spark-osd-guard@.service": ("tools/devcycle/fleet/spark-osd-guard@.service",0o644),
    "/etc/systemd/system/spark-osd-guard-scan.service": ("tools/devcycle/fleet/spark-osd-guard-scan.service",0o644),
    "/etc/systemd/system/spark-osd-guard-scan.timer": ("tools/devcycle/fleet/spark-osd-guard-scan.timer",0o644),
    "/etc/udev/rules.d/71-spark-osd-guard.rules": ("tools/devcycle/fleet/71-spark-osd-guard.rules",0o644),
    "/etc/udev/rules.d/95-ceph-osd-lvm.rules": ("tools/devcycle/fleet/95-ceph-osd-lvm.rules",0o644),
    "/usr/local/sbin/ds4-ceph-warm": ("tools/devcycle/fleet/ds4_ceph_warm.sh",0o755),
    "/etc/systemd/system/ds4-ceph-warm-mount.service": ("tools/devcycle/fleet/ds4-ceph-warm-mount.service",0o644),
    "/etc/systemd/system/restart-resolved.service": ("tools/devcycle/fleet/restart-resolved.service",0o644),
    "/etc/systemd/timesyncd.conf.d/sparkpipe-hub.conf": ("tools/devcycle/fleet/timesyncd-sparkpipe-hub.conf",0o644),
    "/etc/systemd/system/sparkpipe-roce-qos.service": ("tools/fabric/sparkpipe-roce-qos.service",0o644),
    "/usr/local/sbin/sparkpipe_roce_qos.sh": ("tools/fabric/sparkpipe_roce_qos.sh",0o755),
    f"/etc/systemd/system/ceph-{CEPH_CLUSTER_FSID}.target.d/20-ds4-start-after-ssh.conf": ("tools/devcycle/fleet/ceph-cluster-start-after-ssh.conf",0o644),
    f"/etc/systemd/system/ceph-{CEPH_CLUSTER_FSID}@.service.d/20-ds4-start-after-ssh.conf": ("tools/devcycle/fleet/ceph-daemon-start-after-ssh.conf",0o644),
})
LINKED_RANK_ASSET_SOURCES = {
    "/etc/systemd/system/sparkpipe-hub-route.service": ("tools/devcycle/fleet/sparkpipe-hub-route.service",0o644),
    "/etc/systemd/system/sparkpipe-hub-route.service.d/10-retry.conf": ("tools/devcycle/fleet/sparkpipe-hub-route-retry.conf",0o644),
}
USER_SLICE = """[Slice]
MemoryHigh=100G
MemoryMax=108G
MemorySwapMax=0
"""
EARLYOOM_ARGS = "-m 2 -s 100 -r 5 --avoid sshd|systemd|init"
EARLYOOM_DEFAULT = f'EARLYOOM_ARGS="{EARLYOOM_ARGS}"\n'
NO_THRASH_SYSCTL = "vm.swappiness=10\n"
RECOVERY_SYSCTL = """kernel.panic=60
kernel.softlockup_panic=1
kernel.hung_task_panic=1
"""
DIRTY_CAP_SYSCTL = "vm.dirty_bytes=4294967296\nvm.dirty_background_bytes=536870912\n"
DIRTY_CAP_SETTINGS = {"vm.dirty_bytes":"4294967296","vm.dirty_background_bytes":"536870912"}
SYSTEMD_RECOVERY = """[Manager]
RuntimeWatchdogSec=10min
RebootWatchdogSec=5min
"""
EMERGENCY_SSH_CONFIG = """Port 2222
Protocol 2
ListenAddress 0.0.0.0
UsePAM no
PermitRootLogin prohibit-password
PasswordAuthentication no
KbdInteractiveAuthentication no
PubkeyAuthentication yes
AuthorizedKeysFile .ssh/authorized_keys
AllowUsers root sparkemerg
PidFile /run/sshd-spark-emergency.pid
PrintMotd no
UseDNS no
"""
EMERGENCY_SSH_SERVICE = """[Unit]
Description=Spark emergency public-key SSH on port 2222
After=network.target
Wants=network.target

[Service]
Type=simple
ExecStart=/usr/sbin/sshd -D -e -f /etc/ssh/sshd_config_spark_emergency
Restart=always
RestartSec=2s

[Install]
WantedBy=multi-user.target
"""


class BrickproofError(RuntimeError):
    pass


def run(argv: list[str],input_bytes: bytes | None = None,timeout: int = 120,check: bool = True,env: dict[str,str] | None = None) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(argv,input=input_bytes,stdin=None if input_bytes is not None else subprocess.DEVNULL,capture_output=True,timeout=timeout,env=env)
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).decode("utf-8",errors="replace").strip()
        raise BrickproofError(f"command failed ({result.returncode}): {' '.join(argv)}: {detail}")
    return(result)


def command(argv: list[str],timeout: int = 120,check: bool = True) -> str:
    return(run(argv,timeout=timeout,check=check).stdout.decode("utf-8",errors="replace").strip())


def atomic_write(path: Path,data: str,mode: int) -> bool:
    encoded = data.encode("utf-8")
    if path.exists() and path.read_bytes() == encoded and (path.stat().st_mode & 0o777) == mode:
        return(False)
    path.parent.mkdir(parents=True,exist_ok=True)
    handle,temp_name = tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
    try:
        with os.fdopen(handle,"wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temp_name,mode)
        os.replace(temp_name,path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return(True)


def shell_assignment(text: str,name: str) -> str:
    match = re.search(rf"^[ \t]*{re.escape(name)}[ \t]*=(.*)$",text,re.MULTILINE)
    if match is None:
        return("")
    values = shlex.split(match.group(1),comments=False,posix=True)
    return(values[0] if values else "")


def set_shell_assignment(text: str,name: str,value: str) -> str:
    line = f"{name}={shlex.quote(value)}"
    pattern = re.compile(rf"^[ \t]*{re.escape(name)}[ \t]*=.*$",re.MULTILINE)
    if pattern.search(text) is not None:
        return(pattern.sub(line,text,count=1))
    suffix = "" if text.endswith("\n") else "\n"
    return(f"{text}{suffix}{line}\n")


def public_key_material_is_valid(key_type: str,blob: str) -> bool:
    if not key_type.startswith(("ssh-","ecdsa-","sk-")):
        return(False)
    try:
        decoded = base64.b64decode(blob,validate=True)
    except (binascii.Error,ValueError):
        return(False)
    if len(decoded) < 4:
        return(False)
    name_length = int.from_bytes(decoded[:4],"big")
    if name_length == 0 or (4 + name_length) > len(decoded):
        return(False)
    return(decoded[4:4 + name_length].decode("ascii",errors="replace") == key_type)


def canonical_authorized_keys(text: str,required_key: str) -> str:
    required_tokens = required_key.split()
    if len(required_tokens) < 2 or not public_key_material_is_valid(required_tokens[0],required_tokens[1]):
        raise BrickproofError("fleet recovery public key is invalid")
    required_material = (required_tokens[0],required_tokens[1])
    repaired = re.sub(r"(?<=\S)n(?=ssh-(?:ed25519|rsa|dss|ecdsa-|sk-))","\n",text)
    rendered: list[str] = []
    seen: set[tuple[str,str]] = set()
    pending_type = ""
    required_present = False
    for raw_line in repaired.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            fields = shlex.split(line)
        except ValueError:
            fields = line.split()
        key_index = next((index for index,field in enumerate(fields) if field.startswith(("ssh-","ecdsa-","sk-"))),-1)
        if key_index >= 0:
            pending_type = fields[key_index] if key_index == 0 and len(fields) == 1 else ""
            if (key_index + 1) >= len(fields):
                continue
            material = (fields[key_index],fields[key_index + 1])
            if not public_key_material_is_valid(*material) or material in seen:
                continue
            if material == required_material:
                if key_index > 0:
                    continue
                required_present = True
            rendered.append(line)
            seen.add(material)
            continue
        if pending_type and fields and public_key_material_is_valid(pending_type,fields[0]):
            material = (pending_type,fields[0])
            if material != required_material and material not in seen:
                rendered.append(f"{material[0]} {material[1]}")
                seen.add(material)
        pending_type = ""
    if not required_present:
        rendered.append(required_key.strip())
    return("\n".join(rendered) + "\n")


def recovery_ip_token(recovery_network: object) -> str:
    if not isinstance(recovery_network,dict):
        raise BrickproofError("recovery network is invalid")
    names = ("address","gateway","netmask","node_id","interface")
    values = [str(recovery_network.get(name,"")) for name in names]
    if any(not value for value in values):
        raise BrickproofError("recovery network is incomplete")
    return(f"ip={values[0]}::{values[1]}:{values[2]}:{values[3]}:{values[4]}:none")


def canonical_grub(text: str,recovery_network: object) -> str:
    cmdline = shell_assignment(text,"GRUB_CMDLINE_LINUX_DEFAULT")
    tokens = shlex.split(cmdline)
    tokens = [token for token in tokens if not token.startswith(("ip=","fsck.mode=","fsck.repair=","console=ttyS0"))]
    if "console=tty0" not in tokens:
        tokens.append("console=tty0")
    tokens.extend(("console=ttyS0,921600",recovery_ip_token(recovery_network),"fsck.mode=skip","fsck.repair=no"))
    text = set_shell_assignment(text,"GRUB_CMDLINE_LINUX_DEFAULT",shlex.join(tokens))
    return(set_shell_assignment(text,"GRUB_DEFAULT","0"))


def parse_efi_boot_entries(text: str) -> tuple[list[str],dict[str,str]]:
    order_match = re.search(r"^BootOrder:\s*(.*)$",text,re.MULTILINE)
    if order_match is None:
        raise BrickproofError("efibootmgr output has no BootOrder")
    order = [item.strip().upper() for item in order_match.group(1).split(",") if item.strip()]
    entries = {}
    for match in re.finditer(r"^Boot([0-9A-Fa-f]{4})\*?\s+([^\r\n]+)$",text,re.MULTILINE):
        entries[match.group(1).upper()] = match.group(2).strip()
    if not order or any(item not in entries for item in order):
        raise BrickproofError("efibootmgr BootOrder references a missing entry")
    return((order,entries))


def linux_efi_boot_entry(entries: dict[str,str]) -> str:
    candidates = []
    for entry,label in entries.items():
        normalized = label.strip().casefold()
        if any(normalized == name or normalized.startswith(f"{name} ") or normalized.startswith(f"{name}\t") for name in ("ubuntu","dgx os")):
            candidates.append(entry)
    if len(candidates) != 1:
        raise BrickproofError(f"expected one Linux EFI boot entry, found {len(candidates)}")
    return(candidates[0])


def pxe_ipv4_efi_boot_entries(order: list[str],entries: dict[str,str]) -> list[str]:
    candidates = [entry for entry in order if "pxe ipv4" in entries[entry].casefold()]
    if not candidates:
        raise BrickproofError("expected at least one PXE IPv4 EFI boot entry")
    return(candidates)


def desired_efi_boot_order(text: str) -> list[str]:
    order,entries = parse_efi_boot_entries(text)
    linux_entry = linux_efi_boot_entry(entries)
    pxe_entries = pxe_ipv4_efi_boot_entries(order,entries)
    leading = [*pxe_entries,linux_entry]
    return([*leading,*[entry for entry in order if entry not in leading]])


def package_installed(name: str) -> bool:
    result = run(["dpkg-query","-W","-f=${Status}",name],check=False)
    return(result.returncode == 0 and b"install ok installed" in result.stdout)


def kernel_release_from_vmlinuz(path: Path) -> str:
    try:
        target = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise BrickproofError(f"missing boot kernel link: {path}") from error
    prefix = "vmlinuz-"
    if not target.name.startswith(prefix) or len(target.name) == len(prefix):
        raise BrickproofError(f"cannot derive boot kernel release from {target}")
    return(target.name[len(prefix):])


def boot_kernel_release() -> str:
    return(kernel_release_from_vmlinuz(Path("/boot/vmlinuz")))


def running_kernel_release() -> str:
    return(os.uname().release)


def module_tree_present(release: str) -> bool:
    return(Path(f"/lib/modules/{release}").is_dir())


def kernel_module_loaded(module: str) -> bool:
    return((Path("/sys/module") / module).is_dir())


def directory_owner(path: Path) -> str:
    return(f"{path.stat().st_uid}:{path.stat().st_gid}" if path.is_dir() else "missing")


def kernel_releases() -> tuple[str,...]:
    return(tuple(dict.fromkeys((running_kernel_release(),boot_kernel_release()))))


def kernel_packages(kernel_release: str) -> tuple[str,...]:
    return(f"linux-image-{kernel_release}",f"linux-modules-{kernel_release}")


def required_packages(kernel_release: str) -> tuple[str,...]:
    return("earlyoom","dropbear-initramfs","efibootmgr",*FLEET_TOOL_PACKAGES,*kernel_packages(kernel_release))


def package_verification_lines(name: str) -> tuple[str,...]:
    result = run(["dpkg","-V",name],check=False)
    text = (result.stdout + result.stderr).decode("utf-8",errors="replace")
    return(tuple(line for line in text.splitlines() if line.strip()))


def install_packages() -> None:
    packages = list(required_packages(os.uname().release))
    for release in kernel_releases():
        packages.extend(kernel_packages(release))
    packages = list(dict.fromkeys(packages))
    missing = [name for name in packages if not package_installed(name)]
    environment = {**os.environ,"DEBIAN_FRONTEND":"noninteractive"}
    if missing:
        run(["apt-get","install","-y","-o","Dpkg::Options::=--force-confdef","-o","Dpkg::Options::=--force-confold",*missing],timeout=900,env=environment)
    damaged = [name for release in kernel_releases() for name in kernel_packages(release) if package_verification_lines(name)]
    damaged = list(dict.fromkeys(damaged))
    if damaged:
        run(["apt-get","install","--reinstall","-y","-o","Dpkg::Options::=--force-confdef","-o","Dpkg::Options::=--force-confold",*damaged],timeout=900,env=environment)


def rebuild_initramfs() -> None:
    for release in kernel_releases():
        mode = "-u" if Path(f"/boot/initrd.img-{release}").exists() else "-c"
        run(["update-initramfs",mode,"-k",release],timeout=600)


def install_required_kernel_modules() -> None:
    atomic_write(Path("/etc/modules-load.d/90-ds4-network.conf"),KERNEL_MODULES_LOAD,0o644)
    for module in REQUIRED_KERNEL_MODULES:
        run(["modprobe",module])


def fleet_alias_lines() -> tuple[str,...]:
    lines = []
    for rank in range(MAX_SPARK_RANK + 1):
        node = f"spark{rank:x}"
        lines.append(f"{SWITCHED_PREFIX}.{SWITCHED_ADDRESS_BASE + rank} {node} {node}-fabric")
        lines.append(f"{MANAGEMENT_PREFIX}.{MANAGEMENT_ADDRESS_BASE + rank} {node}-mgmt")
    return(tuple(lines))


def fleet_aliases() -> set[str]:
    return({alias for line in fleet_alias_lines() for alias in line.split()[1:]})


def canonical_hosts(text: str,removed_aliases: set[str]) -> str:
    output = []
    managed_end = ""
    aliases_to_replace = fleet_aliases() | removed_aliases
    for line in text.splitlines():
        stripped = line.strip()
        if managed_end:
            if stripped == managed_end:
                managed_end = ""
            continue
        if stripped in LEGACY_HOSTS_BLOCKS:
            managed_end = LEGACY_HOSTS_BLOCKS[stripped]
            continue
        fields = line.split()
        if fields and not fields[0].startswith("#"):
            aliases = [alias for alias in fields[1:] if alias not in aliases_to_replace]
            if aliases:
                output.append(f"{fields[0]}\t{' '.join(aliases)}")
            elif len(fields) == 1:
                output.append(line)
            continue
        output.append(line)
    if managed_end:
        raise BrickproofError(f"unterminated hosts block: expected {managed_end}")
    rendered = "\n".join(output).rstrip()
    block = "\n".join((HOSTS_BLOCK_BEGIN,*fleet_alias_lines(),HOSTS_BLOCK_END))
    return(f"{rendered}\n\n{block}\n" if rendered else f"{block}\n")


def direct_pair_linked(rank: int) -> bool:
    linked = False
    for link in MESH_LINK_MAP.split(","):
        match = re.fullmatch(r"([0-9]+)-([0-9]+):([0-9]+)",link)
        if match is None or int(match.group(1)) ^ 1 != int(match.group(2)) or int(match.group(1)) > int(match.group(2)):
            raise BrickproofError(f"mesh link {link!r} does not join a rank with its rank^1 partner")
        linked = linked or rank in (int(match.group(1)),int(match.group(2)))
    return(linked)


def spark_rank(node: str) -> int:
    match = re.fullmatch(r"spark([0-9a-f])",node.casefold())
    if match is None:
        raise BrickproofError(f"invalid Spark node id: {node}")
    rank = int(match.group(1),16)
    if rank > MAX_SPARK_RANK:
        raise BrickproofError(f"Spark rank {rank} exceeds the fleet")
    return(rank)


def install_node_identity(payload: dict[str,object]) -> bool:
    hostname = str(payload["recovery_network"]["node_id"])
    rank = spark_rank(hostname)
    previous_hostname = command(["hostname","-s"],check=False)
    changed = previous_hostname != hostname
    if changed:
        run(["hostnamectl","set-hostname",hostname])
    atomic_write(Path("/etc/hostname"),f"{hostname}\n",0o644)
    rank_changed = atomic_write(NODE_RANK_PATH,f"{rank}\n",0o644)
    hosts = Path("/etc/hosts")
    hosts_changed = atomic_write(hosts,canonical_hosts(hosts.read_text(),{hostname,previous_hostname}),0o644)
    return(changed or hosts_changed or rank_changed)


def install_sshd_config() -> None:
    target = Path("/etc/ssh/sshd_config_spark_emergency")
    handle,temp_name = tempfile.mkstemp(prefix="sshd-spark-emergency.",dir="/etc/ssh")
    try:
        with os.fdopen(handle,"w",encoding="utf-8") as output:
            output.write(EMERGENCY_SSH_CONFIG)
        os.chmod(temp_name,0o644)
        run(["/usr/sbin/sshd","-t","-f",temp_name])
        os.replace(temp_name,target)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def install_recovery_keys(public_key: str) -> bool:
    changed = False
    root_keys = Path("/root/.ssh/authorized_keys")
    root_keys.parent.mkdir(parents=True,exist_ok=True)
    root_text = root_keys.read_text() if root_keys.exists() else ""
    changed |= atomic_write(root_keys,canonical_authorized_keys(root_text,public_key),0o600)
    os.chmod(root_keys.parent,0o700)
    dropbear_keys = Path("/etc/dropbear/initramfs/authorized_keys")
    dropbear_keys.parent.mkdir(parents=True,exist_ok=True)
    dropbear_text = dropbear_keys.read_text() if dropbear_keys.exists() else ""
    changed |= atomic_write(dropbear_keys,canonical_authorized_keys(dropbear_text,public_key),0o600)
    return(changed)


def node_layout_roots(text: str) -> list[str]:
    return(sorted(json.loads(text)["roots"].values()))


def install_node_layout(node_id: str,roots: object,home_root: Path = Path("/home")) -> list[str]:
    if not isinstance(roots,list) or not roots or any(not isinstance(root,str) or not re.fullmatch(r"[a-z][a-z0-9_]*",root) for root in roots):
        raise BrickproofError(f"payload node layout roots are invalid: {roots!r}")
    home = home_root / node_id
    owner = home.stat()
    created = []
    for root in roots:
        path = home / root
        if path.exists() or path.is_symlink():
            continue
        path.mkdir(mode=0o775)
        os.chown(path,owner.st_uid,owner.st_gid)
        os.chmod(path,0o775)
        created.append(str(path))
    return(created)


def heal_time_sync() -> bool:
    if command(["timedatectl","show-timesync","-p","ServerName","--value"],check=False) in NTP_SERVERS:
        return(False)
    run(["systemctl","restart","systemd-timesyncd.service"])
    return(True)


def install_assets(assets: object) -> bool:
    changed = False
    if not isinstance(assets,dict):
        raise BrickproofError("payload assets are invalid")
    for path_text,specification in assets.items():
        if not isinstance(specification,dict):
            raise BrickproofError(f"invalid asset specification for {path_text}")
        changed |= atomic_write(Path(path_text),str(specification["text"]),int(specification["mode"]))
    return(changed)


def install_grub_policy(recovery_network: object) -> bool:
    path = Path("/etc/default/grub")
    before = path.read_text()
    changed = atomic_write(path,canonical_grub(before,recovery_network),0o644)
    legacy = Path("/etc/grub.d/40_ds4_fastboot")
    if legacy.exists():
        legacy.unlink()
        changed = True
    if changed:
        run(["update-grub"],timeout=300)
    return(changed)


def install_efi_boot_order() -> bool:
    before = command(["efibootmgr","-v"])
    order,_ = parse_efi_boot_entries(before)
    desired = desired_efi_boot_order(before)
    if order == desired:
        return(False)
    run(["efibootmgr","-o",",".join(desired)])
    after = command(["efibootmgr","-v"])
    if desired_efi_boot_order(after) != desired or parse_efi_boot_entries(after)[0] != desired:
        raise BrickproofError("EFI boot order verification failed")
    return(True)


def persistent_unit_links(unit: str,root: Path = Path("/etc/systemd/system")) -> list[Path]:
    return(sorted(
        path for path in root.rglob(unit)
        if path.is_symlink() and path.parent.name.endswith((".wants",".requires"))
    ))


def disable_persistent_unit(unit: str) -> None:
    run(["systemctl","disable",unit],check=False)
    for path in persistent_unit_links(unit):
        path.unlink()


def ceph_unit_file_states() -> dict[str,str]:
    output = command([
        "systemctl","list-unit-files","--no-legend",
        "ceph*.service","ceph*.target",
    ],check=False)
    states = {}
    for row in output.splitlines():
        fields = row.split()
        if len(fields) >= 2 and fields[0].startswith("ceph"):
            states[fields[0]] = fields[1]
    return(states)


def ceph_fsids(root: Path = CEPH_STATE_ROOT,systemd_root: Path = Path("/etc/systemd/system")) -> list[str]:
    if not root.is_dir():
        return([])
    pattern = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
    clusters = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or not pattern.fullmatch(path.name):
            continue
        if (systemd_root / f"ceph-{path.name}.target").exists() or ceph_daemons(path.name,root):
            clusters.append(path.name)
    return(clusters)


def ceph_enabled_instances(fsid: str,systemd_root: Path = Path("/etc/systemd/system")) -> list[str]:
    prefix = f"ceph-{fsid}@"
    names = set()
    for kind in ("wants","requires"):
        for link in systemd_root.glob(f"*.{kind}/{prefix}*.service"):
            if link.is_symlink():
                names.add(link.name[len(prefix):-len(".service")])
    return(sorted(names))


def ceph_daemons(fsid: str,root: Path = CEPH_STATE_ROOT) -> list[str]:
    return(sorted(path.name for path in (root / fsid).iterdir() if path.is_dir() and path.name.startswith(CEPH_DAEMON_PREFIXES)))


def ceph_stray_units(fsid: str,daemons: list[str],enabled: list[str]) -> list[str]:
    return(sorted(f"ceph-{fsid}@{instance}.service" for instance in enabled if instance and instance not in daemons))


def remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def retire_path(path: Path,backup_root: Path) -> bool:
    if not (path.exists() or path.is_symlink()):
        return(False)
    target = backup_root / path.relative_to("/")
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.move(str(path),str(target))
    return(True)


def mask_unit_file(unit: str,root: Path = Path("/etc/systemd/system")) -> None:
    path = root / unit
    if path.is_symlink() and path.readlink() == Path("/dev/null"):
        return
    if path.exists() or path.is_symlink():
        path.unlink()
    path.symlink_to("/dev/null")


def stale_snap_units(root: Path = Path("/etc/systemd/system")) -> list[str]:
    stale = []
    for unit in sorted(root.glob("snap-*.mount")):
        what = next((line.split("=",1)[1].strip() for line in read_optional(unit).splitlines() if line.startswith("What=")),"")
        if what.startswith("/var/lib/snapd/snaps/") and not Path(what).exists():
            stale.append(unit.name)
    return(stale)


def retire_stale_snap_units(root: Path = Path("/etc/systemd/system"),backup_root: Path | None = None) -> list[str]:
    backup_root = backup_root or backup_directory()
    retired = []
    for name in stale_snap_units(root):
        run(["systemctl","disable",name],check=False)
        for link in sorted(root.glob(f"*.wants/{name}")) + sorted(root.glob(f"*.requires/{name}")):
            retire_path(link,backup_root)
        retire_path(root / name,backup_root)
        retired.append(name)
    if retired:
        run(["systemctl","daemon-reload"])
        run(["systemctl","reset-failed",*retired],check=False)
    return(retired)


def reset_vanished_failed_units() -> list[str]:
    vanished = []
    for line in command(["systemctl","list-units","--failed","--all","--plain","--no-legend"],check=False).splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[1] == "not-found":
            vanished.append(fields[0])
    if vanished:
        run(["systemctl","reset-failed",*vanished],check=False)
    return(vanished)


def retired_paths(systemd_root: Path = Path("/etc/systemd/system")) -> list[Path]:
    return([systemd_root / "ceph-osd-retry.service",Path("/usr/local/sbin/ceph-osd-retry-start")])


def backup_directory() -> Path:
    return(Path("/var/backups/ds4-fleet") / datetime.datetime.now(datetime.timezone.utc).strftime("brickproof-%Y%m%dT%H%M%SZ"))


def install_ceph_policy(
        systemd_root: Path = Path("/etc/systemd/system"),
        state_root: Path = CEPH_STATE_ROOT,
        log_root: Path = CEPH_LOG_ROOT,
        marker: Path = LEGACY_CEPH_WARM_STORAGE_MARKER,
        backup_root: Path | None = None) -> list[str]:
    backup_root = backup_root or backup_directory()
    changes = []
    if marker.exists():
        marker.unlink()
        changes.append(f"removed {marker}")
    for unit in OBSOLETE_CEPH_STARTUP_UNITS:
        disable_persistent_unit(unit)
        if retire_path(systemd_root / unit,backup_root):
            changes.append(f"retired {unit}")
    disable_persistent_unit("ceph-osd-retry.service")
    for path in retired_paths(systemd_root):
        if retire_path(path,backup_root):
            changes.append(f"retired {path}")
    for unit in CEPH_MASK_UNITS:
        mask_unit_file(unit,systemd_root)
    run(["systemctl","daemon-reload"])
    run(["systemctl","reset-failed","ceph-osd-retry.service",*OBSOLETE_CEPH_STARTUP_UNITS],check=False)
    for fsid in ceph_fsids(state_root,systemd_root):
        log_dir = log_root / fsid
        log_dir.mkdir(parents=True,exist_ok=True)
        os.chown(log_dir,CEPH_UID,CEPH_UID)
        os.chmod(log_dir,0o770)
        daemons = ceph_daemons(fsid,state_root)
        for unit in ceph_stray_units(fsid,daemons,ceph_enabled_instances(fsid,systemd_root)):
            disable_persistent_unit(unit)
            changes.append(f"disabled stray {unit}")
        required = [f"ceph-{fsid}@{daemon}.service" for daemon in daemons if daemon.startswith(CEPH_REQUIRED_DAEMON_PREFIXES)]
        run(["systemctl","enable","ceph.target",f"ceph-{fsid}.target",*required])
    return(changes)


def enable_policy_services(rank: int,node_id: str = "") -> None:
    run(["systemctl","daemon-reload"])
    run(["sysctl","--system"],timeout=180)
    run(["udevadm","control","--reload"])
    run(["systemctl","enable","earlyoom.service"])
    run(["systemctl","restart","earlyoom.service"])
    run(["systemctl","enable","ssh.service"])
    run(["systemctl","enable","ssh-emergency.service"])
    run(["systemctl","restart","ssh-emergency.service"])
    disable_persistent_unit("sparkpipe-fsck-health.service")
    run(["systemctl","enable","--now","sparkpipe-fsck-health.timer"])
    run(["systemctl","enable","serial-getty@ttyS0.service"])
    disable_persistent_unit("ds4-switched-fabric.service")
    disable_persistent_unit("ds4-direct-pair-fabric.service")
    run(["systemctl","enable","--now","ds4-switched-fabric.timer"])
    if direct_pair_linked(rank):
        run(["systemctl","enable","--now","ds4-direct-pair-fabric.timer"])
    else:
        run(["systemctl","disable","--now","ds4-direct-pair-fabric.timer"])
    if rank == HUB_GATEWAY_RANK:
        disable_persistent_unit("sparkpipe-hub-route.service")
    else:
        run(["systemctl","enable","sparkpipe-hub-route.service"])
    run(["systemctl","enable","spark-firewall.service"])
    run(["systemctl","enable","--now","spark-osd-guard-scan.timer"])
    run(["systemctl","enable","restart-resolved.service","ds4-ceph-warm-mount.service","sparkpipe-roce-qos.service"])
    if node_id:
        run(["loginctl","enable-linger",node_id])
    run(["systemctl","restart","systemd-timesyncd.service"])
    run(["systemctl","unmask","fstrim.service"])
    run(["systemctl","enable","--now","fstrim.timer"])
    run(["systemctl","reset-failed","fstrim.service","fstrim.timer"],check=False)
    run(["systemctl","reset-failed","user@0.service"],check=False)
    run(["systemctl","set-default","multi-user.target"])
    run(["systemctl","set-property","--runtime","user-1000.slice","MemoryHigh=100G","MemoryMax=108G","MemorySwapMax=0"])


def has_configured_remote_storage() -> list[str]:
    configured = []
    rbdmap = read_optional(Path("/etc/ceph/rbdmap"))
    if any(line.strip() and not line.lstrip().startswith("#") for line in rbdmap.splitlines()):
        configured.append("rbdmap")
    fstab = read_optional(Path("/etc/fstab"))
    for line in fstab.splitlines():
        fields = line.split()
        if line.strip() and not line.lstrip().startswith("#") and len(fields) >= 4:
            if fields[2] in ("nfs","nfs4","cifs","ceph") or "_netdev" in fields[3].split(","):
                configured.append("remote-fstab")
    iscsi = Path("/etc/iscsi/nodes")
    if iscsi.exists() and any(path.is_file() for path in iscsi.rglob("*")):
        configured.append("iscsi")
    discovery = read_optional(Path("/etc/nvme/discovery.conf"))
    if any(line.strip() and not line.lstrip().startswith("#") for line in discovery.splitlines()):
        configured.append("nvme-of")
    return(sorted(set(configured)))


def decouple_optional_boot_work() -> None:
    configured = has_configured_remote_storage()
    if configured:
        raise BrickproofError(f"remote storage configuration exists; refusing to mask initiators: {configured}")
    for unit in OPTIONAL_BOOT_UNITS:
        run(["systemctl","disable",unit],check=False)
        run(["systemctl","mask",unit],check=False)
    atomic_write(Path("/etc/cloud/cloud-init.disabled"),"commissioned by ds4_spark_brickproof\n",0o644)


def remote_apply(payload: dict[str,object]) -> dict[str,object]:
    if os.geteuid() != 0:
        raise BrickproofError("remote apply must run as root")
    if not FIREWALL_RULESET.is_file():
        raise BrickproofError(f"{FIREWALL_RULESET} is missing; place the host firewall ruleset from the private coordination store before apply")
    memory_current = int(command(["systemctl","show","user-1000.slice","-p","MemoryCurrent","--value"]) or "0")
    if memory_current >= 108 * 1024 * 1024 * 1024:
        raise BrickproofError(f"user-1000.slice already uses {memory_current} bytes; refusing a 108G cap")
    install_packages()
    install_required_kernel_modules()
    identity_changed = install_node_identity(payload)
    rank = spark_rank(str(payload["recovery_network"]["node_id"]))
    install_assets(payload["assets"])
    if rank != HUB_GATEWAY_RANK:
        install_assets(payload["linked_rank_assets"])
    layout_created = install_node_layout(str(payload["recovery_network"]["node_id"]),payload["node_layout_roots"])
    time_sync_restarted = heal_time_sync()
    run([NETWORK_MANAGER_RUNTIME,"--network-manager"],timeout=120)
    atomic_write(Path("/etc/systemd/system/user-1000.slice.d/20-ds4-brickproof.conf"),USER_SLICE,0o644)
    atomic_write(Path("/etc/default/earlyoom"),EARLYOOM_DEFAULT,0o644)
    atomic_write(Path("/etc/sysctl.d/90-sparkpipe-no-thrash.conf"),NO_THRASH_SYSCTL,0o644)
    atomic_write(Path("/etc/sysctl.d/99-spark-recovery.conf"),RECOVERY_SYSCTL,0o644)
    atomic_write(Path("/etc/sysctl.d/99-dirty-cap.conf"),DIRTY_CAP_SYSCTL,0o644)
    atomic_write(Path("/etc/systemd/system.conf.d/90-ds4-recovery.conf"),SYSTEMD_RECOVERY,0o644)
    install_sshd_config()
    atomic_write(Path("/etc/systemd/system/ssh-emergency.service"),EMERGENCY_SSH_SERVICE,0o644)
    for unit in ("ds4-switched-fabric.service","ds4-direct-pair-fabric.service"):
        legacy_timeout = Path(f"/etc/systemd/system/{unit}.d/10-boot-timeout.conf")
        if legacy_timeout.exists():
            legacy_timeout.unlink()
    keys_changed = install_recovery_keys(str(payload["fleet_public_key"]))
    grub_changed = install_grub_policy(payload["recovery_network"])
    efi_boot_order_changed = install_efi_boot_order()
    rebuild_initramfs()
    decouple_optional_boot_work()
    ceph_changes = install_ceph_policy()
    snap_units_retired = retire_stale_snap_units() + reset_vanished_failed_units()
    enable_policy_services(rank,str(payload["recovery_network"]["node_id"]))
    run([DMA_GUARD_RUNTIME,"install","--notify",HUB_NOTIFY],timeout=120)
    run(["systemctl","start","logrotate.service"])
    return({"ceph_changes":ceph_changes,"layout_created":layout_created,"snap_units_retired":snap_units_retired,"time_sync_restarted":time_sync_restarted,"efi_boot_order_changed":efi_boot_order_changed,"grub_changed":grub_changed,"identity_changed":identity_changed,"keys_changed":keys_changed,"memory_current_before":memory_current,"source_commit":payload["source_commit"]})


def read_optional(path: Path) -> str:
    try:
        return(path.read_text())
    except (FileNotFoundError,PermissionError):
        return("")


def service_value(unit: str,property_name: str) -> str:
    return(command(["systemctl","show",unit,"-p",property_name,"--value"],check=False))


def is_enabled(unit: str) -> str:
    return(command(["systemctl","is-enabled",unit],check=False))


def audit_fleet_hardware(node_id: str) -> tuple[list[str],dict[str,object]]:
    failures = []
    observations: dict[str,object] = {}
    links = [(SWITCHED_INTERFACE,SWITCHED_SPEED_MBIT)]
    if direct_pair_linked(spark_rank(node_id)):
        links.append((DIRECT_INTERFACE,DIRECT_SPEED_MBIT))
    for interface,expected_speed in links:
        root = Path("/sys/class/net") / interface
        link = {
            "carrier":read_optional(root / "carrier").strip(),
            "duplex":read_optional(root / "duplex").strip(),
            "speed_mbit":read_optional(root / "speed").strip(),
        }
        observations[f"link:{interface}"] = link
        if link != {"carrier":"1","duplex":"full","speed_mbit":expected_speed}:
            failures.append(f"fabric-link:{interface}")
    gpu = run(["nvidia-smi","--query-gpu=name,temperature.gpu","--format=csv,noheader,nounits"],timeout=30,check=False)
    observations["gpu"] = gpu.stdout.decode("utf-8",errors="replace").strip()
    if gpu.returncode != 0 or not str(observations["gpu"]).startswith("NVIDIA GB10,"):
        failures.append("gpu-unhealthy")
    tailscale = run(["tailscale","status","--json"],timeout=30,check=False)
    try:
        status = json.loads(tailscale.stdout.decode("utf-8"))
        self_status = status.get("Self",{})
        observations["tailscale"] = {
            "backend":status.get("BackendState"),
            "online":self_status.get("Online"),
            "tags":self_status.get("Tags",[]),
        }
    except json.JSONDecodeError:
        observations["tailscale"] = {}
    if (observations["tailscale"].get("backend") != "Running" or
            observations["tailscale"].get("online") is not True or
            "tag:spark" not in observations["tailscale"].get("tags",[])):
        failures.append("tailscale-not-ready")
    return(failures,observations)


def nm_sections(config: str) -> dict[str,dict[str,str]]:
    sections: dict[str,dict[str,str]] = {}
    current = None
    for raw in config.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections.setdefault(current,{})
        elif current is not None and "=" in line:
            key,value = line.split("=",1)
            sections[current][key.strip()] = value.strip()
    return(sections)


def unmanaged_interfaces(config: str) -> set[str]:
    devices = nm_sections(config).get("keyfile",{}).get("unmanaged-devices","")
    return({spec.split(":",1)[1] for spec in (item.strip() for item in devices.split(";")) if spec.startswith("interface-name:")})


def uplink_assume_disabled(config: str) -> bool:
    for name,values in nm_sections(config).items():
        matches = [item.strip() for item in values.get("match-device","").split(",")]
        if name.startswith("device") and f"interface-name:{MANAGEMENT_INTERFACE}" in matches and values.get("keep-configuration") == "no":
            return(True)
    return(False)


def hosts_entries(text: str) -> set[tuple[str,tuple[str,...]]]:
    entries = set()
    for line in text.splitlines():
        fields = line.split("#",1)[0].split()
        if len(fields) >= 2:
            entries.add((fields[0],tuple(sorted(fields[1:]))))
    return(entries)


def uplink_default_route_present(default_routes: list[str]) -> bool:
    for line in default_routes:
        tokens = line.split()
        if tokens[:5] == ["default","via",UPLINK_GATEWAY,"dev",MANAGEMENT_INTERFACE] and "metric" in tokens[:-1] and tokens[tokens.index("metric") + 1] == UPLINK_METRIC:
            return(True)
    return(False)


def dma_guard_mode(state_path: Path = DMA_GUARD_STATE) -> str:
    starts = re.findall(r"event=start mode=([a-z-]+)",command(["journalctl","-b","-u","spark-dma-guard.service","-o","cat","--no-pager"],check=False))
    if starts:
        return(starts[-1])
    boot_id = read_optional(Path("/proc/sys/kernel/random/boot_id")).strip()
    try:
        boots = json.loads(state_path.read_text()).get("boots",[])
    except (OSError,json.JSONDecodeError):
        return("")
    current = [boot for boot in boots if boot.get("id") == boot_id]
    return("armed" if current and current[-1].get("armed") is True else "")


def asset_drift(assets: dict[str,dict[str,object]]) -> list[str]:
    drifted = []
    for path_text,specification in sorted(assets.items()):
        path = Path(path_text)
        try:
            same = path.read_bytes() == str(specification["text"]).encode("utf-8") and (path.stat().st_mode & 0o777) == int(specification["mode"])
        except OSError:
            same = False
        if not same:
            drifted.append(path_text)
    return(drifted)


def remote_audit(payload: dict[str,object]) -> dict[str,object]:
    public_key = str(payload["fleet_public_key"])
    failures = []
    observations: dict[str,object] = {}
    required_module_state: dict[str,object] = {}
    node_id = str(payload["recovery_network"]["node_id"])
    rank = spark_rank(node_id)
    observations["swappiness"] = command(["sysctl","-n","vm.swappiness"],check=False)
    observations["panic"] = command(["sysctl","-n","kernel.panic"],check=False)
    running_kernel = running_kernel_release()
    try:
        boot_kernel = boot_kernel_release()
    except BrickproofError as error:
        boot_kernel = ""
        failures.append(str(error))
    observations["running_kernel"] = running_kernel
    observations["boot_kernel"] = boot_kernel
    kernel_state: dict[str,object] = {}
    for release in dict.fromkeys((running_kernel,boot_kernel)):
        if not release:
            continue
        release_state: dict[str,object] = {"module_tree":module_tree_present(release)}
        if not release_state["module_tree"]:
            failures.append(f"missing-module-tree:{release}")
        for package in kernel_packages(release):
            installed = package_installed(package)
            verification_lines = package_verification_lines(package) if installed else ()
            release_state[package] = {"installed":installed,"verification_lines":len(verification_lines)}
            if not installed:
                failures.append(f"missing-package:{package}")
            elif verification_lines:
                failures.append(f"damaged-package:{package}:{len(verification_lines)}")
        kernel_state[release] = release_state
    observations["kernel_releases"] = kernel_state
    modules_load = read_optional(Path("/etc/modules-load.d/90-ds4-network.conf"))
    if modules_load != KERNEL_MODULES_LOAD:
        failures.append("kernel-modules-load-config")
    for module in REQUIRED_KERNEL_MODULES:
        available = command(["modinfo","-F","filename",module],check=False)
        loaded = kernel_module_loaded(module)
        required_module_state[module] = {"available":available,"loaded":loaded}
        if not available:
            failures.append(f"kernel-module-unavailable:{module}")
        if not loaded:
            failures.append(f"kernel-module-not-loaded:{module}")
    observations["required_kernel_modules"] = required_module_state
    observations["default_qdisc"] = command(["sysctl","-n","net.core.default_qdisc"],check=False)
    if observations["default_qdisc"] != "fq_codel":
        failures.append(f"default-qdisc:{observations['default_qdisc']}")
    observations["earlyoom_process"] = command(["pgrep","-af","earlyoom"],check=False)
    if not str(observations["earlyoom_process"]).endswith(EARLYOOM_ARGS):
        failures.append("earlyoom-runtime-args")
    for unit in ("earlyoom.service","ssh-emergency.service"):
        state = service_value(unit,"ActiveState")
        observations[f"{unit}.active"] = state
        if state != "active":
            failures.append(f"{unit}:{state}")
    expected_unit_states = {unit:"enabled" for unit in (
        "serial-getty@ttyS0.service","sparkpipe-fsck-health.timer","ds4-switched-fabric.timer","ds4-direct-pair-fabric.timer",
        "ssh.service","spark-dma-guard.timer","spark-osd-guard-scan.timer","restart-resolved.service","ds4-ceph-warm-mount.service",
        "sparkpipe-hub-route.service",
    )}
    if not direct_pair_linked(rank):
        expected_unit_states["ds4-direct-pair-fabric.timer"] = "disabled"
    if rank == HUB_GATEWAY_RANK:
        del expected_unit_states["sparkpipe-hub-route.service"]
    for unit,expected in expected_unit_states.items():
        state = is_enabled(unit)
        observations[f"{unit}.enabled"] = state
        if state != expected:
            failures.append(f"{unit}:not-{expected}")
    fstrim_service = is_enabled("fstrim.service")
    fstrim_timer = is_enabled("fstrim.timer")
    observations["fstrim.service.enabled"] = fstrim_service
    observations["fstrim.timer.enabled"] = fstrim_timer
    if fstrim_service != "static":
        failures.append(f"fstrim.service:{fstrim_service}")
    if fstrim_timer != "enabled":
        failures.append(f"fstrim.timer:{fstrim_timer}")
    cmdline_tokens = shlex.split(read_optional(Path("/proc/cmdline")))
    runtime_masks = {token.split("=",1)[1] for token in cmdline_tokens if token.startswith("systemd.mask=")}
    for unit in ("ds4-switched-fabric.service","ds4-direct-pair-fabric.service","sparkpipe-fsck-health.service"):
        state = is_enabled(unit)
        links = [str(path) for path in persistent_unit_links(unit)]
        observations[f"{unit}.enabled"] = state
        observations[f"{unit}.persistent_links"] = links
        rescue_masked = state == "masked-runtime" and unit in runtime_masks
        if state not in ("disabled","static") and not rescue_masked:
            failures.append(f"boot-critical-link:{unit}={state}")
        if links:
            failures.append(f"boot-critical-links:{unit}")
    for unit in OPTIONAL_BOOT_UNITS:
        state = is_enabled(unit)
        observations[f"{unit}.enabled"] = state
        if state != "masked":
            failures.append(f"optional-boot-unit:{unit}={state}")
    for unit in ("ds4-switched-fabric.service","ds4-direct-pair-fabric.service"):
        before = service_value(unit,"Before")
        timeout = service_value(unit,"TimeoutStartUSec")
        installed = read_optional(Path("/etc/systemd/system") / unit)
        rescue_masked = unit in runtime_masks
        observations[f"{unit}.before"] = before
        observations[f"{unit}.timeout"] = timeout
        observations[f"{unit}.rescue_masked"] = rescue_masked
        if "network-online.target" in installed or "multi-user.target" in installed:
            failures.append(f"fabric-boot-order:{unit}")
        if "TimeoutStartSec=60" not in installed:
            failures.append(f"fabric-timeout:{unit}={timeout}")
    firewall_timeout = service_value("spark-firewall.service","TimeoutStartUSec")
    observations["spark-firewall.timeout"] = firewall_timeout
    if firewall_timeout not in ("15s","15sec"):
        failures.append(f"spark-firewall-timeout={firewall_timeout}")
    expected_limits = {"MemoryHigh":str(100 * 1024**3),"MemoryMax":str(108 * 1024**3),"MemorySwapMax":"0"}
    for property_name,expected in expected_limits.items():
        value = service_value("user-1000.slice",property_name)
        observations[f"user-1000.slice.{property_name}"] = value
        if value != expected:
            failures.append(f"user-1000.slice.{property_name}={value}")
    key_id = " ".join(public_key.split()[:2])
    for path in (Path("/root/.ssh/authorized_keys"),Path("/etc/dropbear/initramfs/authorized_keys")):
        key_text = read_optional(path)
        present = any(" ".join(line.split()[:2]) == key_id for line in key_text.splitlines())
        observations[f"key:{path}"] = present
        if not present:
            failures.append(f"missing-key:{path}")
        if key_text != canonical_authorized_keys(key_text,public_key):
            failures.append(f"noncanonical-key:{path}")
    initrd = command(["readlink","-f","/boot/initrd.img"],check=False)
    initrd_files = command(["lsinitramfs",initrd],timeout=180,check=False) if initrd else ""
    observations["initramfs_recovery_key"] = any(line.endswith("/.ssh/authorized_keys") for line in initrd_files.splitlines())
    if not observations["initramfs_recovery_key"]:
        failures.append("missing-key:initramfs")
    grub = read_optional(Path("/etc/default/grub"))
    cmdline = shell_assignment(grub,"GRUB_CMDLINE_LINUX_DEFAULT")
    observations["grub_default"] = shell_assignment(grub,"GRUB_DEFAULT")
    observations["grub_cmdline"] = cmdline
    if shell_assignment(grub,"GRUB_DEFAULT") != "0":
        failures.append("grub-default")
    for token in ("fsck.mode=skip","fsck.repair=no"):
        if token not in shlex.split(cmdline):
            failures.append(f"grub-missing:{token}")
    expected_recovery_ip = recovery_ip_token(payload["recovery_network"])
    if expected_recovery_ip not in shlex.split(cmdline):
        failures.append("grub-recovery-ip")
    if not any(token.startswith("console=ttyS0") for token in shlex.split(cmdline)):
        failures.append("grub-missing:serial")
    if Path("/etc/grub.d/40_ds4_fastboot").exists():
        failures.append("legacy-fastboot-entry")
    efi_boot = command(["efibootmgr","-v"],check=False)
    observations["efi_boot"] = efi_boot
    try:
        efi_order,efi_entries = parse_efi_boot_entries(efi_boot)
        desired_efi = desired_efi_boot_order(efi_boot)
        if efi_order != desired_efi:
            failures.append("efi-pxe-not-first")
    except BrickproofError as error:
        failures.append(f"efi-boot:{error}")
    listeners = command(["ss","-ltn"],check=False)
    if not re.search(r"(?:\*|0\.0\.0\.0):2222\b",listeners):
        failures.append("emergency-port-not-listening")
    root_options = command(["findmnt","-n","-o","OPTIONS","/"],check=False)
    observations["root_options"] = root_options
    if "rw" not in root_options.split(","):
        failures.append("root-read-only")
    observations["hostname"] = command(["hostname","-s"],check=False)
    if observations["hostname"] != payload["recovery_network"]["node_id"]:
        failures.append(f"hostname:{observations['hostname']}")
    expected_rank = str(spark_rank(str(payload["recovery_network"]["node_id"])))
    observations["node_rank"] = read_optional(NODE_RANK_PATH).strip()
    if observations["node_rank"] != expected_rank:
        failures.append(f"node-rank:{observations['node_rank']}")
    hosts = read_optional(Path("/etc/hosts"))
    if hosts_entries(hosts) != hosts_entries(canonical_hosts(hosts,{node_id})):
        failures.append("hosts-loopback-alias")
    observations["kernel"] = command(["uname","-r"],check=False)
    observations["system_state"] = command(["systemctl","is-system-running"],check=False)
    if observations["system_state"] != "running":
        failures.append(f"system-state:{observations['system_state']}")
    failed_units = command(["systemctl","--failed","--no-legend","--plain"],check=False)
    observations["failed_units"] = failed_units
    for line in failed_units.splitlines():
        fields = line.split()
        if fields:
            failures.append(f"failed-systemd-unit:{fields[0]}")
    logrotate = run(["logrotate","--debug","/etc/logrotate.conf"],timeout=180,check=False)
    observations["logrotate_config_rc"] = logrotate.returncode
    if logrotate.returncode != 0:
        failures.append("logrotate-config")
    observations["default_target"] = command(["systemctl","get-default"],check=False)
    if observations["default_target"] != "multi-user.target":
        failures.append(f"default-target:{observations['default_target']}")
    observations["dma_guard"] = {"active":service_value("spark-dma-guard.service","ActiveState"),"mode":dma_guard_mode()}
    if observations["dma_guard"] != {"active":"active","mode":"armed"}:
        failures.append(f"dma-guard:{observations['dma_guard']['active']}:{observations['dma_guard']['mode'] or 'no-start'}")
    expected_assets = dict(payload["assets"])
    if rank != HUB_GATEWAY_RANK:
        expected_assets.update(payload["linked_rank_assets"])
    drifted = asset_drift(expected_assets)
    observations["asset_drift"] = drifted
    failures.extend(f"asset-drift:{path}" for path in drifted)
    network_manager = command(["NetworkManager","--print-config"],check=False)
    missing_unmanaged = sorted(set(FABRIC_INTERFACES) - unmanaged_interfaces(network_manager))
    observations["networkmanager_managed_fabric"] = missing_unmanaged
    if missing_unmanaged:
        failures.append("fabric-managed:" + ",".join(missing_unmanaged))
    if not uplink_assume_disabled(network_manager):
        failures.append("uplink-assume-enabled")
    active_profiles = command(["nmcli","-t","-f","NAME,DEVICE","connection","show","--active"],check=False).splitlines()
    default_routes = command(["ip","-4","route","show","default"],check=False).splitlines()
    observations["default_routes"] = default_routes
    if rank != HUB_GATEWAY_RANK and f"{UPLINK_PROFILE}:{MANAGEMENT_INTERFACE}" not in active_profiles:
        failures.append("uplink-profile-inactive")
    if rank != HUB_GATEWAY_RANK and not uplink_default_route_present(default_routes):
        failures.append("uplink-default-route")
    stale_snaps = stale_snap_units()
    observations["stale_snap_units"] = stale_snaps
    failures.extend(f"stale-snap-unit:{unit}" for unit in stale_snaps)
    memtotal = int(next((line.split()[1] for line in read_optional(Path("/proc/meminfo")).splitlines() if line.startswith("MemTotal:")),"0"))
    observations["memtotal_kb"] = memtotal
    if memtotal < MIN_MEMTOTAL_KB:
        failures.append(f"firmware-carveout:{memtotal}")
    observations["ntp"] = [command(["timedatectl","show-timesync","-p","ServerName","--value"],check=False),command(["timedatectl","show","-p","NTPSynchronized","--value"],check=False)]
    if observations["ntp"][0] not in NTP_SERVERS or observations["ntp"][1] != "yes":
        failures.append(f"ntp:{observations['ntp'][0] or 'none'}:{observations['ntp'][1]}")
    observations["watchdog"] = [command(["systemctl","show","-p",name,"--value"],check=False) for name in ("RuntimeWatchdogUSec","RebootWatchdogUSec")]
    if observations["watchdog"] != ["10min","5min"]:
        failures.append(f"systemd-watchdog:{observations['watchdog'][0]}:{observations['watchdog'][1]}")
    for key,expected in {**RECOVERY_SETTINGS,**DIRTY_CAP_SETTINGS}.items():
        value = command(["sysctl","-n",key],check=False)
        if value != expected:
            failures.append(f"sysctl:{key}={value}")
    failures.extend(f"missing-package:{name}" for name in FLEET_TOOL_PACKAGES if not package_installed(name))
    if not FIREWALL_RULESET.is_file():
        failures.append(f"firewall-ruleset-missing:{FIREWALL_RULESET}")
    for unit in ("spark-firewall.service","sparkpipe-roce-qos.service"):
        state = (is_enabled(unit),service_value(unit,"ActiveState"))
        if state != ("enabled","active"):
            failures.append(f"{unit}:{state[0]}:{state[1]}")
    observations["linger"] = command(["loginctl","show-user",node_id,"-p","Linger","--value"],check=False)
    if observations["linger"] != "yes":
        failures.append(f"linger:{observations['linger'] or 'none'}")
    ceph_states = ceph_unit_file_states()
    observations["ceph_unit_states"] = ceph_states
    unmasked = sorted(unit for unit in CEPH_MASK_UNITS if ceph_states.get(unit) != "masked")
    if unmasked:
        failures.append("ceph-package-units-unmasked:" + ",".join(unmasked))
    if ceph_states.get("ceph.target") != "enabled":
        failures.append(f"ceph.target:{ceph_states.get('ceph.target','missing')}")
    for fsid in ceph_fsids():
        daemons = ceph_daemons(fsid)
        enabled_instances = ceph_enabled_instances(fsid)
        observations[f"ceph:{fsid}"] = daemons
        if ceph_states.get(f"ceph-{fsid}.target") != "enabled":
            failures.append(f"ceph-{fsid}.target:{ceph_states.get(f'ceph-{fsid}.target','missing')}")
        for daemon in daemons:
            unit = f"ceph-{fsid}@{daemon}.service"
            state = (is_enabled(unit),service_value(unit,"ActiveState"))
            required = daemon.startswith(CEPH_REQUIRED_DAEMON_PREFIXES)
            if (required and state != ("enabled","active")) or (state[0] == "enabled" and state[1] != "active"):
                failures.append(f"ceph-daemon:{daemon}:{state[0]}:{state[1]}")
        failures.extend(f"ceph-stray-unit:{unit}" for unit in ceph_stray_units(fsid,daemons,enabled_instances))
        log_dir = CEPH_LOG_ROOT / fsid
        owner = directory_owner(log_dir)
        if owner != f"{CEPH_UID}:{CEPH_UID}":
            failures.append(f"ceph-log-dir:{owner}")
    for path in (*retired_paths(),*(Path("/etc/systemd/system") / unit for unit in OBSOLETE_CEPH_STARTUP_UNITS),LEGACY_CEPH_WARM_STORAGE_MARKER):
        if path.exists() or path.is_symlink():
            failures.append(f"retired-path-present:{path}")
    hardware_failures,hardware_observations = audit_fleet_hardware(
        str(payload["recovery_network"]["node_id"]))
    failures.extend(hardware_failures)
    observations.update(hardware_observations)
    for root in payload["node_layout_roots"]:
        path = Path("/home") / str(payload["recovery_network"]["node_id"]) / str(root)
        stat_result = run(["timeout","15","stat",str(path)],check=False)
        mount = command(["findmnt","-rn","-T",str(path),"-o","TARGET,SOURCE,FSTYPE,OPTIONS"],check=False)
        observations[f"layout:{root}"] = mount
        if stat_result.returncode != 0 or not mount or "rw" not in mount.split()[-1].split(","):
            failures.append(f"layout-root-not-writable:{root}")
    health_path = Path("/var/lib/sparkpipe/fsck-health/last.json")
    if health_path.exists():
        try:
            observations["fsck_health"] = json.loads(health_path.read_text()).get("classification","unknown")
        except json.JSONDecodeError:
            failures.append("fsck-health-json")
    else:
        observations["fsck_health"] = "pending-next-boot"
    return({"failures":failures,"observations":observations,"source_commit":payload["source_commit"]})


def git_show(repo: Path,ref: str,path: str) -> str:
    result = run(["git","-C",str(repo),"show",f"{ref}:{path}"])
    return(result.stdout.decode("utf-8"))


def asset_texts(repo: Path,ref: str,sources: dict[str,tuple[str,int]]) -> dict[str,dict[str,object]]:
    assets: dict[str,dict[str,object]] = {}
    for destination,(source,mode) in sources.items():
        text = git_show(repo,ref,source)
        assets[destination] = {"mode":mode,"text":text if text.endswith("\n") else text + "\n"}
    return(assets)


def build_payload(repo: Path,ref: str) -> dict[str,object]:
    commit = command(["git","-C",str(repo),"rev-parse","--verify",ref])
    assets = asset_texts(repo,ref,ASSET_SOURCES)
    for unit,source_name in BOOT_TIMEOUT_SOURCES.items():
        destination = f"/etc/systemd/system/{unit}.d/10-boot-timeout.conf"
        source = f"tools/devcycle/boot-unblock/{source_name}"
        text = git_show(repo,ref,source)
        assets[destination] = {"mode":0o644,"text":text if text.endswith("\n") else text + "\n"}
    public_key = git_show(repo,ref,"tools/devcycle/sparkpipe_fleet_root.pub").strip()
    if not public_key.startswith("ssh-ed25519 "):
        raise BrickproofError("fleet recovery key is not an Ed25519 public key")
    remote_script = git_show(repo,ref,SELF_SOURCE)
    if Path(__file__).read_text(encoding="utf-8") != remote_script:
        raise BrickproofError(f"controller does not match {ref}:{SELF_SOURCE}")
    return({"assets":assets,"fleet_public_key":public_key,"linked_rank_assets":asset_texts(repo,ref,LINKED_RANK_ASSET_SOURCES),"node_layout_roots":node_layout_roots(git_show(repo,ref,NODE_LAYOUT_SOURCE)),
            "remote_script":remote_script,"source_commit":commit})


def recovery_network_for_node(node: str) -> dict[str,str]:
    rank = spark_rank(node)
    return({
        "address":f"{MANAGEMENT_PREFIX}.{MANAGEMENT_ADDRESS_BASE + rank}",
        "gateway":MANAGEMENT_GATEWAY,
        "interface":MANAGEMENT_INTERFACE,
        "netmask":MANAGEMENT_NETMASK,
        "node_id":node.casefold(),
    })


def ssh(host: str,*argv: str,input_bytes: bytes | None = None,timeout: int = 120,check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return(run(["ssh","-T","-o","BatchMode=yes","-o","ConnectTimeout=8",host,*argv],input_bytes=input_bytes,timeout=timeout,check=check))


def remote_action(host: str,action: str,payload: dict[str,object]) -> dict[str,object]:
    if action not in ("apply","audit"):
        raise BrickproofError(f"unknown remote action {action}")
    node_payload = dict(payload)
    node_payload["recovery_network"] = recovery_network_for_node(host)
    script = str(node_payload.pop("remote_script"))
    request = json.dumps({"payload":node_payload,"script":script},sort_keys=True).encode("utf-8")
    result = ssh(host,"sudo","-n","python3","-I","-c",shlex.quote(REMOTE_BOOTSTRAP),action,input_bytes=request,timeout=1200)
    document = json.loads(result.stdout.decode("utf-8"))
    document["node"] = host
    return(document)


def emergency_probe(host: str,identity: Path) -> str:
    detail = ""
    for target in (host,f"{host}.local",f"{LAN_PREFIX}.{LAN_ADDRESS_BASE + spark_rank(host)}"):
        result = run(["ssh","-T","-p","2222","-i",str(identity),"-o","BatchMode=yes","-o","IdentitiesOnly=yes","-o",f"HostKeyAlias={host}-emergency","-o","StrictHostKeyChecking=accept-new","-o","ConnectTimeout=8",f"root@{target}","true"],timeout=20,check=False)
        if result.returncode == 0:
            return("ok")
        detail = (result.stderr or result.stdout).decode("utf-8",errors="replace").strip()
    return(detail)


def apply_one(host: str,payload: dict[str,object],identity: Path) -> dict[str,object]:
    applied = remote_action(host,"apply",payload)
    audited = remote_action(host,"audit",payload)
    audited["apply"] = applied
    audited["emergency_login"] = emergency_probe(host,identity)
    if audited["emergency_login"] != "ok":
        audited["failures"].append("controller-emergency-login")
    if audited["failures"]:
        raise BrickproofError(f"{host} failed post-apply audit: {json.dumps(audited,sort_keys=True)}")
    return(audited)


def audit_one(host: str,payload: dict[str,object],identity: Path) -> dict[str,object]:
    try:
        audited = remote_action(host,"audit",payload)
    except (BrickproofError,subprocess.TimeoutExpired,json.JSONDecodeError,OSError) as error:
        return({"node":host,"failures":[f"controller-error:{error}"]})
    audited["emergency_login"] = emergency_probe(host,identity)
    if audited["emergency_login"] != "ok":
        audited["failures"].append("controller-emergency-login")
    return(audited)


def write_receipt(action: str,documents: list[dict[str,object]]) -> Path:
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(tempfile.gettempdir()) / f"ds4_spark_brickproof_{action}_{timestamp}.json"
    path.write_text(json.dumps({"action":action,"nodes":documents},indent=2,sort_keys=True) + "\n")
    return(path)


def parse_nodes(value: str) -> tuple[str,...]:
    nodes = tuple(item.strip() for item in value.split(",") if item.strip())
    if not nodes:
        raise BrickproofError("at least one node is required")
    return(nodes)


def controller_main(args: argparse.Namespace) -> int:
    nodes = parse_nodes(args.nodes)
    payload = build_payload(args.sparkpipe_repo,args.source_ref)
    documents = []
    errors = []
    if not args.recovery_identity.is_file():
        raise BrickproofError(f"missing recovery identity: {args.recovery_identity}")
    try:
        if args.action == "apply":
            canary = args.canary if args.canary in nodes else nodes[0]
            print(f"canary={canary}",flush=True)
            documents.append(apply_one(canary,payload,args.recovery_identity))
            waiting = [node for node in nodes if node != canary]
            jobs = max(1,args.jobs)
            running: dict[concurrent.futures.Future[dict[str,object]],str] = {}
            failed = False
            with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
                while True:
                    while waiting and not failed and len(running) < jobs:
                        node = waiting.pop(0)
                        running[executor.submit(apply_one,node,payload,args.recovery_identity)] = node
                    if not running:
                        break
                    done,_ = concurrent.futures.wait(running,return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        node = running.pop(future)
                        try:
                            documents.append(future.result())
                            print(f"{node}: PASS",flush=True)
                        except Exception as error:
                            errors.append({"node":node,"failures":[f"controller-error:{error}"]})
                            failed = True
            errors.extend({"node":node,"failures":["skipped: rollout stopped after a failure"]} for node in waiting)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,min(args.jobs,len(nodes)))) as executor:
                futures = {executor.submit(audit_one,node,payload,args.recovery_identity):node for node in nodes}
                for future in concurrent.futures.as_completed(futures):
                    document = future.result()
                    documents.append(document)
                    print(f"{document['node']}: {'PASS' if not document['failures'] else 'FAIL'}",flush=True)
    except Exception as error:
        errors.append({"node":"controller","failures":[f"controller-error:{error}"]})
    finally:
        documents.sort(key=lambda item:str(item["node"]))
        receipt = write_receipt(args.action,documents + errors)
        print(f"receipt={receipt}")
    for item in errors:
        print(f"{item['node']}: FAIL {item['failures'][0][:300]}",flush=True)
    return(1 if errors or any(document.get("failures") for document in documents) else 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",nargs="?",choices=("audit","apply"),default="audit")
    parser.add_argument("--nodes",default=",".join(DEFAULT_NODES))
    parser.add_argument("--canary",default="spark3")
    parser.add_argument("--jobs",type=int,default=3)
    parser.add_argument("--sparkpipe-repo",type=Path,default=DEFAULT_SPARKPIPE_REPO)
    parser.add_argument("--source-ref",default="origin/main")
    parser.add_argument("--recovery-identity",type=Path,default=DEFAULT_RECOVERY_IDENTITY)
    return(parser.parse_args())


def main() -> int:
    return(controller_main(parse_args()))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BrickproofError,subprocess.TimeoutExpired,json.JSONDecodeError) as error:
        print(f"ds4_spark_brickproof: {error}",file=sys.stderr)
        raise SystemExit(1)
