import hashlib
import os
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","tools"))

import sparkcp

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "sparkcp.py"
FLEET = ROOT / "tools" / "devcycle" / "fleet"
TOKEN = "0123456789abcdef" * 4
MTIME_NS = 1700000000123456789


def pattern(size):
    block = hashlib.sha256(str(size).encode()).digest() * 1024
    return((block * (size // len(block) + 1))[:size])


def entry(rel,size=0,kind="f"):
    return({"rel":rel,"kind":kind,"size":size,"mode":0o640,"mtime_ns":MTIME_NS})


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def run_tool(self,*args):
        return(subprocess.run([sys.executable,str(TOOL),*[str(arg) for arg in args]],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=120))

    def copy(self,*args):
        done = self.run_tool(*args)
        self.assertEqual(done.returncode,0,done.stderr)
        return(done.stdout)

    def leftovers(self):
        return([str(path) for path in self.dir.rglob(".sparkcp-*")])


class CopyTest(Base):
    def test_tree_keeps_content_modes_times_and_links(self):
        src = self.dir / "src"
        (src / "sub" / "empty").mkdir(parents=True)
        (src / "sub" / "data.bin").write_bytes(pattern(100000))
        (src / "run.sh").write_text("#!/bin/sh\n")
        (src / "zero").write_bytes(b"")
        os.chmod(src / "run.sh",0o750)
        os.chmod(src / "sub",0o700)
        os.utime(src / "run.sh",ns=(MTIME_NS,MTIME_NS))
        os.utime(src / "sub",ns=(MTIME_NS,MTIME_NS))
        dst = self.dir / "dst"
        self.copy(src,dst)
        self.assertEqual((dst / "sub" / "data.bin").read_bytes(),pattern(100000))
        self.assertEqual((dst / "zero").read_bytes(),b"")
        self.assertTrue((dst / "sub" / "empty").is_dir())
        self.assertEqual(stat.S_IMODE((dst / "run.sh").stat().st_mode),0o750)
        self.assertEqual(stat.S_IMODE((dst / "sub").stat().st_mode),0o700)
        self.assertEqual((dst / "run.sh").stat().st_mtime_ns,MTIME_NS)
        self.assertEqual((dst / "sub").stat().st_mtime_ns,MTIME_NS)
        self.assertEqual(self.leftovers(),[])

    def test_links_arrive_as_the_real_files_they_point_to(self):
        src = self.dir / "src"
        (src / "real").mkdir(parents=True)
        (src / "real" / "pack").write_bytes(pattern(5000))
        (src / "file_link").symlink_to("real/pack")
        (src / "dir_link").symlink_to("real")
        outside = self.dir / "outside"
        outside.write_text("elsewhere")
        (src / "far_link").symlink_to(outside)
        dst = self.dir / "dst"
        self.copy(src,dst)
        self.assertEqual((dst / "file_link").read_bytes(),pattern(5000))
        self.assertEqual((dst / "dir_link" / "pack").read_bytes(),pattern(5000))
        self.assertEqual((dst / "far_link").read_text(),"elsewhere")
        self.assertEqual([path for path in dst.rglob("*") if path.is_symlink()],[])

    def test_a_link_named_on_the_command_line_is_followed(self):
        real = self.dir / "real"
        real.write_text("x")
        (self.dir / "current").symlink_to("real")
        self.copy(self.dir / "current",self.dir / "dst")
        self.assertEqual((self.dir / "dst").read_text(),"x")
        self.assertFalse((self.dir / "dst").is_symlink())

    def test_loops_broken_links_and_special_files_are_skipped_and_named(self):
        src = self.dir / "src"
        (src / "sub").mkdir(parents=True)
        (src / "sub" / "keep").write_text("k")
        (src / "sub" / "back").symlink_to("..")
        (src / "broken").symlink_to("gone")
        os.mkfifo(src / "pipe")
        dst = self.dir / "dst"
        output = self.copy(src,dst)
        self.assertEqual((dst / "sub" / "keep").read_text(),"k")
        self.assertFalse((dst / "sub" / "back").exists())
        self.assertFalse((dst / "broken").exists())
        self.assertFalse((dst / "pipe").exists())
        self.assertIn("skipped 3",output)
        for name in ("back","broken","pipe"):
            self.assertIn(name,output)

    def test_a_link_at_the_destination_is_replaced_by_the_file(self):
        src = self.dir / "src"
        src.write_text("new")
        target = self.dir / "target"
        target.write_text("old")
        dst = self.dir / "dst"
        dst.symlink_to("target")
        self.copy(src,self.dir / "dst")
        self.assertFalse(dst.is_symlink())
        self.assertEqual(dst.read_text(),"new")
        self.assertEqual(target.read_text(),"old")

    def test_sizes_around_every_boundary(self):
        sizes = [0,1,sparkcp.ALIGN_BYTES - 1,sparkcp.ALIGN_BYTES,sparkcp.ALIGN_BYTES + 1,sparkcp.CHUNK_BYTES - 1,sparkcp.CHUNK_BYTES,sparkcp.CHUNK_BYTES + 1,sparkcp.DIRECT_MIN_BYTES,sparkcp.DIRECT_MIN_BYTES + sparkcp.ALIGN_BYTES + 5]
        src = self.dir / "src"
        src.mkdir()
        for size in sizes:
            (src / str(size)).write_bytes(pattern(size))
        dst = self.dir / "dst"
        self.copy(src,dst)
        for size in sizes:
            self.assertEqual((dst / str(size)).stat().st_size,size)
            self.assertEqual((dst / str(size)).read_bytes(),pattern(size))
        self.assertEqual(self.leftovers(),[])

    def test_destinations_follow_cp(self):
        data = self.dir / "file"
        data.write_text("x")
        tree = self.dir / "tree"
        tree.mkdir()
        (tree / "inner").write_text("y")
        existing = self.dir / "existing"
        existing.mkdir()
        self.copy(data,self.dir / "renamed")
        self.assertEqual((self.dir / "renamed").read_text(),"x")
        self.copy(data,existing)
        self.assertEqual((existing / "file").read_text(),"x")
        self.copy(data,str(self.dir / "new" / "deep") + "/")
        self.assertEqual((self.dir / "new" / "deep" / "file").read_text(),"x")
        self.copy(tree,self.dir / "tree2")
        self.assertEqual((self.dir / "tree2" / "inner").read_text(),"y")
        self.copy(tree,existing)
        self.assertEqual((existing / "tree" / "inner").read_text(),"y")
        self.copy(data,tree,self.dir / "both")
        self.assertEqual((self.dir / "both" / "file").read_text(),"x")
        self.assertEqual((self.dir / "both" / "tree" / "inner").read_text(),"y")

    def test_rerun_copies_only_what_changed(self):
        src = self.dir / "src"
        src.mkdir()
        (src / "same").write_text("a")
        (src / "edited").write_text("b")
        dst = self.dir / "dst"
        self.copy(src,dst)
        before = (dst / "same").stat().st_ino
        (src / "edited").write_text("bigger")
        output = self.copy(src / "same",dst / "same")
        self.assertIn("0 files",output)
        self.assertIn("1 already up to date",output)
        self.assertEqual((dst / "same").stat().st_ino,before)
        output = self.copy(src / "edited",dst / "edited")
        self.assertIn("1 file,",output)
        self.assertEqual((dst / "edited").read_text(),"bigger")

    def test_same_size_with_a_new_mtime_is_copied_again(self):
        src = self.dir / "src"
        src.write_text("old")
        dst = self.dir / "dst"
        self.copy(src,dst)
        src.write_text("new")
        os.utime(src,ns=(MTIME_NS,MTIME_NS))
        output = self.copy(src,dst)
        self.assertIn("1 file,",output)
        self.assertEqual(dst.read_text(),"new")

    def test_missing_source_fails_and_touches_nothing(self):
        dst = self.dir / "dst"
        done = self.run_tool(self.dir / "absent",dst)
        self.assertEqual(done.returncode,1)
        self.assertIn("absent",done.stderr)
        self.assertFalse(dst.exists())

    def test_failed_replace_reports_the_receiver_error(self):
        src = self.dir / "src"
        src.write_bytes(pattern(1000))
        dst = self.dir / "dst"
        (dst / "src").mkdir(parents=True)
        (dst / "src" / "blocker").write_text("keep")
        done = self.run_tool(src,dst)
        self.assertEqual(done.returncode,1)
        self.assertIn("src",done.stderr)
        self.assertEqual((dst / "src" / "blocker").read_text(),"keep")
        self.assertEqual(self.leftovers(),[])

    def test_sources_on_two_nodes_are_refused(self):
        done = self.run_tool("spark3:a","spark4:b",self.dir / "dst")
        self.assertEqual(done.returncode,1)
        self.assertIn("one node",done.stderr)

    def test_unknown_node_is_named(self):
        done = self.run_tool("sparkz:a",self.dir / "dst")
        self.assertEqual(done.returncode,1)
        self.assertIn("sparkz",done.stderr)
        self.assertIn("lenovo9",done.stderr)


class ReceiverTest(Base):
    def receiver(self,entries):
        return(sparkcp.Receiver(str(self.dir / "dst"),entries,TOKEN))

    def frame(self,index,offset,data,digest=None):
        return(sparkcp.FRAME.pack(index,offset,len(data),digest or hashlib.sha256(data).digest()) + data)

    def feed(self,receiver,payload):
        ours,theirs = socket.socketpair()
        self.addCleanup(ours.close)
        self.addCleanup(theirs.close)
        writer = threading.Thread(target=lambda: (ours.sendall(payload),ours.shutdown(socket.SHUT_WR)),daemon=True)
        writer.start()
        sparkcp.receive_stream(theirs,receiver)
        writer.join()

    def test_good_frames_complete_the_file(self):
        data = pattern(5000)
        receiver = self.receiver([entry("f",5000)])
        receiver.plan()
        self.feed(receiver,self.frame(0,0,data))
        receiver.finish()
        self.assertEqual((self.dir / "dst").read_bytes(),data)

    def test_corrupt_chunk_never_reaches_the_destination(self):
        data = pattern(5000)
        receiver = self.receiver([entry("f",5000)])
        receiver.plan()
        with self.assertRaisesRegex(sparkcp.CopyError,"checksum mismatch"):
            self.feed(receiver,self.frame(0,0,data,b"\0" * 32))
        receiver.abort()
        self.assertEqual(list(self.dir.iterdir()),[])

    def test_truncated_stream_is_an_error(self):
        data = pattern(5000)
        receiver = self.receiver([entry("f",5000)])
        receiver.plan()
        with self.assertRaises(sparkcp.CopyError):
            self.feed(receiver,self.frame(0,0,data)[:100])
        receiver.abort()
        self.assertFalse((self.dir / "dst").exists())

    def test_missing_chunk_fails_the_finish(self):
        data = pattern(sparkcp.ALIGN_BYTES)
        receiver = self.receiver([entry("f",2 * sparkcp.ALIGN_BYTES)])
        receiver.plan()
        self.feed(receiver,self.frame(0,0,data))
        with self.assertRaisesRegex(sparkcp.CopyError,"complete"):
            receiver.finish()
        receiver.abort()
        self.assertEqual(self.leftovers(),[])
        self.assertFalse((self.dir / "dst").exists())

    def test_chunk_past_the_end_is_refused(self):
        receiver = self.receiver([entry("f",10)])
        receiver.plan()
        with self.assertRaisesRegex(sparkcp.CopyError,"past the end"):
            self.feed(receiver,self.frame(0,5,pattern(20)))
        receiver.abort()

    def test_unknown_file_index_is_refused(self):
        receiver = self.receiver([entry("f",10)])
        receiver.plan()
        with self.assertRaisesRegex(sparkcp.CopyError,"not pending"):
            self.feed(receiver,self.frame(7,0,pattern(10)))
        receiver.abort()

    def test_manifest_cannot_escape_the_destination(self):
        for rel in ("../evil","/etc/passwd","a/../../evil","a//b","./a"):
            receiver = sparkcp.Receiver(str(self.dir / "dst") + "/",[entry("root",0,"d"),entry(rel)],TOKEN)
            with self.assertRaisesRegex(sparkcp.CopyError,"unsafe"):
                receiver.plan()
        self.assertFalse((self.dir / "evil").exists())

    def test_large_file_is_preallocated_before_data_arrives(self):
        size = sparkcp.DIRECT_MIN_BYTES
        receiver = self.receiver([entry("f",size)])
        receiver.plan()
        with mock.patch.object(sparkcp,"preallocate",wraps=sparkcp.preallocate) as call:
            state = receiver.open_file(0)
        call.assert_called_once_with(int(state["fd"]),size)
        receiver.abort()

    def test_preallocation_reports_a_full_disk(self):
        with tempfile.TemporaryFile() as fp:
            with mock.patch.object(sparkcp.os,"posix_fallocate",create=True,side_effect=OSError(sparkcp.errno.ENOSPC,"full")):
                with self.assertRaises(OSError):
                    sparkcp.preallocate(fp.fileno(),4096)

    def test_preallocation_falls_back_where_it_is_unsupported(self):
        with tempfile.TemporaryFile() as fp:
            with mock.patch.object(sparkcp.os,"posix_fallocate",create=True,side_effect=OSError(sparkcp.errno.EOPNOTSUPP,"no")):
                sparkcp.preallocate(fp.fileno(),4096)
            self.assertEqual(os.fstat(fp.fileno()).st_size,4096)

    def test_disk_too_small_fails_before_any_write(self):
        receiver = self.receiver([entry("f",5 * 10 ** 9)])
        with mock.patch.object(sparkcp.os,"statvfs",return_value=mock.Mock(f_bavail=1024,f_frsize=4096)):
            with self.assertRaisesRegex(sparkcp.CopyError,"free"):
                receiver.plan()
        self.assertEqual(self.leftovers(),[])


class HandshakeTest(unittest.TestCase):
    def test_wrong_token_is_dropped_and_the_right_one_accepted(self):
        with socket.socket() as listener:
            listener.bind((sparkcp.LOOPBACK,0))
            listener.listen(4)
            port = listener.getsockname()[1]
            intruder = socket.create_connection((sparkcp.LOOPBACK,port))
            intruder.sendall(b"x" * len(TOKEN))
            honest = socket.create_connection((sparkcp.LOOPBACK,port))
            honest.sendall(TOKEN.encode())
            accepted = sparkcp.accept_verified(listener,TOKEN)
            accepted.sendall(b"ok")
            self.assertEqual(honest.recv(2),b"ok")
            self.assertEqual(intruder.recv(2),b"")
            for sock in (accepted,honest,intruder):
                sock.close()


class PathTest(unittest.TestCase):
    def test_cisco_is_the_first_choice(self):
        addresses = ["172.17.0.1","100.68.1.1","192.168.50.3","10.20.0.13","10.10.100.13","10.40.0.7","10.10.201.3"]
        self.assertEqual([label for label,_ in sparkcp.ranked_addresses(addresses)],["cisco","crs804","mgmt","lan","tailscale"])

    def test_a_dead_cisco_path_falls_back(self):
        def connect(address,port,token,timeout_s):
            if address.startswith(sparkcp.CISCO_PREFIX):
                raise OSError("no route")
            return("socket")
        with mock.patch.object(sparkcp,"connect_verified",side_effect=connect):
            self.assertEqual(sparkcp.connect_first(["10.10.201.3","10.10.100.13"],1,TOKEN),("socket","crs804","10.10.100.13"))

    def test_no_path_is_an_error(self):
        with mock.patch.object(sparkcp,"connect_verified",side_effect=OSError("no route")):
            with self.assertRaisesRegex(sparkcp.CopyError,"no reachable path"):
                sparkcp.connect_first(["10.10.201.3"],1,TOKEN)

    def test_every_machine_has_a_unique_cisco_address(self):
        nodes = sparkcp.fleet_nodes()
        self.assertEqual(len(nodes),20)
        self.assertEqual(len({node.cisco for node in nodes.values()}),20)
        self.assertEqual(nodes["sparkf"],sparkcp.Node("sparkf","10.10.201.15"))
        self.assertEqual(nodes["lenovo8"],sparkcp.Node("spark8","10.10.201.248"))
        self.assertEqual(nodes["hub"],sparkcp.Node("spec","10.10.201.250"))

    def test_endpoints(self):
        nodes = sparkcp.fleet_nodes()
        parse = sparkcp.parse_endpoint
        self.assertEqual(parse("spark3:/a",nodes,""),sparkcp.Endpoint("spark3","/a"))
        self.assertEqual(parse("rtx5090:/a",nodes,""),sparkcp.Endpoint("hub","/a"))
        self.assertEqual(parse("lenovo7:",nodes,""),sparkcp.Endpoint("lenovo7","."))
        self.assertEqual(parse("spark3:/a",nodes,"spark3"),sparkcp.Endpoint("","/a"))
        self.assertEqual(parse("/x:y/z",nodes,""),sparkcp.Endpoint("","/x:y/z"))
        self.assertEqual(parse("plain",nodes,""),sparkcp.Endpoint("","plain"))

    def test_control_prefers_cisco_only_from_a_cisco_host(self):
        nodes = sparkcp.fleet_nodes()
        self.assertEqual(sparkcp.ssh_targets("lenovo7",nodes,True),["spark7@10.10.201.247","lenovo7"])
        self.assertEqual(sparkcp.ssh_targets("lenovo7",nodes,False),["lenovo7"])
        self.assertEqual(sparkcp.ssh_targets("hub",nodes,False),["rtx5090"])


class AgentLifetimeTest(Base):
    def test_losing_the_controller_removes_the_partial_file(self):
        agent = sparkcp.Agent("",sparkcp.fleet_nodes(),False,TOOL.read_bytes())
        offer = agent.start({"role":"receive","token":TOKEN,"dest":str(self.dir / "dst")})
        control = sparkcp.connect_verified(sparkcp.LOOPBACK,int(offer["port"]),TOKEN,5.0)
        sparkcp.send_json(control,{"entries":[entry("f",1000)]})
        reply = sparkcp.read_json(control)
        self.assertEqual(reply["need"],[0])
        streams = [sparkcp.connect_verified(sparkcp.LOOPBACK,int(offer["port"]),TOKEN,5.0) for _ in range(reply["streams"])]
        payload = pattern(100)
        streams[0].sendall(sparkcp.FRAME.pack(0,0,100,hashlib.sha256(payload).digest()) + payload)
        deadline = time.monotonic() + 20.0
        while self.leftovers() == [] and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(len(self.leftovers()),1)
        agent.stop()
        control.close()
        for sock in streams:
            sock.close()
        self.assertEqual(self.leftovers(),[])
        self.assertFalse((self.dir / "dst").exists())


class CiscoPortTest(Base):
    def stub(self,host):
        bin_dir = self.dir / "bin"
        bin_dir.mkdir(exist_ok=True)
        log = self.dir / "calls"
        for name in ("ip","ethtool","sysctl"):
            path = bin_dir / name
            path.write_text('#!/bin/sh\necho "%s $*" >> "%s"\n' % (name,log))
            path.chmod(0o755)
        path = bin_dir / "hostname"
        path.write_text("#!/bin/sh\necho %s\n" % host)
        path.chmod(0o755)
        sysfs = self.dir / "sysfs"
        for device in ("enp1s0f0np0","enp1s0f1np1"):
            (sysfs / device).mkdir(parents=True,exist_ok=True)
        env = dict(os.environ,PATH="%s:%s" % (bin_dir,os.environ["PATH"]),DS4_CISCO_NO_SUDO="1",DS4_SYSFS_NET=str(sysfs))
        done = subprocess.run([str(FLEET / "ds4_cisco_port_apply.sh")],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env,timeout=30)
        return(done,log.read_text() if log.exists() else "")

    def test_every_lenovo_gets_the_address_sparkcp_expects(self):
        nodes = sparkcp.fleet_nodes()
        for host in ("lenovo7","lenovo8","lenovo9"):
            done,calls = self.stub(host)
            self.assertEqual(done.returncode,0,done.stderr)
            self.assertRegex(calls,r"ip address replace %s/24 dev enp1s0f[01]np[01]\n" % nodes[host].cisco.replace(".",r"\."))
            self.assertIn("mtu 9000",calls)
            (self.dir / "calls").unlink()

    def test_a_host_without_a_known_port_changes_nothing(self):
        done,calls = self.stub("spark3")
        self.assertEqual(done.returncode,2)
        self.assertEqual(calls,"")

    def test_the_unit_stays_out_of_the_boot_path_and_retries_are_bounded(self):
        service = (FLEET / "ds4-cisco-port.service").read_text()
        timer = (FLEET / "ds4-cisco-port.timer").read_text()
        self.assertNotIn("[Install]",service)
        self.assertIn("WantedBy=timers.target",timer)
        self.assertIn("OnBootSec=30",timer)
        self.assertIn("StartLimitBurst=5",service)
        self.assertNotIn("Restart=always",service)
        self.assertNotIn("multi-user.target",service + timer)

    def test_hub_ssh_prefers_cisco_and_keeps_the_known_host_key(self):
        block = subprocess.run([str(FLEET / "install_sparkcp.sh"),"hub-block"],stdout=subprocess.PIPE,text=True,timeout=30).stdout
        nodes = sparkcp.fleet_nodes()
        for name,node in nodes.items():
            if name.startswith("spark"):
                self.assertIn("originalhost %s exec" % name,block)
                self.assertIn("/dev/tcp/%s/22" % node.cisco,block)
                self.assertIn("HostName %s\n" % node.cisco,block)
        self.assertEqual(block.count("HostKeyAlias 10.10.100."),16)
        self.assertTrue(block.rstrip().endswith("Host *\n# END cisco primary path"))


if __name__ == "__main__":
    unittest.main()
