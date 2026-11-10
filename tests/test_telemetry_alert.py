import io
import json
import os
import plistlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","tools","telemetry"))

import spark_telemetry_alert as alert
import spark_telemetry_collect as collect
import spark_telemetry_mail as mail

ROOT = Path(__file__).resolve().parents[1]
HEALTHY = {"sample_count":720,"error":"","fetch_error":"","last_sample_age_s":2.0}
FAILED = {"sample_count":0,"error":"ssh timed out"}
HOST = "hub"
SCRIPT = "/hub/mail.py"


class AlertTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.state = os.path.join(self.dir.name,"state.json")
        self.sent = []
        self.send_error = ""

    def fake_send(self,host,script,subject,body,ssh_config=""):
        self.sent.append((host,script,subject,body,ssh_config))
        return(self.send_error)

    def step(self,now,**rows):
        with mock.patch.object(alert.time,"time",return_value=now), mock.patch.object(alert,"send_mail",side_effect=self.fake_send), mock.patch("sys.stderr",io.StringIO()):
            return(alert.notify_dead_nodes({"nodes":rows},self.state,HOST,SCRIPT))

    def test_working_node_mails_once_after_the_dead_time(self):
        self.step(1000.0,spark1=HEALTHY)
        self.step(1005.0,spark1=FAILED)
        self.assertEqual(self.step(1005.0 + alert.DEAD_AFTER_S - 1.0,spark1=FAILED),[])
        self.assertEqual(self.sent,[])
        self.assertEqual(self.step(1005.0 + alert.DEAD_AFTER_S,spark1=FAILED),["spark1"])
        self.step(1005.0 + alert.DEAD_AFTER_S + 5.0,spark1=FAILED)
        self.step(9999.0,spark1=FAILED)
        self.assertEqual(len(self.sent),1)
        self.assertEqual(self.sent[0][2],"SparkPipe: spark1 dead")
        self.assertIn("ssh timed out",self.sent[0][3])
        self.assertIn("failing since 1970-01-01T00:16:45Z",self.sent[0][3])

    def test_a_failure_shorter_than_the_dead_time_never_mails(self):
        self.step(1000.0,spark1=HEALTHY)
        self.step(1005.0,spark1=FAILED)
        self.step(1005.0 + alert.DEAD_AFTER_S - 1.0,spark1=FAILED)
        self.step(1005.0 + alert.DEAD_AFTER_S,spark1=HEALTHY)
        self.step(1010.0 + alert.DEAD_AFTER_S,spark1=FAILED)
        self.step(1011.0 + alert.DEAD_AFTER_S,spark1=FAILED)
        self.assertEqual(self.sent,[])

    def test_recovery_is_silent_and_rearms_the_next_death(self):
        self.step(1000.0,spark1=HEALTHY)
        self.step(1001.0,spark1=FAILED)
        self.step(1001.0 + alert.DEAD_AFTER_S,spark1=FAILED)
        self.assertEqual(len(self.sent),1)
        self.step(2000.0,spark1=HEALTHY)
        self.assertEqual(len(self.sent),1)
        self.step(2001.0,spark1=FAILED)
        self.step(2001.0 + alert.DEAD_AFTER_S,spark1=FAILED)
        self.assertEqual(len(self.sent),2)

    def test_a_node_already_failing_at_first_sight_never_mails(self):
        self.step(1000.0,spark1=FAILED)
        self.step(1000.0 + 10.0 * alert.DEAD_AFTER_S,spark1=FAILED)
        self.assertEqual(self.sent,[])
        self.step(9000.0,spark1=HEALTHY)
        self.step(9001.0,spark1=FAILED)
        self.step(9001.0 + alert.DEAD_AFTER_S,spark1=FAILED)
        self.assertEqual(len(self.sent),1)

    def test_a_failed_send_retries_after_the_hold_until_it_lands(self):
        self.send_error = "hub unreachable"
        self.step(1000.0,spark1=HEALTHY)
        self.step(1001.0,spark1=FAILED)
        due = 1001.0 + alert.DEAD_AFTER_S
        self.assertEqual(self.step(due,spark1=FAILED),[])
        self.assertEqual(self.step(due + alert.RETRY_AFTER_S - 1.0,spark1=FAILED),[])
        self.assertEqual(len(self.sent),1)
        self.send_error = ""
        self.assertEqual(self.step(due + alert.RETRY_AFTER_S,spark1=FAILED),["spark1"])
        self.step(due + alert.RETRY_AFTER_S + 5.0,spark1=FAILED)
        self.assertEqual(len(self.sent),2)

    def test_nodes_dying_together_share_one_mail(self):
        self.step(1000.0,spark1=HEALTHY,spark2=HEALTHY,spark3=HEALTHY)
        self.step(1001.0,spark1=FAILED,spark2=FAILED,spark3=HEALTHY)
        self.step(1001.0 + alert.DEAD_AFTER_S,spark1=FAILED,spark2=FAILED,spark3=HEALTHY)
        self.assertEqual(len(self.sent),1)
        self.assertEqual(self.sent[0][2],"SparkPipe: spark1, spark2 dead")

    def test_a_stopped_monitor_counts_as_failing_while_ssh_works(self):
        stopped = dict(HEALTHY,last_sample_age_s=alert.telemetry.SAMPLE_STALE_AFTER_S + 1.0)
        self.step(1000.0,lenovo9=HEALTHY)
        self.step(1001.0,lenovo9=stopped)
        self.step(1001.0 + alert.DEAD_AFTER_S,lenovo9=stopped)
        self.assertEqual(self.sent[0][2],"SparkPipe: lenovo9 dead")
        self.assertIn("no new sample for 31 s",self.sent[0][3])

    def test_a_fetch_error_on_cached_samples_counts_as_failing(self):
        cached = dict(HEALTHY,fetch_error="ssh refused",stale_data=1)
        self.assertEqual(alert.failure_reason(cached),"ssh refused")

    def test_the_reason_keeps_the_end_of_a_long_ssh_error(self):
        row = {"sample_count":0,"error":"x" * 500 + " timed out after 8.0 seconds"}
        reason = alert.failure_reason(row)
        self.assertEqual(len(reason),alert.REASON_CHARS)
        self.assertTrue(reason.endswith("timed out after 8.0 seconds"))

    def test_state_survives_a_restart_and_keeps_the_failure_clock(self):
        self.step(1000.0,spark1=HEALTHY)
        self.step(1001.0,spark1=FAILED)
        state = json.loads(Path(self.state).read_text())
        self.assertEqual(state["nodes"]["spark1"],{"failing_since":1001.0,"state":"working"})
        self.step(1001.0 + alert.DEAD_AFTER_S,spark1=FAILED)
        self.assertEqual(len(self.sent),1)

    def test_a_corrupt_state_file_rebaselines_without_mail(self):
        Path(self.state).write_text("{not json")
        self.assertEqual(self.step(1000.0,spark1=FAILED),[])
        self.assertEqual(json.loads(Path(self.state).read_text())["nodes"]["spark1"]["state"],"dead")

    def test_send_mail_quotes_the_remote_command_and_pipes_the_body(self):
        done = subprocess.CompletedProcess([],0,"","")
        with mock.patch.object(alert.subprocess,"run",return_value=done) as run:
            self.assertEqual(alert.send_mail("hub","/a b/mail.py","SparkPipe: x, y dead","body","/cfg"),"")
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[:3],["ssh","-F","/cfg"])
        self.assertIn("BatchMode=yes",cmd)
        self.assertEqual(cmd[-2:],["hub","python3 '/a b/mail.py' --subject 'SparkPipe: x, y dead'"])
        self.assertEqual(run.call_args.kwargs["input"],"body")
        self.assertEqual(run.call_args.kwargs["timeout"],alert.SEND_TIMEOUT_S)

    def test_send_mail_reports_failures_instead_of_raising(self):
        failed = subprocess.CompletedProcess([],255,"","no route")
        with mock.patch.object(alert.subprocess,"run",return_value=failed):
            self.assertEqual(alert.send_mail("hub","s","t","b"),"no route")
        timeout = subprocess.TimeoutExpired("ssh",20)
        with mock.patch.object(alert.subprocess,"run",side_effect=timeout):
            self.assertIn("timed out",alert.send_mail("hub","s","t","b"))


class CollectWiringTest(unittest.TestCase):
    def args(self,*extra):
        out = tempfile.TemporaryDirectory()
        self.addCleanup(out.cleanup)
        with mock.patch("sys.argv",["spark_telemetry_collect.py","--nodes","spark1","--out-dir",out.name,*extra]):
            return(collect.parse_args())

    def run_pass(self,*extra):
        with mock.patch.object(collect,"fetch_node",return_value=("spark1","","ssh timed out")), \
                mock.patch.object(collect.telemetry,"read_ds4_api_queue",return_value={}), \
                mock.patch.object(collect.alert,"notify_dead_nodes") as notify:
            summary = collect.collect_once(self.args(*extra))
        return(summary,notify)

    def test_a_pass_alerts_on_the_summary_it_wrote(self):
        summary,notify = self.run_pass("--alert-state","/s.json","--alert-host","hub","--alert-mail-script","/m.py","--ssh-config","/cfg")
        notify.assert_called_once_with(summary,"/s.json","hub","/m.py","/cfg")

    def test_a_pass_without_alert_state_sends_nothing(self):
        _,notify = self.run_pass()
        notify.assert_not_called()


class DeploymentTest(unittest.TestCase):
    def test_launchd_keeps_alert_state_out_of_the_reboot_wiped_tmp(self):
        data = (ROOT / "deployment/telemetry/com.ds4.spark-telemetry-collector.plist").read_text().replace("@DEST@","/dest").replace("@HOME@","/home")
        argv = plistlib.loads(data.encode())["ProgramArguments"]
        self.assertEqual(argv[argv.index("--alert-state") + 1],"/dest/alert_state.json")

    def test_install_ships_every_module_the_collector_imports(self):
        text = (ROOT / "tools/telemetry/install.sh").read_text()
        line = next(item for item in text.splitlines() if "spark_telemetry_collect.py" in item and "for file in" in item)
        self.assertIn("spark_telemetry_alert.py",line)
        self.assertIn("spark_telemetry_common.py",line)
        self.assertIn('scp -q "$SRC/spark_telemetry_mail.py"',text)

    def test_the_default_script_path_matches_the_hub_install_dir(self):
        text = (ROOT / "tools/telemetry/install.sh").read_text()
        with mock.patch("sys.argv",["spark_telemetry_collect.py"]):
            args = collect.parse_args()
        self.assertIn(os.path.dirname(args.alert_mail_script),text)


class InstallBootstrapTest(unittest.TestCase):
    def run_install(self,failures):
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        home = Path(root.name)
        stubs = home / "stubs"
        stubs.mkdir()
        (home / "Library/LaunchAgents").mkdir(parents=True)
        calls = home / "calls"
        script = "#!/bin/sh\necho \"$1\" >> %s\n[ \"$1\" = bootstrap ] || exit 0\n[ \"$(grep -c bootstrap %s)\" -gt %d ]\n" % (calls,calls,failures)
        for name,text in (("launchctl",script),("plutil","#!/bin/sh\n"),("sleep","#!/bin/sh\n")):
            path = stubs / name
            path.write_text(text)
            path.chmod(0o755)
        env = dict(os.environ,HOME=str(home),PATH="%s:%s" % (stubs,os.environ["PATH"]),SPARKPIPE_TELEMETRY_DIR=str(home / "dest"))
        done = subprocess.run(["sh",str(ROOT / "tools/telemetry/install.sh"),"mac"],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        return(done,calls.read_text().split())

    def test_bootstrap_retries_while_launchd_finishes_the_bootout(self):
        done,calls = self.run_install(2)
        self.assertEqual(done.returncode,0,done.stderr)
        self.assertEqual(calls.count("bootout"),2)
        self.assertEqual(calls.count("bootstrap"),4)

    def test_a_bootstrap_that_never_lands_fails_loudly(self):
        done,calls = self.run_install(100)
        self.assertNotEqual(done.returncode,0)
        self.assertIn("launchctl bootstrap failed",done.stderr)
        self.assertEqual(calls.count("bootstrap"),10)


class MailTest(unittest.TestCase):
    def test_message_goes_from_notices_to_the_recipient(self):
        message = mail.build_message("SparkPipe: spark1 dead","body\n","owner@example.com")
        self.assertIn(mail.SENDER,message["From"])
        self.assertEqual(message["To"],"owner@example.com")
        self.assertEqual(message["Subject"],"SparkPipe: spark1 dead")
        self.assertEqual(message.get_content(),"body\n")
        self.assertIn("@sparkpipe.ai>",message["Message-ID"])

    def test_send_uses_starttls_then_login_before_sending(self):
        message = mail.build_message("s","b\n","owner@example.com")
        with mock.patch.object(mail.smtplib,"SMTP") as smtp:
            mail.send(message,"secret")
        smtp.assert_called_once_with(mail.SMTP_HOST,587,timeout=mail.SMTP_TIMEOUT_S)
        names = [call[0] for call in smtp.return_value.__enter__.return_value.method_calls]
        self.assertEqual(names,["starttls","login","send_message"])
        session = smtp.return_value.__enter__.return_value
        session.login.assert_called_once_with(mail.SENDER,"secret")

    def test_main_reads_the_secrets_and_stdin(self):
        with tempfile.TemporaryDirectory() as root:
            password = Path(root,"pass")
            password.write_text("secret\n")
            recipient = Path(root,"to")
            recipient.write_text("owner@example.com\n")
            argv = ["spark_telemetry_mail.py","--subject","s","--password-file",str(password),"--to-file",str(recipient)]
            with mock.patch("sys.argv",argv), mock.patch("sys.stdin",io.StringIO("body\n")), mock.patch.object(mail,"send") as send:
                self.assertEqual(mail.main(),0)
        message,used = send.call_args.args
        self.assertEqual(used,"secret")
        self.assertEqual(message["To"],"owner@example.com")

    def test_an_empty_recipient_file_fails_loudly(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root,"to")
            path.write_text("\n")
            with self.assertRaises(ValueError):
                mail.read_value(str(path))


if __name__ == "__main__":
    unittest.main()
