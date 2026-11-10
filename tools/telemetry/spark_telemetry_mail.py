#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import smtplib
import ssl
import sys
from email.message import EmailMessage
from email.utils import formatdate, make_msgid


SENDER = "notices@sparkpipe.ai"
SMTP_HOST = "sparkpipe.ai"
SMTP_PORT = 587
SMTP_TIMEOUT_S = 15.0
SECRETS_DIR = "/srv/workspace/sparkpipe/secrets"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject",required=True)
    p.add_argument("--password-file",default=os.path.join(SECRETS_DIR,"smtp-notices.pass"))
    p.add_argument("--to-file",default=os.path.join(SECRETS_DIR,"telemetry-alert-to"))
    return(p.parse_args())


def read_value(path: str) -> str:
    with open(path,"r",encoding="utf-8") as fp:
        value = fp.read().strip()
    if value == "":
        raise ValueError("%s is empty" % path)
    return(value)


def build_message(subject: str, body: str, recipient: str) -> EmailMessage:
    mail = EmailMessage()
    mail["From"] = "SparkPipe Telemetry <%s>" % SENDER
    mail["To"] = recipient
    mail["Subject"] = subject
    mail["Date"] = formatdate(usegmt=True)
    mail["Message-ID"] = make_msgid(domain=SENDER.split("@")[1])
    mail.set_content(body)
    return(mail)


def send(mail: EmailMessage, password: str) -> None:
    with smtplib.SMTP(SMTP_HOST,SMTP_PORT,timeout=SMTP_TIMEOUT_S) as smtp:
        smtp.starttls(context=ssl.create_default_context())
        smtp.login(SENDER,password)
        smtp.send_message(mail)


def main() -> int:
    args = parse_args()
    send(build_message(args.subject,sys.stdin.read(),read_value(args.to_file)),read_value(args.password_file))
    return(0)


if __name__ == "__main__":
    raise SystemExit(main())
