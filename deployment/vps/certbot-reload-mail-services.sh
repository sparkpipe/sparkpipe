#!/bin/sh
set -u

status=0
systemctl reload postfix || status=1
systemctl reload dovecot || status=1
exit "$status"
