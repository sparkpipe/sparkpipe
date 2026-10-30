#!/bin/sh
set -eu

systemctl reload postfix
systemctl reload dovecot
