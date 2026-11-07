# VPS mail

The VPS that serves sparkpipe.ai (27.102.120.99) is also the mail server
for sparkpipe.ai and hyor.io. Postfix receives and sends mail, Dovecot
holds the mailboxes and serves IMAP, and rspamd filters incoming mail and
signs outgoing mail with DKIM. Both domains' MX records name
`sparkpipe.ai`, and their SPF records name 27.102.120.99.

A mail client uses `sparkpipe.ai` for both servers, with the full address
as the user name: IMAP on port 993 over TLS, SMTP on port 587 with
STARTTLS. The certificate is sparkpipe.ai's Let's Encrypt certificate.

## Files

| `deployment/vps/` | On the VPS | What it sets |
|---|---|---|
| `postfix-main.cf` | `/etc/postfix/main.cf` | The two mail domains, delivery to Dovecot over LMTP, TLS, the rspamd milter, the sender login check |
| `postfix-master.cf` | `/etc/postfix/master.cf` | Ubuntu's standard services, plus submission on 587: TLS and a login required, a sender only as an address its login owns, 10 connections and 60 messages a minute per client |
| `dovecot-10-auth.conf` | `/etc/dovecot/conf.d/10-auth.conf` | Logins from `/etc/dovecot/users`, passwords hashed with ARGON2ID |
| `dovecot-99-sparkpipe.conf` | `/etc/dovecot/conf.d/99-sparkpipe.conf` | IMAP only over TLS on 993, maildirs under `/var/vmail/<domain>/<user>` owned by `vmail` (uid 5000), a 5 GB quota per mailbox, the auth and LMTP sockets Postfix uses |
| `rspamd-actions.conf` | `/etc/rspamd/local.d/actions.conf` | Spam scores: greylist from 4, a spam header from 6, reject from 15 |
| `rspamd-redis.conf` | `/etc/rspamd/local.d/redis.conf` | rspamd's Redis on 127.0.0.1 |
| `rspamd-dkim_signing.conf` | `/etc/rspamd/local.d/dkim_signing.conf` | DKIM signing for both domains (`docs/PAYMENTS.md`, "Mail") |
| `certbot-reload-mail-services.sh` | `/etc/letsencrypt/renewal-hooks/deploy/reload-mail-services.sh` | Reloads Postfix and Dovecot when the certificate renews |

The other files under `/etc/postfix`, `/etc/dovecot` and `/etc/rspamd`
are the Ubuntu packages' own.

## What stays on the VPS

The repository is public, so the address lists, passwords and keys stay on
the VPS only:

- `/etc/postfix/virtual_mailboxes`: one line per mailbox,
  `address domain/user/`.
- `/etc/postfix/virtual_aliases`: `address target`. `postmaster@`,
  `abuse@` and `dmarc@` of each domain deliver to one of its mailboxes.
- `/etc/postfix/sender_login_maps`: `address login`. A login may send only
  as the addresses listed for it.
- `/etc/dovecot/users` (mode 640, `root:dovecot`): one line per login,
  `address:{ARGON2ID}hash::::::`.
- The DKIM keys under `/var/lib/rspamd/dkim/` and the TLS keys under
  `/etc/letsencrypt/`.

Back each file up before editing it. Postfix reads the maps from the `.db`
files that `postmap` builds, so run `postmap` on a map after editing it.
Neither service needs a reload for these files.

## Adding a mailbox or setting a password

1. Add the address to `virtual_mailboxes` and `sender_login_maps`, and run
   `postmap` on both.
2. The mailbox's owner runs `doveadm pw -s ARGON2ID` on the VPS, types the
   password, and puts `address:` followed by the printed hash and `::::::`
   in `/etc/dovecot/users`, replacing the address's old line if there is
   one. Nobody else types or sees the password.

A new domain also needs `virtual_mailbox_domains` in `postfix-main.cf`, a
DKIM key and selector in `rspamd-dkim_signing.conf`, and MX, SPF, DMARC
and DKIM records at its DNS host. Those go through a PR like any other
change.

## Deploying a change

Only what is merged to `main` goes to the VPS, as in `docs/PAYMENTS.md`.
Back up each file being replaced, then:

```sh
sudo install -m 644 deployment/vps/postfix-main.cf /etc/postfix/main.cf
sudo install -m 644 deployment/vps/postfix-master.cf /etc/postfix/master.cf
sudo install -m 644 deployment/vps/dovecot-10-auth.conf /etc/dovecot/conf.d/10-auth.conf
sudo install -m 644 deployment/vps/dovecot-99-sparkpipe.conf /etc/dovecot/conf.d/99-sparkpipe.conf
sudo install -m 644 deployment/vps/rspamd-actions.conf /etc/rspamd/local.d/actions.conf
sudo install -m 644 deployment/vps/rspamd-redis.conf /etc/rspamd/local.d/redis.conf
sudo install -m 755 deployment/vps/certbot-reload-mail-services.sh /etc/letsencrypt/renewal-hooks/deploy/reload-mail-services.sh
sudo postfix check && sudo systemctl reload postfix
sudo doveconf -n > /dev/null && sudo systemctl reload dovecot
sudo rspamadm configtest && sudo systemctl reload rspamd
```

The live files must then match `origin/main` byte for byte. `postconf -n`
and `doveconf -n` print the settings in force.
