# Joining the tailnet without a browser login

A node joins the tailnet in one of two ways, and neither opens a login URL.

- **A reimaged node** keeps its identity. The
  [disk install](HOSTED_NODE_IMAGE.md#disk-install) copies `/var/lib/tailscale`
  from the node's previous root into the new one, so `tailscaled` starts at
  boot with the node's own keys and the node rejoins as itself.
- **A new node** has no state to carry. During bring-up,
  [`tools/tailscale_join.sh`](../tools/tailscale_join.sh) logs it in with an
  auth key, over SSH, from an operator's workstation or the hub.

The node's tailnet hostname and tags come from the site's own records, not
from this repository.

## The join tool

```sh
tools/tailscale_join.sh --ssh NODE [--hostname NAME] [--tags tag:NAME[,tag:NAME]] [--operator USER]
```

- The key file is `$TAILSCALE_AUTH_KEY_FILE`, by default
  `~/.config/sparkpipe/tailscale-authkey`.
- The tailnet name is `--hostname`, by default the node's own `hostname`,
  which must be a lower-case DNS label.
- `--tags` passes `--advertise-tags`, and the tool then checks for exactly
  those tags. Without it, the node gets the tags the auth key carries. An
  OAuth client secret always needs `--tags`.

It runs only when someone runs it. It installs nothing on the node, and it
never runs at boot. The SSH user needs passwordless `sudo` on the node.

**Before it connects**, it checks:

- any hostname given is a DNS label, any tags are `tag:` names, and any
  operator is a user name;
- the key file is a regular file, not a link, owned by the user running the
  tool, with no group or other permissions (mode 0600 or 0400);
- the file holds one line: a Tailscale auth key (`tskey-auth-`) or an OAuth
  client secret (`tskey-client-`). A client secret must carry
  `?ephemeral=false&preauthorized=true`: without them `tailscale up` makes an
  ephemeral node, which the tailnet removes when it goes offline, and
  without `preauthorized=true` the node waits for approval.

It never prints the key, puts it on a command line or writes it to a log.

**On the node**, it first reads `tailscale status --json`.

- A node already `Running` and online with this hostname, and these tags if
  `--tags` is given, gets no key. The tool says it has nothing to do and
  exits 0.
- A node in `NeedsLogin`, `NoState` or `Stopped` is joined.
- Any other state is refused without sending the key, for example a node
  logged in under another name or waiting for approval.

**The join itself** is a script run as root on the node. The key travels only
on the SSH connection's standard input.

1. The script checks that the key directory's parent (`/run` by default) is on
   tmpfs and that the directory does not exist.
2. It creates `/run/sparkpipe-tailscale-join` (mode 0700) and writes the key
   to `key` in it (mode 0600, root's).
3. It runs `tailscale up --auth-key=file:/run/sparkpipe-tailscale-join/key
   --hostname=NAME [--advertise-tags=TAGS] [--operator=USER] --timeout=120s`.
   The pinned Tailscale 1.104.7 reads an `--auth-key` value that starts with
   `file:` from that file and trims it. The same flag takes an OAuth client
   secret.
4. The script removes the key and its directory when it exits, also when
   `tailscale up` fails or the connection drops.

The tool then checks that the directory is gone. Within ten seconds,
`tailscale status --json` must show the node `Running` and online with this
hostname, and with exactly the tags given in `--tags`. The tool prints the node's Tailscale
addresses. Anything else fails, and the error names the state it found.

`--key-dir DIR` puts the key elsewhere. The directory's parent must still be
on tmpfs.

## Setting up the tailnet, once

These steps are for the tailnet's owner or an admin.

1. **Tag.** In the tailnet policy file, define the nodes' tag and who may
   apply it, for example `"tagOwners": {"tag:NAME": ["autogroup:admin"]}`.
   A tagged node's key expiry is disabled by default. So a node joined this
   way does not have to log in again, and its state stays valid when the
   disk install carries it.
2. **Credential.** Create one of these:
   - an auth key: **Reusable**, **Pre-approved**, with **Tags** set to the
     nodes' tag, and not **Ephemeral**. An auth key expires after 1 to 90
     days. A node it already joined stays joined. Only new joins need a fresh
     key.
   - an OAuth client with the `auth_keys` scope and the nodes' tag. Its
     secret does not expire. Write it into the file as
     `SECRET?ephemeral=false&preauthorized=true`.
3. **Key file.** Store the key or secret as one line in
   `~/.config/sparkpipe/tailscale-authkey` on the machine that runs the tool,
   owned by the user who runs it, mode 0600. To keep it elsewhere, set
   `TAILSCALE_AUTH_KEY_FILE` to its path. It never goes into a repository.

## Tests

[`tests/test_tailscale_join.py`](../tests/test_tailscale_join.py) runs the tool
against fake `ssh`, `sudo`, `findmnt` and `tailscale` commands on `PATH`. The
fake `ssh` runs the remote command locally with `/bin/sh`, as a node's login
shell would. The fake `tailscale up` records whether the key file held the key,
with its file and directory modes, at the moment it ran.

The cases:

- **A good join:**
  - `tailscale up` gets exactly the flags above, with and without tags and
    an operator, and the key arrives intact at 0600 in a 0700 directory;
  - the default key file and the node's own hostname are used unless
    overridden;
  - the directory is gone afterwards;
  - the key appears in no output, command line or log.
- **Joined or refused without the key:**
  - a node already joined gets no key;
  - a node logged in as another name, or waiting for approval, is refused
    without the key.
- **Refused before connecting:** a key file that is readable by others,
  linked, missing, not one line, not a Tailscale key, or a client secret
  without its attributes or without `--tags`; and bad hostnames, tags,
  operators and key directories.
- **Refused before the key is sent:** a node hostname that is not a
  lower-case DNS label, when no `--hostname` is given.
- **Failures on the node:**
  - a key directory off tmpfs is refused, with nothing written;
  - a leftover key directory is refused;
  - a failed login still removes the key;
  - a join that never reaches `Running` fails and names the state.
