#!/bin/sh
# Fetch the diagnostics archive. See docs/archive/DIAGNOSTICS_ARCHIVE.md.
#
# The captures live in a release rather than the tree: they were 2,533 of the
# repository's files and 38 MB, which made every ratio meaningless. They are
# evidence for the measured claims in docs/, so they are archived, hashed and
# fetchable rather than deleted.
set -e
DEST=${1:-diagnostics}
TAG=diagnostics-20260727
ASSET=$TAG.tar.gz
SHA=638c2b8088052b60dfc5756ee92ff741f66f9145d0b46ed97833f152cafb76b1
URL=https://github.com/sparkpipe/sparkpipe/releases/download/$TAG/$ASSET
WORK=$(mktemp -d)
curl -sL "$URL" -o "$WORK/$ASSET"
echo "$SHA  $WORK/$ASSET" | sha256sum -c - || { echo "checksum mismatch; not unpacking" >&2; rm -rf "$WORK"; exit 2; }
mkdir -p "$DEST"
tar xzf "$WORK/$ASSET" -C "$WORK"
cp -r "$WORK/diagnostics/." "$DEST/"
rm -rf "$WORK"
printf "%s runs, %s files -> %s\n" "$(ls "$DEST" | wc -l)" "$(find "$DEST" -type f | wc -l)" "$DEST"
