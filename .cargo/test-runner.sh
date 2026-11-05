#!/bin/sh
# Cargo runs every Linux binary through this script (see .cargo/config.toml).
# It points test binaries' temporary directories at tmpfs, then runs the binary.
#
# Crash and durability tests spend nearly all their time in fsync, and on a real
# disk that makes the suite take many times longer. The tests that crash in-process
# keep the page cache, so those fsyncs check nothing there.
#
# Tests stay on the normal temporary directory when:
#   - TMPDIR is already set, so `TMPDIR=/var/tmp cargo test` is a real-disk run;
#   - /dev/shm is missing, not writable, or smaller than 1 GiB (the suite peaks
#     near 0.75 GiB; Docker's default is 64 MiB).
#
# Only test binaries change. `cargo run` binaries live outside `deps/`, and
# `cargo bench` passes `--bench`: a benchmark on tmpfs would measure no fsync cost.

min_shm_kib=1048576

use_tmpfs() {
    [ -z "${TMPDIR:-}" ] || return 1
    case "$1" in
        */deps/*) ;;
        *) return 1 ;;
    esac
    for arg in "$@"; do
        [ "$arg" = "--bench" ] && return 1
    done
    [ -d /dev/shm ] && [ -w /dev/shm ] || return 1
    size_kib=$(df -Pk /dev/shm 2>/dev/null | awk 'NR == 2 { print $2 }')
    [ "${size_kib:-0}" -ge "$min_shm_kib" ]
}

if use_tmpfs "$@"; then
    TMPDIR=/dev/shm
    export TMPDIR
fi

exec "$@"
