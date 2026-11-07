//! A real process killed mid-write: every acknowledged write survives.

use super::*;

// ── Process-level crash durability ─────────────────────────────────
//
// Everything else here asserts on counters inside one process. The crash pair
// below is the end-to-end guard: a real child process writes, is killed by a
// real `SIGKILL`, and the parent reopens the database and recounts. That is the
// only shape that tests the actual promise — "once `put` returns, the write
// survives" — rather than a proxy for it.

/// Env var carrying the database path from the parent to the crash child.
const CRASH_TEST_DIR_VAR: &str = "MINNAL_CRASH_TEST_DIR";

/// Written to the namespace that never fills its memtable.
const QUIET_KEYS: u32 = 100;

/// Written to the namespace that flushes repeatedly, dragging the watermark.
const BUSY_KEYS: u32 = 1_500;

fn quiet_key(i: u32) -> String {
    format!("quiet-{i:04}")
}

fn busy_key(i: u32) -> String {
    format!("busy-{i:04}")
}

/// The child half of [`acknowledged_writes_survive_a_process_kill`].
///
/// Inert unless the parent sets [`CRASH_TEST_DIR_VAR`], so a bare
/// `cargo test -- --ignored` cannot make it do anything.
///
/// It writes `quiet` first and `busy` second on purpose: `busy`'s flushes
/// then happen while `quiet`'s entries are still only in a memtable, which is
/// exactly the ordering that made one namespace's flush mark another's WAL
/// entries persisted.
#[test]
#[ignore = "spawned as a child process by acknowledged_writes_survive_a_process_kill"]
fn crash_child_writes_two_namespaces_then_dies() {
    let Ok(dir) = std::env::var(CRASH_TEST_DIR_VAR) else {
        return;
    };

    let db = Database::open(std::path::Path::new(&dir), crash_test_config()).unwrap();
    let quiet = db.create_namespace("quiet").unwrap();
    let busy = db.create_namespace("busy").unwrap();

    for i in 0..QUIET_KEYS {
        db.put_ns(quiet, quiet_key(i).as_bytes(), b"quiet-value").unwrap();
    }
    for i in 0..BUSY_KEYS {
        db.put_ns(busy, busy_key(i).as_bytes(), b"busy-value").unwrap();
    }

    // Every `put_ns` above returned `Ok`, which is the durability
    // acknowledgement: the WAL entry is fsynced before it returns. Die
    // without unwinding, running a destructor, or flushing anything —
    // `SIGKILL` cannot be caught, so nothing here gets a chance to tidy up,
    // exactly as in a power loss.
    unsafe { libc::raise(libc::SIGKILL) };
    unreachable!("SIGKILL did not terminate the crash child");
}

/// Every acknowledged write must survive a real `SIGKILL`, across two
/// namespaces with different flush histories.
///
/// This is the guard the suite was missing. Both critical defects found by
/// the stress run — a value-log fsync marking WAL entries persisted, and one
/// namespace's flush marking another's — were live while 1133 tests passed,
/// because nothing killed a process and recounted. Reintroduce either and
/// this fails: `quiet`'s entries are marked persisted while they exist only
/// in a memtable, so recovery skips them and the data is gone.
#[test]
fn acknowledged_writes_survive_a_process_kill() {
    use std::os::unix::process::ExitStatusExt;

    let dir = TempDir::new().unwrap();

    // Re-invoke this same test binary, running only the ignored child.
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        // `--exact` matches the full test path, not the bare function name.
        .args([
            "db::database::tests::crash::crash_child_writes_two_namespaces_then_dies",
            "--exact",
            "--ignored",
            "--nocapture",
        ])
        .env(CRASH_TEST_DIR_VAR, dir.path())
        .status()
        .expect("failed to spawn the crash child");

    assert_eq!(
        status.signal(),
        Some(libc::SIGKILL),
        "child should have been killed by SIGKILL, but exited with {status:?} — \
             it probably failed before reaching the kill, so nothing below is meaningful"
    );

    // Reopen exactly as a restart would: this runs WAL recovery.
    let db = Database::open(dir.path(), crash_test_config()).unwrap();
    let quiet = db.get_namespace_id("quiet").expect("'quiet' should have survived the crash");
    let busy = db.get_namespace_id("busy").expect("'busy' should have survived the crash");

    let mut lost_quiet = Vec::new();
    for i in 0..QUIET_KEYS {
        if db.get_ns(quiet, quiet_key(i).as_bytes()).unwrap().as_deref() != Some(&b"quiet-value"[..]) {
            lost_quiet.push(i);
        }
    }
    let mut lost_busy = Vec::new();
    for i in 0..BUSY_KEYS {
        if db.get_ns(busy, busy_key(i).as_bytes()).unwrap().as_deref() != Some(&b"busy-value"[..]) {
            lost_busy.push(i);
        }
    }

    assert!(
        lost_quiet.is_empty() && lost_busy.is_empty(),
        "acknowledged writes did not survive the kill: {} of {QUIET_KEYS} lost from the \
             never-flushed namespace, {} of {BUSY_KEYS} from the flushing one",
        lost_quiet.len(),
        lost_busy.len()
    );

    db.shutdown().unwrap();
}
