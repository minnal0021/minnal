//! Single-writer lock over a database directory.
//!
//! The lock is an advisory `flock(2)` held on `<dir>/.lock` for as long as the
//! guard lives. Two properties matter, and neither is provided by checking
//! whether the file exists:
//!
//! * **The kernel releases it when the owner dies.** A process killed with
//!   `SIGKILL`, OOM-killed, or lost to a power cut runs no destructors, so any
//!   scheme that relies on cleanup code leaves the database permanently
//!   un-openable. Every crash restart then needs a human to delete a file
//!   before recovery can even be attempted.
//! * **Acquisition is atomic.** `exists()` followed by `create` is a TOCTOU
//!   race: two processes starting together both see no file and both proceed
//!   to open the same data directory.
//!
//! The lock file itself is **not** the lock — it is only a handle to lock and a
//! place to record the owner's PID for diagnostics. It is deliberately never
//! unlinked. Unlinking a path another process already holds open would leave
//! that process locking an orphaned inode while a third process creates a fresh
//! file at the same path and locks that, producing two simultaneous "owners".
//! A leftover `.lock` file is inert.
//!
//! `flock` rather than `fcntl(F_SETLK)`: POSIX record locks are per-*process*,
//! so a second open-and-lock of the same file inside one process silently
//! succeeds. `flock` locks attach to the open file description, so opening the
//! same database twice in one process is caught too.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};

/// Name of the lock file inside the database directory.
pub(crate) const LOCK_FILE_NAME: &str = ".lock";

/// Why a directory could not be locked.
#[derive(Debug)]
pub(crate) enum LockError {
    /// Another live process holds the lock. Carries the owner's PID when the
    /// lock file could be read and parsed.
    Held { pid: Option<u32> },
    /// The lock file could not be opened or locked for some other reason.
    Io(std::io::Error),
}

impl From<std::io::Error> for LockError {
    fn from(e: std::io::Error) -> Self {
        LockError::Io(e)
    }
}

/// An exclusive lock on a database directory, released when dropped — or when
/// the process dies, by the kernel.
///
/// Hold this for as long as the database is open. Because the guard is a plain
/// value, an error path that returns before it is stored releases the lock
/// automatically; the previous file-based scheme leaked the lock whenever
/// opening failed after the file had been created.
#[derive(Debug)]
pub(crate) struct DirLock {
    /// Closing this fd is what releases the lock. Never unlink the path.
    _file: File,
    path: PathBuf,
}

impl DirLock {
    /// Take the exclusive lock on `dir`, failing immediately if it is held.
    ///
    /// The directory must already exist.
    pub(crate) fn acquire(dir: impl AsRef<Path>) -> Result<Self, LockError> {
        let path = dir.as_ref().join(LOCK_FILE_NAME);
        // Not `create_new`: the file surviving a previous run is expected and
        // says nothing about whether anyone holds the lock.
        let mut file = OpenOptions::new().read(true).write(true).create(true).truncate(false).open(&path)?;

        // SAFETY: `fd` is owned by `file` and stays open for the duration of the
        // call; `flock` only records lock state against it.
        let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if rc != 0 {
            let err = std::io::Error::last_os_error();
            return match err.raw_os_error() {
                Some(code) if code == libc::EWOULDBLOCK || code == libc::EAGAIN => Err(LockError::Held { pid: read_owner_pid(&path) }),
                _ => Err(LockError::Io(err)),
            };
        }

        // Record the owner for diagnostics. Best-effort: a failure here means a
        // less informative message for whoever is locked out later, which is not
        // worth failing an otherwise successful open over.
        let _ = write_owner_pid(&mut file);

        Ok(DirLock { _file: file, path })
    }

    /// Path of the lock file, for logging.
    #[allow(dead_code)]
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

/// Overwrite the lock file with the current process's PID.
fn write_owner_pid(file: &mut File) -> std::io::Result<()> {
    let pid = std::process::id();
    file.set_len(0)?;
    file.seek(SeekFrom::Start(0))?;
    // The PID must come first and alone on its line so the reader stays trivial.
    write!(file, "{pid}\nminnal: pid of the process holding this database open\n")?;
    file.flush()
}

/// Read the PID recorded by the lock holder.
///
/// Returns `None` when the file cannot be read or does not start with a number
/// — the holder may not have written it yet, or the file may predate this
/// format. Diagnostics only; never load-bearing.
fn read_owner_pid(path: &Path) -> Option<u32> {
    let mut contents = String::new();
    File::open(path).ok()?.take(64).read_to_string(&mut contents).ok()?;
    contents.lines().next()?.trim().parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn second_acquire_fails_while_the_first_is_alive() {
        let dir = TempDir::new().unwrap();
        let _held = DirLock::acquire(dir.path()).expect("first acquire should succeed");

        match DirLock::acquire(dir.path()) {
            Err(LockError::Held { pid }) => {
                assert_eq!(pid, Some(std::process::id()), "the holder's pid should be reported");
            }
            Err(e) => panic!("expected Held, got {e:?}"),
            Ok(_) => panic!("two locks on one directory were granted at once"),
        }
    }

    #[test]
    fn dropping_the_guard_releases_the_lock() {
        let dir = TempDir::new().unwrap();
        {
            let _held = DirLock::acquire(dir.path()).unwrap();
        }
        DirLock::acquire(dir.path()).expect("lock should be free once the guard is dropped");
    }

    /// The regression test for the reported bug: a crash leaves the lock file on
    /// disk with no live owner, and that must not block the next open. The old
    /// existence check failed exactly here, making every crash need a manual
    /// `rm` before recovery could run.
    #[test]
    fn a_stale_lock_file_from_a_crashed_process_does_not_block_acquisition() {
        let dir = TempDir::new().unwrap();
        // Exactly what a SIGKILLed process leaves behind: the file, with its
        // (now meaningless) pid in it, and no holder.
        std::fs::write(dir.path().join(LOCK_FILE_NAME), "999999\nminnal: stale\n").unwrap();

        let lock = DirLock::acquire(dir.path()).expect("a stale lock file must not block acquisition");
        assert_eq!(read_owner_pid(&dir.path().join(LOCK_FILE_NAME)), Some(std::process::id()));
        drop(lock);
    }

    #[test]
    fn the_lock_file_survives_release_rather_than_being_unlinked() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join(LOCK_FILE_NAME);
        {
            let _held = DirLock::acquire(dir.path()).unwrap();
            assert!(path.exists());
        }
        // Unlinking on release would let a third process lock a fresh file at
        // the same path while a second still holds the old inode.
        assert!(path.exists(), "the lock file must outlive the guard");
    }

    #[test]
    fn acquiring_records_this_process_id() {
        let dir = TempDir::new().unwrap();
        let _held = DirLock::acquire(dir.path()).unwrap();
        assert_eq!(read_owner_pid(&dir.path().join(LOCK_FILE_NAME)), Some(std::process::id()));
    }
}
