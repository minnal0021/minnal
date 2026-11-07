//! Log segments zero-filled ahead: the next segment of a log, written in full
//! in the background before the log needs it.
//!
//! Both logs use this. An `fdatasync` after a write into blocks that already
//! hold data commits neither a new file size nor a block allocation, so it
//! costs about a third of one on a file that grows. Once the active segment is
//! half full the log calls [`SegmentPrep::start`]: a thread writes the next
//! segment under a temporary `.prep` name — zeros, with the log's header at its
//! start — and fsyncs it. When the log moves to that segment it calls
//! [`SegmentPrep::install`], which renames the file into place and fsyncs the
//! directory, waiting for the thread if it has not finished.
//!
//! The temporary name is what keeps a half-written file out of the log: each
//! log's recovery takes "the next segment's file exists" to mean the log
//! continues there, so each deletes leftover `.prep` files at open
//! ([`remove_leftovers`]). Design: `minnal_db/docs/Main-WAL-Rework-Design.md`
//! §17, §18.

use parking_lot::Mutex;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::JoinHandle;

/// One preparation under way (or finished, not yet installed).
struct Pending {
    segment_id: u64,
    prep: PathBuf,
    thread: JoinHandle<io::Result<()>>,
}

/// See the module docs. Dropping it waits for a preparation under way and
/// removes its file.
pub(crate) struct SegmentPrep {
    /// The log's name, for log messages.
    log: &'static str,
    pending: Mutex<Option<Pending>>,
    /// One more than the highest segment id a preparation was started for (0:
    /// none), so [`started`](Self::started), asked on every append past half
    /// a segment, reads an atomic instead of taking `pending`. Segment ids only
    /// grow, so this answers the same question.
    started_below: AtomicU64,
}

impl SegmentPrep {
    /// No preparation under way. `log` names the log in messages.
    pub(crate) fn new(log: &'static str) -> Self {
        Self {
            log,
            pending: Mutex::new(None),
            started_below: AtomicU64::new(0),
        }
    }

    /// Whether a preparation of segment `segment_id`, or a later one, was ever
    /// started (it may since have been installed). A lock-free check before
    /// building [`start`](Self::start)'s arguments.
    pub(crate) fn started(&self, segment_id: u64) -> bool {
        self.started_below.load(Ordering::Acquire) > segment_id
    }

    /// Start writing segment `segment_id` at `prep`: `size` bytes of zeros
    /// with `header` at the start, fsynced. Does nothing if that segment, or a
    /// later one, is already being prepared. If the thread cannot be started,
    /// the log creates the segment itself when it gets there.
    pub(crate) fn start(&self, segment_id: u64, prep: PathBuf, size: u64, header: Vec<u8>) {
        let mut pending = self.pending.lock();
        if pending.as_ref().is_some_and(|p| p.segment_id >= segment_id) {
            return;
        }
        let path = prep.clone();
        let thread = std::thread::Builder::new()
            .name("wal-prepare".into())
            .spawn(move || zero_fill(&path, size, &header));
        match thread {
            Ok(thread) => {
                *pending = Some(Pending { segment_id, prep, thread });
                self.started_below.fetch_max(segment_id + 1, Ordering::Release);
            }
            Err(e) => log::warn!(
                "[{}] could not start preparing segment {segment_id}; it will be created when needed: {e}",
                self.log
            ),
        }
    }

    /// Move the prepared segment `segment_id` to `path` and fsync the
    /// directory, so the name is durable before anything is acknowledged into
    /// it. Waits for an unfinished preparation. `Ok(false)` when there is none
    /// for this segment (one for another segment, or one that failed, is
    /// discarded): the caller creates the segment itself.
    pub(crate) fn install(&self, segment_id: u64, path: &Path) -> io::Result<bool> {
        let Some(p) = self.pending.lock().take() else {
            return Ok(false);
        };
        let done = p.thread.join().unwrap_or_else(|_| Err(io::Error::other("the preparing thread panicked")));
        if p.segment_id != segment_id || done.is_err() {
            if let Err(e) = &done {
                log::warn!(
                    "[{}] preparing segment {} failed; segment {segment_id} will be created when needed: {e}",
                    self.log,
                    p.segment_id
                );
            }
            let _ = std::fs::remove_file(&p.prep);
            return Ok(false);
        }
        std::fs::rename(&p.prep, path)?;
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            File::open(dir)?.sync_all()?;
        }
        Ok(true)
    }

    /// Test hook: the segment being prepared and whether its thread is done.
    #[cfg(test)]
    pub(crate) fn pending(&self) -> Option<(u64, bool)> {
        self.pending.lock().as_ref().map(|p| (p.segment_id, p.thread.is_finished()))
    }
}

impl Drop for SegmentPrep {
    fn drop(&mut self) {
        // A segment still being prepared is not needed: let its thread finish,
        // then remove the file (the next open would remove it anyway).
        if let Some(p) = self.pending.get_mut().take() {
            let _ = p.thread.join();
            let _ = std::fs::remove_file(&p.prep);
        }
    }
}

/// Write `path` in full: `size` bytes of zeros with `header` at the start,
/// then fsync it.
fn zero_fill(path: &Path, size: u64, header: &[u8]) -> io::Result<()> {
    let file = OpenOptions::new().create(true).truncate(true).write(true).open(path)?;
    let chunk = vec![0u8; (1 << 20).min(size as usize)];
    let mut offset = 0u64;
    while offset < size {
        let n = chunk.len().min((size - offset) as usize);
        file.write_all_at(&chunk[..n], offset)?;
        offset += n as u64;
    }
    if !header.is_empty() {
        file.write_all_at(header, 0)?;
    }
    file.sync_all()
}

/// Delete the files in `dir` whose names start with `prefix` and end in
/// `.prep`: segments a previous run left half-prepared, not part of the log.
pub(crate) fn remove_leftovers(dir: &Path, prefix: &str) {
    let dir = if dir.as_os_str().is_empty() { Path::new(".") } else { dir };
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with(prefix) && name.ends_with(".prep") {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wait_until_prepared(prep: &SegmentPrep) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while !prep.pending().is_some_and(|(_, done)| done) {
            assert!(std::time::Instant::now() < deadline, "the segment was never prepared");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    /// A prepared segment holds its header, then zeros, at full size, and is
    /// installed under its final name.
    #[test]
    fn a_prepared_segment_is_installed_whole() {
        let dir = tempfile::tempdir().unwrap();
        let prep = SegmentPrep::new("test");
        let tmp = dir.path().join("seg1.prep");
        prep.start(1, tmp.clone(), 8192, b"HEAD".to_vec());
        wait_until_prepared(&prep);
        assert!(tmp.exists());
        let path = dir.path().join("seg1");
        assert!(prep.install(1, &path).unwrap());
        assert!(!tmp.exists());
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.len(), 8192);
        assert_eq!(&bytes[..4], b"HEAD");
        assert!(bytes[4..].iter().all(|&b| b == 0));
    }

    /// A preparation for another segment is discarded, not installed.
    #[test]
    fn a_preparation_for_another_segment_is_discarded() {
        let dir = tempfile::tempdir().unwrap();
        let prep = SegmentPrep::new("test");
        let tmp = dir.path().join("seg2.prep");
        prep.start(2, tmp.clone(), 4096, Vec::new());
        assert!(!prep.install(1, &dir.path().join("seg1")).unwrap());
        assert!(!tmp.exists());
        assert!(!dir.path().join("seg1").exists());
    }

    /// Dropping it removes an uninstalled preparation; leftovers matching the
    /// prefix are removed, others kept.
    #[test]
    fn drop_and_open_leave_no_prepared_file() {
        let dir = tempfile::tempdir().unwrap();
        let tmp = dir.path().join("log.seg1.prep");
        {
            let prep = SegmentPrep::new("test");
            prep.start(1, tmp.clone(), 4096, Vec::new());
        }
        assert!(!tmp.exists());
        std::fs::write(dir.path().join("log.seg7.prep"), b"x").unwrap();
        std::fs::write(dir.path().join("other.prep"), b"x").unwrap();
        remove_leftovers(dir.path(), "log");
        assert!(!dir.path().join("log.seg7.prep").exists());
        assert!(dir.path().join("other.prep").exists());
    }
}
