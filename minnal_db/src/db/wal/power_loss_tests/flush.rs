//! Power losses around a memtable flush: the WAL progress it saves, and the
//! value-log records its L0 files point at.

use super::*;

/// A write whose fsync fails stops holding the cut back, but its frame stays
/// in the live tail, so a flush after it saves a cut past the durable tail. If
/// a power loss then takes that frame, the reopened log must still put new
/// writes above the saved cut, or the next recovery skips them: an
/// acknowledged write lost.
#[test]
fn progress_saved_past_a_failed_fsync_does_not_hide_later_writes() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    let ns = db.create_namespace("ns").unwrap();
    db.put_ns(ns, b"before", b"v").unwrap();
    db.wal.fail_next_sync();
    assert!(matches!(db.put_ns(ns, b"failed", &[b'x'; 4096]), Err(KVError::WalPoisoned(_))));
    db.get_store(ns).unwrap().flush_memtable_to_level0().unwrap();
    let saved = db.wal_metadata.read().clone();
    assert!(
        saved.cut > db.wal.durable_tail(),
        "setup: the flush should save a cut past the durable tail"
    );

    // Power loss: the failed frame is gone.
    let (_first, reopened) = open_image(&db);
    let ns1 = reopened.get_namespace_id("ns").unwrap();
    reopened.put_ns(ns1, b"after", b"acknowledged").unwrap();
    // A second power loss, before anything is flushed.
    let (_second, again) = open_image(&reopened);
    let ns2 = again.get_namespace_id("ns").unwrap();
    assert_eq!(again.get_ns(ns2, b"before").unwrap(), Some(b"v".to_vec()));
    assert_eq!(again.get_ns(ns2, b"failed").unwrap(), None);
    assert_eq!(
        again.get_ns(ns2, b"after").unwrap(),
        Some(b"acknowledged".to_vec()),
        "a write acknowledged after the reopen was skipped by the next recovery"
    );
}

/// An L0 file is durable the moment a flush writes it: open finds L0 files by
/// listing their directories, registered or not. So a power loss right after
/// the write, before anything else of the flush, must find every value the
/// file points at. The value log was fsynced only at the written event, after
/// the file was renamed and its directory fsynced: in that window the durable
/// L0 entry pointed at a lost record, and replay did not heal it, since it
/// drops a WAL record whose version equals the stored copy's (the same write).
/// The read failed with `CorruptedLog`. A GC relocation or a no-WAL write, in
/// no log, had nothing to heal it at all.
#[test]
fn an_l0_file_never_points_at_a_value_a_power_loss_could_take() {
    use crate::storage::lsm::tree::LsmFlushObserver;
    use crate::storage::value_log::ValueLog;
    use std::path::{Path, PathBuf};

    fn copy_dir(from: &Path, to: &Path) {
        std::fs::create_dir_all(to).unwrap();
        for entry in std::fs::read_dir(from).unwrap() {
            let entry = entry.unwrap();
            let target = to.join(entry.file_name());
            if entry.file_type().unwrap().is_dir() {
                copy_dir(&entry.path(), &target);
            } else {
                std::fs::copy(entry.path(), &target).unwrap();
            }
        }
    }

    /// At the written event (the L0 files are on disk, nothing else of the
    /// flush has happened), adds them to `image` with the value log as a power
    /// loss would leave it then.
    struct PowerLossWhenWritten {
        inner: Arc<dyn LsmFlushObserver>,
        logs: Vec<Arc<ValueLog>>,
        live: PathBuf,
        image: PathBuf,
        lsm_dir: PathBuf,
    }
    impl LsmFlushObserver for PowerLossWhenWritten {
        fn on_memtable_sealed(&self, version: u64) {
            self.inner.on_memtable_sealed(version);
        }
        fn on_ro_memtable_flushing(&self, version: u64) -> std::io::Result<()> {
            self.inner.on_ro_memtable_flushing(version)
        }
        fn on_ro_memtable_written(&self, version: u64) -> std::io::Result<()> {
            let relative = self.lsm_dir.strip_prefix(&self.live).unwrap();
            copy_dir(&self.lsm_dir, &self.image.join(relative));
            for log in &self.logs {
                let (file, durable) = log.power_loss_extent();
                let copy = self.image.join(file.strip_prefix(&self.live).unwrap());
                std::fs::copy(&file, &copy).unwrap();
                std::fs::OpenOptions::new().write(true).open(&copy).unwrap().set_len(durable).unwrap();
            }
            self.inner.on_ro_memtable_written(version)
        }
        fn on_ro_memtable_flushed_to_level0(&self, version: u64) {
            self.inner.on_ro_memtable_flushed_to_level0(version);
        }
    }

    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    let ns = db.create_namespace("ns").unwrap();
    db.put_ns(ns, b"k", b"acknowledged").unwrap();
    let store = db.get_store(ns).unwrap();
    let unsynced: u64 = store.value_log.bucket_logs().iter().map(|l| l.unsynced_bytes()).sum();
    assert!(unsynced > 0, "setup: the value is not fsynced in the value log yet");

    let image_dir = TempDir::new().unwrap();
    let image = image_dir.path().join("db");
    db.power_loss_image(&image).unwrap();
    let ns_dir = crate::layout::namespace_data_dir(dir.path(), "ns");
    store.lsm.set_flush_observer(Some(Arc::new(PowerLossWhenWritten {
        inner: store.lsm.flush_observer().unwrap(),
        logs: store.value_log.bucket_logs(),
        live: dir.path().to_path_buf(),
        image: image.clone(),
        lsm_dir: crate::layout::namespace_lsm_dir(&ns_dir),
    })));
    store.flush_memtable_to_level0().unwrap();

    let reopened = Database::open(&image, create_db_config()).unwrap();
    let ns = reopened.get_namespace_id("ns").unwrap();
    assert_eq!(
        reopened.get_ns(ns, b"k").unwrap(),
        Some(b"acknowledged".to_vec()),
        "a power loss after the L0 write lost an acknowledged value"
    );
}
