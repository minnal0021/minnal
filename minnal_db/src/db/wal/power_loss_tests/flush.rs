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

/// Copy `from` to `to`, recursively.
fn copy_dir(from: &std::path::Path, to: &std::path::Path) {
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

/// Takes a power-loss image of a store at its next flush's written event: the
/// L0 files are on disk, the value log holds only what was fsynced, and
/// nothing else of the flush (the fsync before the WAL progress is saved, the
/// save) has happened. Wraps the store's own observer, so it runs first.
struct PowerLossWhenWritten {
    inner: Arc<dyn crate::storage::lsm::tree::LsmFlushObserver>,
    logs: Vec<Arc<crate::storage::value_log::ValueLog>>,
    live: std::path::PathBuf,
    image: std::path::PathBuf,
    lsm_dir: std::path::PathBuf,
    /// Bytes of the value log the image lost (appended, never fsynced).
    lost: AtomicUsize,
    taken: AtomicBool,
}

impl PowerLossWhenWritten {
    /// Install on `ns`'s store; the image of `db` (taken now, before the
    /// flush) is completed at the written event.
    fn install(db: &Database, ns: NamespaceId, ns_name: &str, image: &std::path::Path) -> Arc<Self> {
        let store = db.get_store(ns).unwrap();
        db.power_loss_image(image).unwrap();
        let ns_dir = crate::layout::namespace_data_dir(&db.db_path, ns_name);
        let observer = Arc::new(Self {
            inner: store.lsm.flush_observer().unwrap(),
            logs: store.value_log.bucket_logs(),
            live: db.db_path.clone(),
            image: image.to_path_buf(),
            lsm_dir: crate::layout::namespace_lsm_dir(&ns_dir),
            lost: AtomicUsize::new(0),
            taken: AtomicBool::new(false),
        });
        store.lsm.set_flush_observer(Some(observer.clone()));
        observer
    }
}

impl crate::storage::lsm::tree::LsmFlushObserver for PowerLossWhenWritten {
    fn on_memtable_sealed(&self, version: u64) {
        self.inner.on_memtable_sealed(version);
    }
    fn on_ro_memtable_flushing(&self, version: u64) -> std::io::Result<()> {
        self.inner.on_ro_memtable_flushing(version)
    }
    fn on_ro_memtable_written(&self, version: u64) -> std::io::Result<()> {
        if !self.taken.swap(true, Ordering::SeqCst) {
            let relative = self.lsm_dir.strip_prefix(&self.live).unwrap();
            copy_dir(&self.lsm_dir, &self.image.join(relative));
            for log in &self.logs {
                let (file, durable) = log.power_loss_extent();
                self.lost.fetch_add(log.unsynced_bytes() as usize, Ordering::SeqCst);
                let copy = self.image.join(file.strip_prefix(&self.live).unwrap());
                std::fs::copy(&file, &copy).unwrap();
                std::fs::OpenOptions::new().write(true).open(&copy).unwrap().set_len(durable).unwrap();
            }
        }
        self.inner.on_ro_memtable_written(version)
    }
    fn on_ro_memtable_flushed_to_level0(&self, version: u64) {
        self.inner.on_ro_memtable_flushed_to_level0(version);
    }
}

/// A WAL-backed write's value is fsynced before its WAL record is retired,
/// not before its L0 file is written (an fsync there cost concurrent writers
/// up to 14% of their median latency). A power loss between the two leaves a
/// durable L0 entry pointing at a value that is gone, and the WAL record
/// that can restore it. Replay finds the stored copy is the same write as
/// the record and its value missing, writes the value again and points the
/// key at it, at the copy's next system version (as a GC relocation does).
/// Dropped as "already stored", the read failed with `CorruptedLog`.
#[test]
fn replay_writes_again_a_value_a_power_loss_took_from_under_an_l0_file() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    let ns = db.create_namespace("ns").unwrap();
    db.put_ns(ns, b"k", b"acknowledged").unwrap();
    let image_dir = TempDir::new().unwrap();
    let image = image_dir.path().join("db");
    let power_loss = PowerLossWhenWritten::install(&db, ns, "ns", &image);
    db.get_store(ns).unwrap().flush_memtable_to_level0().unwrap();
    assert!(
        power_loss.lost.load(Ordering::SeqCst) > 0,
        "setup: the value is not fsynced when the L0 file is written"
    );

    let reopened = Database::open(&image, create_db_config()).unwrap();
    let ns = reopened.get_namespace_id("ns").unwrap();
    assert_eq!(
        reopened.get_ns(ns, b"k").unwrap(),
        Some(b"acknowledged".to_vec()),
        "a power loss after the L0 write lost an acknowledged value"
    );
    let store = reopened.get_store(ns).unwrap();
    let (_, version) = store.lsm.get_with_seq(b"k").unwrap().unwrap();
    assert_eq!(version.system, 1, "the key points at the value written again, one system version up");
    assert!(store.lsm.layer_order_violations().unwrap().is_empty());

    // Healed once: another power loss and replay of the same record leave it.
    let (_again_dir, again) = open_image(&reopened);
    let ns = again.get_namespace_id("ns").unwrap();
    assert_eq!(again.get_ns(ns, b"k").unwrap(), Some(b"acknowledged".to_vec()));
}

/// A value no log covers — a no-WAL write's, a GC copy — could not be written
/// again after a power loss, so a flush fsyncs it before an L0 file points at
/// it (`ValueLog::mark_unlogged`), while a logged write's waits for the WAL
/// progress save. Without that, the durable L0 entry pointed at a lost record
/// and the read failed with `CorruptedLog`.
#[test]
fn an_l0_file_never_points_at_an_unlogged_value_a_power_loss_could_take() {
    // A no-WAL write.
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    let ns = db.create_namespace("ns").unwrap();
    db.put_ns_no_wal(ns, b"k", b"unlogged").unwrap();
    let image_dir = TempDir::new().unwrap();
    let image = image_dir.path().join("db");
    let power_loss = PowerLossWhenWritten::install(&db, ns, "ns", &image);
    db.get_store(ns).unwrap().flush_memtable_to_level0().unwrap();
    assert!(power_loss.taken.load(Ordering::SeqCst), "setup: the flush wrote its L0 files");
    let reopened = Database::open(&image, create_db_config()).unwrap();
    let ns = reopened.get_namespace_id("ns").unwrap();
    assert_eq!(reopened.get_ns(ns, b"k").unwrap(), Some(b"unlogged".to_vec()));

    // GC's copies: every key's WAL record is retired before GC runs.
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    let ns = db.create_namespace("ns").unwrap();
    let store = db.get_store(ns).unwrap();
    let key = |i: u32| format!("key:{i:04}").into_bytes();
    for i in 0..200 {
        db.put_ns(ns, &key(i), &[b'a'; 512]).unwrap();
    }
    for i in (0..200).step_by(2) {
        db.put_ns(ns, &key(i), &[b'b'; 512]).unwrap();
    }
    store.flush_memtable_to_level0().unwrap();
    let image_dir = TempDir::new().unwrap();
    let image = image_dir.path().join("db");
    let power_loss = PowerLossWhenWritten::install(&db, ns, "ns", &image);
    let stats = store.garbage_collect_with_thresholds(0.0, 0.0).unwrap();
    assert!(stats.bytes_reclaimed > 0, "setup: GC relocated the survivors");
    assert!(power_loss.taken.load(Ordering::SeqCst), "setup: GC flushed its copies to L0");
    let reopened = Database::open(&image, create_db_config()).unwrap();
    let ns = reopened.get_namespace_id("ns").unwrap();
    for i in 0..200 {
        let expected = if i % 2 == 0 { [b'b'; 512] } else { [b'a'; 512] };
        assert_eq!(reopened.get_ns(ns, &key(i)).unwrap(), Some(expected.to_vec()), "key:{i:04}");
    }
}
