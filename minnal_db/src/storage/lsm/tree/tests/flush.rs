//! Flush observer events, vetoed flushes, and concurrent flushes.

use super::*;

// Two threads flushing the same read-only memtable must produce ONE L0 file,
// and both must return only once it is durable. Without the per-memtable flush
// lock both would write it (a duplicate file; in the same millisecond, one file
// name with the second `File::create` truncating the first).
/// Records each flush event with the number of L0 files registered at that
/// moment, reading the tree through a weak handle set after it is built.
struct EventLog {
    tree: std::sync::OnceLock<std::sync::Weak<LSMTree>>,
    events: parking_lot::Mutex<Vec<(&'static str, u64, usize)>>,
}

impl EventLog {
    fn registered(&self) -> usize {
        let tree = self.tree.get().and_then(|t| t.upgrade()).expect("tree alive");
        (0..tree.config.num_buckets as u32).map(|b| tree.level0_entries_snapshot(b).len()).sum()
    }
}

impl LsmFlushObserver for EventLog {
    fn on_memtable_sealed(&self, _version: u64) {}
    fn on_ro_memtable_flushing(&self, version: u64) -> std::io::Result<()> {
        let n = self.registered();
        self.events.lock().push(("flushing", version, n));
        Ok(())
    }
    fn on_ro_memtable_written(&self, version: u64) -> std::io::Result<()> {
        let n = self.registered();
        self.events.lock().push(("written", version, n));
        Ok(())
    }
    fn on_ro_memtable_flushed_to_level0(&self, version: u64) {
        let n = self.registered();
        self.events.lock().push(("flushed", version, n));
    }
}

// The vector index WAL syncs in `on_ro_memtable_flushing`, makes its
// progress durable in `on_ro_memtable_written`, and relies on: the events
// come in that order, none of the memtable's L0 files is registered
// (mergeable) at either, and memtables arrive in version order even when
// several threads flush at once. All are pinned here.
#[test]
fn the_written_event_comes_before_registration_and_in_version_order() -> Result<()> {
    for round in 0..50 {
        let temp_dir = TempDir::new()?;
        let lsm = Arc::new(LSMTree::open(
            temp_dir.path(),
            LSMConfig {
                num_buckets: 2,
                ..LSMConfig::default()
            },
        )?);
        let log = Arc::new(EventLog {
            tree: std::sync::OnceLock::new(),
            events: parking_lot::Mutex::new(Vec::new()),
        });
        log.tree.set(Arc::downgrade(&lsm)).ok();
        lsm.set_flush_observer(Some(Arc::clone(&log) as Arc<dyn LsmFlushObserver>));
        let mut seq = 0u64;
        for _ in 0..3 {
            for i in 0..50u64 {
                seq += 1;
                lsm.insert_with_seq(format!("k{i:03}").as_bytes(), u128::from(seq), seq)?;
            }
            lsm.flush_memtable()?; // three read-only memtables, none on disk
        }
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let flushers: Vec<_> = (0..2)
            .map(|_| {
                let (lsm, barrier) = (Arc::clone(&lsm), Arc::clone(&barrier));
                std::thread::spawn(move || {
                    barrier.wait();
                    lsm.flush_memtable_to_level0()
                })
            })
            .collect();
        for flusher in flushers {
            flusher.join().expect("flusher panicked")?;
        }
        // 50 keys over 2 buckets: each memtable registers one file per bucket.
        let want: Vec<(&str, u64, usize)> = (0..3u64)
            .flat_map(|v| {
                [
                    ("flushing", v, 2 * v as usize),
                    ("written", v, 2 * v as usize),
                    ("flushed", v, 2 * (v as usize + 1)),
                ]
            })
            .collect();
        assert_eq!(*log.events.lock(), want, "round {round}");
    }
    Ok(())
}

/// Refuses the first `fail` flushing or written events (`at_flushing`
/// picks which), then accepts.
struct Veto {
    fail: std::sync::atomic::AtomicUsize,
    at_flushing: bool,
}

impl Veto {
    fn refuse(&self) -> std::io::Result<()> {
        let left = self.fail.load(Ordering::SeqCst);
        if left == 0 {
            return Ok(());
        }
        self.fail.store(left - 1, Ordering::SeqCst);
        Err(std::io::Error::other("cannot record the flush"))
    }
}

impl LsmFlushObserver for Veto {
    fn on_memtable_sealed(&self, _version: u64) {}
    fn on_ro_memtable_flushing(&self, _version: u64) -> std::io::Result<()> {
        if self.at_flushing { self.refuse() } else { Ok(()) }
    }
    fn on_ro_memtable_written(&self, _version: u64) -> std::io::Result<()> {
        if self.at_flushing { Ok(()) } else { self.refuse() }
    }
    fn on_ro_memtable_flushed_to_level0(&self, _version: u64) {}
}

// A flushing or written event that fails must leave nothing registered and
// no file behind, and the memtable must stay readable and be flushed by a
// retry.
#[test]
fn a_failed_flush_event_undoes_the_flush_and_a_retry_completes_it() -> Result<()> {
    for at_flushing in [true, false] {
        a_failed_flush_event_undoes_the_flush(at_flushing)?;
    }
    Ok(())
}

fn a_failed_flush_event_undoes_the_flush(at_flushing: bool) -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(
        temp_dir.path(),
        LSMConfig {
            num_buckets: 2,
            ..LSMConfig::default()
        },
    )?;
    lsm.set_flush_observer(Some(Arc::new(Veto {
        fail: std::sync::atomic::AtomicUsize::new(1),
        at_flushing,
    })));
    for i in 0..20u64 {
        lsm.insert_with_seq(format!("k{i:02}").as_bytes(), u128::from(i), i + 1)?;
    }
    let registered = |lsm: &LSMTree| (0..2).map(|b| lsm.level0_entries_snapshot(b).len()).sum::<usize>();
    let l0_files = || {
        std::fs::read_dir(temp_dir.path().join("level0"))
            .map(|d| d.flatten().filter(|e| e.path().is_file()).count())
            .unwrap_or(0)
    };

    assert!(lsm.flush_memtable_to_level0().is_err(), "the vetoed flush reports its error");
    assert_eq!(registered(&lsm), 0, "nothing registered");
    assert_eq!(l0_files(), 0, "no L0 file left behind");
    assert_eq!(lsm.get(b"k07")?, Some(7), "the read-only memtable still serves reads");

    lsm.flush_memtable_to_level0()?;
    assert_eq!(registered(&lsm), 2, "the retry registers one file per bucket");
    assert_eq!(lsm.get(b"k07")?, Some(7));
    Ok(())
}

#[test]
fn test_concurrent_flushes_of_one_memtable_write_it_once() -> Result<()> {
    for round in 0..100 {
        let temp_dir = TempDir::new()?;
        let lsm = Arc::new(LSMTree::open(
            temp_dir.path(),
            LSMConfig {
                num_buckets: 1,
                ..LSMConfig::default()
            },
        )?);
        for i in 0..500u64 {
            lsm.insert_with_seq(format!("k{i:05}").as_bytes(), u128::from(i) + 1, i + 1)?;
        }
        lsm.flush_memtable()?; // one read-only memtable, not yet on disk
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let flushers: Vec<_> = (0..2)
            .map(|_| {
                let (lsm, barrier) = (Arc::clone(&lsm), Arc::clone(&barrier));
                std::thread::spawn(move || {
                    barrier.wait();
                    lsm.flush_memtable_to_level0()
                })
            })
            .collect();
        for flusher in flushers {
            flusher.join().expect("flusher panicked")?;
        }
        assert_eq!(lsm.level0_entries_snapshot(0).len(), 1, "round {round}: the memtable was flushed twice");
        lsm.purge_ro_memtables_for_test(); // read from L0 alone
        for i in 0..500u64 {
            assert_eq!(lsm.get(format!("k{i:05}").as_bytes())?, Some(u128::from(i) + 1), "round {round}: key {i}");
        }
    }
    Ok(())
}

// GC relocation re-inserts a key at its existing seq with a new pointer, so the
// old and relocated copies tie on seq and only L0 file order says which is
// newer. Were file order taken from the raw wall clock, a clock step backwards
// between the two flushes would invert it: every read (and the L0→L1 merge)
// would choose the stale pointer, whose segment GC then unlinks. Simulated here by moving
// the first file's stamp into the future — the state a clock step leaves
// behind — and reopening, which also checks the stamp floor is seeded from
// disk.
#[test]
fn test_l0_order_survives_the_wall_clock_stepping_back() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let config = LSMConfig {
        num_buckets: 1,
        ..LSMConfig::default()
    };
    let (original, relocated) = (0xA_u128, 0xB_u128);
    {
        let lsm = LSMTree::open(temp_dir.path(), config.clone())?;
        lsm.insert_with_seq(b"k", original, 5)?;
        lsm.flush_memtable_to_level0()?;
    }
    // The clock that named that file was an hour ahead of the one we have now.
    let l0_dir = LSMTree::level0_bucket_dir_from(temp_dir.path(), 0);
    let first = std::fs::read_dir(&l0_dir)?.next().expect("one L0 file")?.path();
    let stamp = LSMTree::level0_created_at_from_path(&first);
    let hour = 3_600_000u128 * 1_000_000;
    std::fs::rename(&first, l0_dir.join(format!("{}.dat", stamp + hour)))?;

    let lsm = LSMTree::open(temp_dir.path(), config.clone())?;
    lsm.insert_with_seq(b"k", relocated, 5)?; // GC relocation: same seq, new pointer
    lsm.flush_memtable_to_level0()?;
    // Drop the read-only memtable (still holding the relocation) so reads must
    // settle the tie by L0 file order alone.
    lsm.purge_ro_memtables_for_test();

    assert_eq!(lsm.get(b"k")?, Some(relocated), "point read chose the pre-relocation copy");
    assert_eq!(lsm.get_multiple(&[b"k".to_vec()])?[0].map(|(p, _)| p), Some(relocated), "get_multiple");
    lsm.compact_all()?;
    assert_eq!(lsm.get(b"k")?, Some(relocated), "the L0→L1 merge baked in the stale copy");

    // And in-process: a candidate stamp from a clock that went backwards is
    // still issued above everything before it.
    let before = *lsm.last_l0_stamp.lock();
    assert!(lsm.next_l0_stamp(before - hour) > before, "a stamp went backwards");
    Ok(())
}
