//! Per-namespace dense row-ID map.
//!
//! Assigns **dense, monotonic** row IDs (0, 1, 2, …) to keys so the field-index
//! [`RoaringBitmap`](crate::RoaringBitmap)s pack densely — consecutive IDs share
//! a high key and fill a container instead of scattering one-per-doc the way
//! random hash IDs do. Replaces the stateless hash row-ID derivation.
//!
//! Two directions plus a counter, persisted as an mmap sidecar that is flushed
//! at the index checkpoint and rebuilt by WAL replay on open (the same
//! durability model as the field index — it is a *derived* structure):
//!
//! - **`key → id`** — an open-addressing hash table over the **full key bytes**
//!   (not a hash of them, so there are no ID collisions). Consulted on every put,
//!   delete, and replay. `O(1)` expected lookup / `O(1)` amortised insert. The
//!   slot table is in-memory (anonymous mmap) and **rebuilt from the id array on
//!   open**, so it is never a persisted source of truth.
//! - **`id → key`** — an append-only array indexed directly by the dense ID
//!   (`rows.idarray`), pointing into an append-only key-bytes region
//!   (`rows.keybytes`). `O(1)` direct lookup, used to resolve query hits back to
//!   keys.
//! - **counter** — `next_id`, persisted in the marker.
//!
//! ## Durability
//!
//! Writes mutate the mmaps in memory; nothing is fsynced per write. [`flush`]
//! msyncs the append-only regions and then writes the `rowmap.ckpt` marker
//! (`{next_id, keybytes_pos, wal_offset}`) via tmp-then-rename+fsync — the atomic
//! commit point. Any bytes appended past the marker's recorded lengths are
//! ignored on [`open`] (treated as a torn, un-checkpointed tail) and rebuilt by
//! WAL replay, so a crash mid-flush never yields an inconsistent map.
//!
//! Entries are **never removed** (a deleted-then-recreated key reuses its ID), so
//! the table has no tombstones and `count == next_id` always.
//!
//! [`flush`]: RowMap::flush
//! [`open`]: RowMap::open

use std::io;
use std::path::{Path, PathBuf};

use crate::blob_store::GrowableMmap;

// ── Layout constants ──────────────────────────────────────────────────────────

const KEYBYTES_FILE: &str = "rows.keybytes";
const IDARRAY_FILE: &str = "rows.idarray";
const MARKER_FILE: &str = "rowmap.ckpt";

const MARKER_MAGIC: u64 = 0x4D494E4E414C524D; // "MINNALRM"
const MARKER_VERSION: u32 = 1;
const MARKER_SIZE: usize = 40;

/// `[key_off: u64 LE | key_len: u32 LE]` per ID in `rows.idarray`.
const ID_ENTRY_SIZE: usize = 12;

const SLOT_SIZE: usize = 40;
const SLOT_EMPTY: u8 = 0;
const SLOT_OCCUPIED: u8 = 1;

const INITIAL_SLOT_CAP: usize = 16;
const INITIAL_KEYBYTES: usize = 4096;
const INITIAL_IDARRAY: usize = INITIAL_SLOT_CAP * ID_ENTRY_SIZE;

// ── Slot (in-memory hash table) ────────────────────────────────────────────────
//
// Byte layout (40 bytes):
//   0       state: u8
//   1..8    pad
//   8..16   hash:    u64 LE  (full-byte hash, for fast reject before key compare)
//   16..24  key_off: u64 LE  (offset into the key-bytes region)
//   24..28  key_len: u32 LE
//   28..32  pad
//   32..40  id:      u64 LE

#[derive(Clone, Copy)]
struct Slot {
    state: u8,
    hash: u64,
    key_off: u64,
    key_len: u32,
    id: u64,
}

fn read_slot(data: &[u8], i: usize) -> Slot {
    let b = i * SLOT_SIZE;
    Slot {
        state: data[b],
        hash: u64::from_le_bytes(data[b + 8..b + 16].try_into().unwrap()),
        key_off: u64::from_le_bytes(data[b + 16..b + 24].try_into().unwrap()),
        key_len: u32::from_le_bytes(data[b + 24..b + 28].try_into().unwrap()),
        id: u64::from_le_bytes(data[b + 32..b + 40].try_into().unwrap()),
    }
}

fn write_slot(data: &mut [u8], i: usize, s: &Slot) {
    let b = i * SLOT_SIZE;
    data[b] = s.state;
    data[b + 1..b + 8].fill(0);
    data[b + 8..b + 16].copy_from_slice(&s.hash.to_le_bytes());
    data[b + 16..b + 24].copy_from_slice(&s.key_off.to_le_bytes());
    data[b + 24..b + 28].copy_from_slice(&s.key_len.to_le_bytes());
    data[b + 28..b + 32].fill(0);
    data[b + 32..b + 40].copy_from_slice(&s.id.to_le_bytes());
}

/// FNV-1a over the raw key bytes.
fn hash_bytes(key: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in key {
        h ^= b as u64;
        h = h.wrapping_mul(0x00000100000001b3);
    }
    h
}

// ── RowMap ──────────────────────────────────────────────────────────────────────

/// Per-namespace dense row-ID map. See the module docs.
pub struct RowMap {
    dir: PathBuf,
    /// Append-only raw key bytes.
    keybytes: GrowableMmap,
    /// Append-only `id → (key_off, key_len)` array, indexed by ID.
    idarray: GrowableMmap,
    /// In-memory open-addressing `key → id` table (anonymous; rebuilt on open).
    slots: GrowableMmap,
    cap: usize,
    next_id: u64,
    keybytes_pos: usize,
}

impl RowMap {
    /// Create a fresh, empty row map in `dir` (creating the directory).
    pub fn create(dir: &Path) -> io::Result<Self> {
        std::fs::create_dir_all(dir)?;
        let keybytes = GrowableMmap::create_file(&dir.join(KEYBYTES_FILE), INITIAL_KEYBYTES)?;
        let idarray = GrowableMmap::create_file(&dir.join(IDARRAY_FILE), INITIAL_IDARRAY)?;
        let slots = GrowableMmap::new_anon(INITIAL_SLOT_CAP * SLOT_SIZE)?;
        Ok(Self {
            dir: dir.to_path_buf(),
            keybytes,
            idarray,
            slots,
            cap: INITIAL_SLOT_CAP,
            next_id: 0,
            keybytes_pos: 0,
        })
    }

    /// Open an existing row map, or create a fresh one if no marker exists.
    ///
    /// The in-memory `key → id` table is rebuilt from `rows.idarray[0..next_id]`,
    /// so any torn tail appended past the marker is ignored.
    pub fn open(dir: &Path) -> io::Result<Self> {
        let Some((next_id, keybytes_pos)) = read_marker(&dir.join(MARKER_FILE))? else {
            return Self::create(dir);
        };

        let keybytes = GrowableMmap::open_file(&dir.join(KEYBYTES_FILE))?;
        let idarray = GrowableMmap::open_file(&dir.join(IDARRAY_FILE))?;

        // Size the slot table for next_id entries at < 0.7 load, power-of-two.
        let mut cap = INITIAL_SLOT_CAP;
        while (next_id as usize) * 10 >= cap * 7 {
            cap *= 2;
        }
        let slots = GrowableMmap::new_anon(cap * SLOT_SIZE)?;

        let mut me = Self {
            dir: dir.to_path_buf(),
            keybytes,
            idarray,
            slots,
            cap,
            next_id,
            keybytes_pos,
        };

        // Rebuild the hash table from the durable id array.
        for id in 0..next_id {
            let (off, len) = me.id_entry(id);
            let key = me.keybytes.as_slice()[off as usize..off as usize + len as usize].to_vec();
            let hash = hash_bytes(&key);
            let idx = me.find_empty(&key, hash);
            write_slot(
                me.slots.as_mut_slice(),
                idx,
                &Slot {
                    state: SLOT_OCCUPIED,
                    hash,
                    key_off: off,
                    key_len: len,
                    id,
                },
            );
        }
        Ok(me)
    }

    /// Number of distinct IDs allocated (also the next ID to be assigned).
    pub fn next_id(&self) -> u64 {
        self.next_id
    }

    /// True if no IDs have been allocated.
    pub fn is_empty(&self) -> bool {
        self.next_id == 0
    }

    /// Return the existing ID for `key`, allocating a new dense one if unseen.
    pub fn get_or_alloc(&mut self, key: &[u8]) -> u128 {
        let hash = hash_bytes(key);
        match self.probe(key, hash) {
            Probe::Found(id) => id as u128,
            Probe::Empty(slot_idx) => {
                if self.needs_rehash() {
                    self.rehash();
                    // Re-probe in the resized table.
                    let slot_idx = match self.probe(key, hash) {
                        Probe::Found(id) => return id as u128,
                        Probe::Empty(i) => i,
                    };
                    return self.insert_at(slot_idx, key, hash);
                }
                self.insert_at(slot_idx, key, hash)
            }
        }
    }

    /// Return the ID for `key`, or `None` if it was never allocated.
    pub fn get(&self, key: &[u8]) -> Option<u128> {
        match self.probe(key, hash_bytes(key)) {
            Probe::Found(id) => Some(id as u128),
            Probe::Empty(_) => None,
        }
    }

    /// Resolve a dense ID back to its key bytes.
    pub fn key_for(&self, id: u128) -> Option<Vec<u8>> {
        let id = u64::try_from(id).ok()?;
        if id >= self.next_id {
            return None;
        }
        let (off, len) = self.id_entry(id);
        Some(self.keybytes.as_slice()[off as usize..off as usize + len as usize].to_vec())
    }

    /// Flush the append-only regions and atomically commit the marker recording
    /// `wal_offset` as the WAL position this map reflects. The caller must flush
    /// the row map **before** any field index in the same checkpoint pass so the
    /// map stays at least as durable as every persisted bitmap bit.
    pub fn flush(&self, wal_offset: u64) -> io::Result<()> {
        self.keybytes.flush()?;
        self.idarray.flush()?;
        write_marker(&self.dir, self.next_id, self.keybytes_pos as u64, wal_offset)
    }

    // ── internals ──────────────────────────────────────────────────────────────

    fn insert_at(&mut self, slot_idx: usize, key: &[u8], hash: u64) -> u128 {
        let id = self.next_id;
        let key_off = self.keybytes_pos as u64;
        let key_len = key.len() as u32;

        // Append the key bytes.
        self.keybytes
            .ensure_capacity(self.keybytes_pos + key.len())
            .expect("keybytes grow failed");
        self.keybytes.as_mut_slice()[self.keybytes_pos..self.keybytes_pos + key.len()].copy_from_slice(key);
        self.keybytes_pos += key.len();

        // Append the id → (off, len) entry.
        let base = id as usize * ID_ENTRY_SIZE;
        self.idarray.ensure_capacity(base + ID_ENTRY_SIZE).expect("idarray grow failed");
        let ida = self.idarray.as_mut_slice();
        ida[base..base + 8].copy_from_slice(&key_off.to_le_bytes());
        ida[base + 8..base + 12].copy_from_slice(&key_len.to_le_bytes());

        write_slot(
            self.slots.as_mut_slice(),
            slot_idx,
            &Slot {
                state: SLOT_OCCUPIED,
                hash,
                key_off,
                key_len,
                id,
            },
        );
        self.next_id += 1;
        id as u128
    }

    fn id_entry(&self, id: u64) -> (u64, u32) {
        let base = id as usize * ID_ENTRY_SIZE;
        let d = self.idarray.as_slice();
        let off = u64::from_le_bytes(d[base..base + 8].try_into().unwrap());
        let len = u32::from_le_bytes(d[base + 8..base + 12].try_into().unwrap());
        (off, len)
    }

    fn key_eq(&self, slot: &Slot, key: &[u8]) -> bool {
        slot.key_len as usize == key.len() && &self.keybytes.as_slice()[slot.key_off as usize..slot.key_off as usize + slot.key_len as usize] == key
    }

    fn probe(&self, key: &[u8], hash: u64) -> Probe {
        let start = (hash as usize) % self.cap;
        let data = self.slots.as_slice();
        let mut i = start;
        loop {
            let s = read_slot(data, i);
            match s.state {
                SLOT_EMPTY => return Probe::Empty(i),
                SLOT_OCCUPIED if s.hash == hash && self.key_eq(&s, key) => return Probe::Found(s.id),
                _ => {}
            }
            i = (i + 1) % self.cap;
            if i == start {
                return Probe::Empty(start);
            }
        }
    }

    /// Find the slot a new `key` would occupy (no duplicate-key check — callers
    /// only use this during a from-scratch rebuild where keys are unique).
    fn find_empty(&self, _key: &[u8], hash: u64) -> usize {
        let start = (hash as usize) % self.cap;
        let data = self.slots.as_slice();
        let mut i = start;
        loop {
            if read_slot(data, i).state == SLOT_EMPTY {
                return i;
            }
            i = (i + 1) % self.cap;
        }
    }

    fn needs_rehash(&self) -> bool {
        (self.next_id as usize) * 10 >= self.cap * 7
    }

    fn rehash(&mut self) {
        let new_cap = self.cap * 2;
        let mut new_slots = GrowableMmap::new_anon(new_cap * SLOT_SIZE).expect("rehash slot alloc failed");
        let old = self.slots.as_slice();
        for i in 0..self.cap {
            let s = read_slot(old, i);
            if s.state != SLOT_OCCUPIED {
                continue;
            }
            let mut j = (s.hash as usize) % new_cap;
            let nd = new_slots.as_mut_slice();
            while read_slot(nd, j).state != SLOT_EMPTY {
                j = (j + 1) % new_cap;
            }
            write_slot(new_slots.as_mut_slice(), j, &s);
        }
        self.slots = new_slots;
        self.cap = new_cap;
    }
}

enum Probe {
    Found(u64),
    Empty(usize),
}

// ── Marker I/O ──────────────────────────────────────────────────────────────────

/// Read `(next_id, keybytes_pos)` from the marker, or `None` if it is missing or
/// unreadable (treated as a fresh map).
fn read_marker(path: &Path) -> io::Result<Option<(u64, usize)>> {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    if bytes.len() < MARKER_SIZE || u64::from_le_bytes(bytes[0..8].try_into().unwrap()) != MARKER_MAGIC {
        return Ok(None);
    }
    let next_id = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    let keybytes_pos = u64::from_le_bytes(bytes[24..32].try_into().unwrap()) as usize;
    Ok(Some((next_id, keybytes_pos)))
}

fn write_marker(dir: &Path, next_id: u64, keybytes_pos: u64, wal_offset: u64) -> io::Result<()> {
    let mut buf = [0u8; MARKER_SIZE];
    buf[0..8].copy_from_slice(&MARKER_MAGIC.to_le_bytes());
    buf[8..12].copy_from_slice(&MARKER_VERSION.to_le_bytes());
    buf[16..24].copy_from_slice(&next_id.to_le_bytes());
    buf[24..32].copy_from_slice(&keybytes_pos.to_le_bytes());
    buf[32..40].copy_from_slice(&wal_offset.to_le_bytes());

    let marker = dir.join(MARKER_FILE);
    let tmp = dir.join("rowmap.ckpt.tmp");
    {
        use std::io::Write;
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(&buf)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, &marker)?;
    // fsync the directory so the rename is durable.
    std::fs::File::open(dir)?.sync_all()
}

// ── tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn alloc_is_dense_and_stable() {
        let dir = TempDir::new().unwrap();
        let mut rm = RowMap::create(dir.path()).unwrap();
        assert_eq!(rm.get_or_alloc(b"alpha"), 0);
        assert_eq!(rm.get_or_alloc(b"beta"), 1);
        assert_eq!(rm.get_or_alloc(b"gamma"), 2);
        // Re-resolving returns the same dense id, not a new one.
        assert_eq!(rm.get_or_alloc(b"beta"), 1);
        assert_eq!(rm.get_or_alloc(b"alpha"), 0);
        assert_eq!(rm.next_id(), 3);
    }

    #[test]
    fn get_only_does_not_allocate() {
        let mut rm = RowMap::create(TempDir::new().unwrap().path()).unwrap();
        rm.get_or_alloc(b"x");
        assert_eq!(rm.get(b"x"), Some(0));
        assert_eq!(rm.get(b"missing"), None);
        assert_eq!(rm.next_id(), 1, "get must not allocate");
    }

    #[test]
    fn key_for_round_trips() {
        let mut rm = RowMap::create(TempDir::new().unwrap().path()).unwrap();
        let id = rm.get_or_alloc(b"hello world");
        assert_eq!(rm.key_for(id).unwrap(), b"hello world");
        assert_eq!(rm.key_for(999), None);
    }

    #[test]
    fn survives_rehash() {
        let mut rm = RowMap::create(TempDir::new().unwrap().path()).unwrap();
        for i in 0u32..500 {
            assert_eq!(rm.get_or_alloc(format!("key-{i}").as_bytes()), i as u128);
        }
        // All still resolve to their original ids after several rehashes.
        for i in 0u32..500 {
            assert_eq!(rm.get(format!("key-{i}").as_bytes()), Some(i as u128));
            assert_eq!(rm.key_for(i as u128).unwrap(), format!("key-{i}").as_bytes());
        }
    }

    #[test]
    fn persists_across_reopen() {
        let dir = TempDir::new().unwrap();
        {
            let mut rm = RowMap::create(dir.path()).unwrap();
            for i in 0u32..300 {
                rm.get_or_alloc(format!("doc:{i}").as_bytes());
            }
            rm.flush(12345).unwrap();
        }
        let rm = RowMap::open(dir.path()).unwrap();
        assert_eq!(rm.next_id(), 300);
        for i in 0u32..300 {
            assert_eq!(rm.get(format!("doc:{i}").as_bytes()), Some(i as u128));
            assert_eq!(rm.key_for(i as u128).unwrap(), format!("doc:{i}").as_bytes());
        }
    }

    #[test]
    fn ignores_torn_tail_past_marker() {
        let dir = TempDir::new().unwrap();
        {
            let mut rm = RowMap::create(dir.path()).unwrap();
            rm.get_or_alloc(b"committed");
            rm.flush(1).unwrap();
            // Allocate more WITHOUT flushing — simulates a post-checkpoint tail
            // whose pages partially reached disk before a crash.
            rm.get_or_alloc(b"uncommitted-a");
            rm.get_or_alloc(b"uncommitted-b");
            rm.keybytes.flush().unwrap();
            rm.idarray.flush().unwrap();
        }
        let rm = RowMap::open(dir.path()).unwrap();
        assert_eq!(rm.next_id(), 1, "only the marker-committed id is recovered");
        assert_eq!(rm.get(b"committed"), Some(0));
        assert_eq!(rm.get(b"uncommitted-a"), None);
    }

    #[test]
    fn open_missing_marker_is_fresh() {
        let dir = TempDir::new().unwrap();
        let mut rm = RowMap::open(dir.path()).unwrap();
        assert!(rm.is_empty());
        assert_eq!(rm.get_or_alloc(b"first"), 0);
    }
}
