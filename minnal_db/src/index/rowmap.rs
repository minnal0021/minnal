//! Per-namespace dense row-ID map.
//!
//! Assigns **dense, monotonic** row IDs (0, 1, 2, …) to keys so the field-index
//! [`RoaringBitmap`](crate::index::RoaringBitmap)s pack densely — consecutive IDs share
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
//!   slot table is file-backed (`rows.slots`) but **rebuilt from the id array on
//!   every open**, so it is never a persisted source of truth — see
//!   *The slot table is write-only* below.
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
//! ### The slot table is write-only (do not "optimise" the rebuild away)
//!
//! `rows.slots` is mapped from a file purely to keep it off the heap: a slot is
//! 24 bytes held under a 0.7 load factor, so ~34–69 bytes are resident per
//! distinct key ever written — 384 MiB at 10M keys. Anonymous pages can be
//! swapped but never *dropped*, so that was unreclaimable while the store was
//! open; file-backed pages the kernel can evict and re-read.
//!
//! Nothing ever reads those bytes back across a restart. [`open`] recreates the
//! file with `.truncate(true)` and rebuilds from `idarray[0..next_id]`, and that
//! is **required, not laziness**. A `MAP_SHARED` mapping is written back by the
//! kernel whenever it likes, with no inter-page ordering, so after a crash the
//! file can hold entries for ids at or past the marker's `next_id`, or a
//! half-finished [`rehash`](RowMap::rehash). Unlike `keybytes`/`idarray` — which
//! are append-only, so the marker's length is a clean cut and anything past it
//! is an ignorable torn tail — scattered hash positions give no such cut, and a
//! stale entry is indistinguishable from a good one.
//!
//! Adopting one would be silent, permanent corruption rather than a slow path:
//! a surviving `"user:alice" → 7` when the committed id array reaches only 5
//! makes `get_or_alloc` hand back 7 without allocating, so the field index sets
//! bit 7 while `key_for(7)` resolves to nothing — and the next genuine
//! allocation reuses that id for a different key. **WAL replay cannot heal this:
//! replay adds entries, it never removes them.** Note this is *not* fixed by
//! guaranteeing the WAL tail survives (FR-001): that solves a missing tail, and
//! this is unproven content.
//!
//! Trusting the file at open is possible, but it needs a real commit protocol —
//! an in-file `CLEAN`/`DIRTY` header msynced *before* the first slot mutation,
//! plus `header.next_id == marker.next_id`. It is deliberately not built,
//! because after a crash the table is `DIRTY` by construction and rebuilds
//! anyway: it would only ever speed up a clean restart, which is the one nobody
//! is waiting on.
//!
//! [`flush`]: RowMap::flush
//! [`open`]: RowMap::open

use std::io;
use std::path::{Path, PathBuf};

use crate::index::blob_store::GrowableMmap;

// ── Layout constants ──────────────────────────────────────────────────────────

const KEYBYTES_FILE: &str = "rows.keybytes";
const IDARRAY_FILE: &str = "rows.idarray";
const SLOTS_FILE: &str = "rows.slots";
/// Scratch file `rehash` builds the doubled table in before renaming it over
/// [`SLOTS_FILE`]. A leftover copy is harmless — the next open truncates it.
const SLOTS_TMP_FILE: &str = "rows.slots.tmp";
const MARKER_FILE: &str = "rowmap.ckpt";

const MARKER_MAGIC: u64 = 0x4D494E4E414C524D; // "MINNALRM"
const MARKER_VERSION: u32 = 1;
const MARKER_SIZE: usize = 40;

/// `[key_off: u64 LE | key_len: u32 LE]` per ID in `rows.idarray`.
const ID_ENTRY_SIZE: usize = 12;

const SLOT_SIZE: usize = 24;

const INITIAL_SLOT_CAP: usize = 16;
const INITIAL_KEYBYTES: usize = 4096;
const INITIAL_IDARRAY: usize = INITIAL_SLOT_CAP * ID_ENTRY_SIZE;

// ── Slot (file-backed hash table; see the module docs) ─────────────────────────
//
// Byte layout (24 bytes), packed — every field is read with `from_le_bytes` on a
// byte slice, so nothing here needs alignment padding:
//
//   0..8    id_plus_one: u64 LE  (0 ⇒ EMPTY; an occupied slot stores id + 1)
//   8..16   key_off:     u64 LE  (offset into the key-bytes region)
//   16..20  key_len:     u32 LE
//   20..24  hash:        u32 LE  (low 32 bits of the key's FNV-1a hash)
//
// **A zero-filled slot must read as EMPTY**, which is what lets `create_file`'s
// truncate produce a valid empty table for free. Row IDs are dense from 0, so a
// raw `id` cannot be the sentinel — hence the `+ 1` encoding.
//
// The previous layout was 40 bytes, 11 of them padding that bought nothing.

#[derive(Clone, Copy)]
struct Slot {
    hash: u32,
    key_off: u64,
    key_len: u32,
    id: u64,
}

/// Read slot `i`, or `None` when it is empty.
fn read_slot(data: &[u8], i: usize) -> Option<Slot> {
    let b = i * SLOT_SIZE;
    let id_plus_one = u64::from_le_bytes(data[b..b + 8].try_into().unwrap());
    if id_plus_one == 0 {
        return None;
    }
    Some(Slot {
        id: id_plus_one - 1,
        key_off: u64::from_le_bytes(data[b + 8..b + 16].try_into().unwrap()),
        key_len: u32::from_le_bytes(data[b + 16..b + 20].try_into().unwrap()),
        hash: u32::from_le_bytes(data[b + 20..b + 24].try_into().unwrap()),
    })
}

/// Write an **occupied** slot. There is no way to write an empty one because
/// entries are never removed; emptiness comes only from zeroed bytes.
fn write_slot(data: &mut [u8], i: usize, s: &Slot) {
    let b = i * SLOT_SIZE;
    data[b..b + 8].copy_from_slice(&(s.id + 1).to_le_bytes());
    data[b + 8..b + 16].copy_from_slice(&s.key_off.to_le_bytes());
    data[b + 16..b + 20].copy_from_slice(&s.key_len.to_le_bytes());
    data[b + 20..b + 24].copy_from_slice(&s.hash.to_le_bytes());
}

/// FNV-1a over the raw key bytes, truncated to 32 bits.
///
/// Two independent uses, and the truncation is safe for both:
///
/// - **Position** (`hash % cap`). `cap` is a power of two, so this only ever
///   consumed the low `log2(cap)` bits — the high half was never read. Keeping
///   32 bits is therefore identical placement, not an approximation, as long as
///   `cap <= 2^32` (asserted in `rehash`; 2^32 slots would be a 96 GiB table).
/// - **Fast reject** before comparing key bytes. A 32-bit tag collides one time
///   in ~4 billion *within a probe chain*, and a collision costs one key compare
///   that then rejects — the key comparison is what decides, so this cannot
///   return a wrong id.
fn hash_bytes(key: &[u8]) -> u32 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in key {
        h ^= b as u64;
        h = h.wrapping_mul(0x00000100000001b3);
    }
    h as u32
}

// ── RowMap ──────────────────────────────────────────────────────────────────────

/// Per-namespace dense row-ID map. See the module docs.
pub struct RowMap {
    dir: PathBuf,
    /// Append-only raw key bytes.
    keybytes: GrowableMmap,
    /// Append-only `id → (key_off, key_len)` array, indexed by ID.
    idarray: GrowableMmap,
    /// Open-addressing `key → id` table, file-backed by `rows.slots` and
    /// **rebuilt from `idarray` on every open** — see the durability note in the
    /// type docs for why it is never adopted from disk.
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
        let slots = GrowableMmap::create_file(&dir.join(SLOTS_FILE), INITIAL_SLOT_CAP * SLOT_SIZE)?;
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

        // Validate the marker against the backing files before trusting it: the
        // id array must hold `next_id` entries, and `keybytes_pos` must lie within
        // the key-bytes file. Otherwise the rebuild below would slice out of
        // bounds (panic) on a corrupt/mismatched sidecar.
        let need_idarray = next_id
            .checked_mul(ID_ENTRY_SIZE as u64)
            .ok_or_else(|| invalid("next_id overflows id-array size"))?;
        if need_idarray > idarray.as_slice().len() as u64 {
            return Err(invalid(format!(
                "id array {} bytes too small for next_id {next_id} (need {need_idarray})",
                idarray.as_slice().len()
            )));
        }
        if keybytes_pos > keybytes.as_slice().len() {
            return Err(invalid(format!(
                "keybytes_pos {keybytes_pos} beyond key-bytes file ({} bytes)",
                keybytes.as_slice().len()
            )));
        }

        // Size the slot table for next_id entries at < 0.7 load, power-of-two.
        let mut cap = INITIAL_SLOT_CAP;
        while (next_id as usize) * 10 >= cap * 7 {
            cap *= 2;
        }
        // `create_file` opens with `.truncate(true)`, so this is a zeroed table of
        // exactly the right size and the rebuild below runs unchanged. The
        // truncate is load-bearing, not inherited habit: it is what keeps the
        // file write-only with respect to correctness, so no stale entry from a
        // previous run can survive to be read back. See the type docs.
        let slots = GrowableMmap::create_file(&dir.join(SLOTS_FILE), cap * SLOT_SIZE)?;

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
            // Each entry's key bytes must lie within the committed key region.
            let end = off
                .checked_add(len as u64)
                .ok_or_else(|| invalid("id-array entry offset+len overflows"))?;
            if end > keybytes_pos as u64 {
                return Err(invalid(format!(
                    "id {id} key bytes [{off}, {end}) extend past keybytes_pos {keybytes_pos}"
                )));
            }
            let key = me.keybytes.as_slice()[off as usize..off as usize + len as usize].to_vec();
            let hash = hash_bytes(&key);
            let idx = me.find_empty(&key, hash);
            write_slot(
                me.slots.as_mut_slice(),
                idx,
                &Slot {
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
    ///
    /// IDs are `u64` internally (dense, monotonic from 0 — `next_id` cannot
    /// realistically reach `u64::MAX`) but surfaced as `u128` for API uniformity
    /// with the rest of the index. A `u128` beyond `u64::MAX` therefore was never
    /// allocated here: `try_from` yields `None` rather than truncating the id.
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

    /// Claim `slot_idx` for `key`, appending its bytes and id-array entry.
    ///
    /// `slot_idx` must be an **empty** slot. `probe` returns `Empty(start)` when
    /// it wraps a completely full table, and `start` is occupied in that case —
    /// writing there would drop a live `key → id` binding on the floor and hand
    /// the same id to two keys. The load factor makes a full table unreachable
    /// (`get_or_alloc` rehashes first), so this is a cheap assertion against a
    /// silent-corruption path, not an expected condition.
    fn insert_at(&mut self, slot_idx: usize, key: &[u8], hash: u32) -> u128 {
        debug_assert!(
            read_slot(self.slots.as_slice(), slot_idx).is_none(),
            "insert would overwrite an occupied slot ({slot_idx})"
        );
        let id = self.next_id;
        let key_off = self.keybytes_pos as u64;
        let key_len = crate::index::blob_store::u32_len(key.len(), "row key");

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

        write_slot(self.slots.as_mut_slice(), slot_idx, &Slot { hash, key_off, key_len, id });
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

    fn probe(&self, key: &[u8], hash: u32) -> Probe {
        let start = (hash as usize) % self.cap;
        let data = self.slots.as_slice();
        let mut i = start;
        loop {
            match read_slot(data, i) {
                None => return Probe::Empty(i),
                Some(s) if s.hash == hash && self.key_eq(&s, key) => return Probe::Found(s.id),
                Some(_) => {}
            }
            i = (i + 1) % self.cap;
            if i == start {
                return Probe::Empty(start);
            }
        }
    }

    /// Find the slot a new `key` would occupy (no duplicate-key check — callers
    /// only use this during a from-scratch rebuild where keys are unique).
    /// Find the slot an insert should land in.
    ///
    /// Bounded by `cap`: the 0.7 load factor means a correct table always has
    /// empty slots, so exhausting the probe is impossible in normal operation —
    /// but an unbounded loop turns any violation of that invariant into a
    /// **silent hang** with no diagnostics, which is the worst way for a storage
    /// engine to fail. Panicking says what went wrong. (Reachable, for example,
    /// if the slot file were ever adopted from disk instead of rebuilt, leaving
    /// stale entries on top of the rebuilt ones — see the module docs.)
    fn find_empty(&self, _key: &[u8], hash: u32) -> usize {
        let start = (hash as usize) % self.cap;
        let data = self.slots.as_slice();
        let mut i = start;
        for _ in 0..self.cap {
            if read_slot(data, i).is_none() {
                return i;
            }
            i = (i + 1) % self.cap;
        }
        panic!(
            "row map slot table is full (cap {}, next_id {}) — the load-factor invariant was violated",
            self.cap, self.next_id
        );
    }

    fn needs_rehash(&self) -> bool {
        (self.next_id as usize) * 10 >= self.cap * 7
    }

    /// Double the table and re-insert every occupied slot.
    ///
    /// Builds into a **scratch file** and renames it over `rows.slots`, rather
    /// than into anonymous memory. Both tables are live at once here, so this is
    /// the moment of peak footprint (~1.5× the final size); allocating the new
    /// one anonymously would reintroduce exactly the unreclaimable RSS this file
    /// backing exists to remove, at the worst possible time.
    ///
    /// The rename replaces the directory entry while our mapping keeps the old
    /// inode alive, so no remap is needed — assigning `self.slots` drops the old
    /// map and frees it. Nothing is fsynced: the table is rebuilt from `idarray`
    /// at every open, so a torn or leftover file costs nothing.
    fn rehash(&mut self) {
        let new_cap = self.cap * 2;
        let tmp_path = self.dir.join(SLOTS_TMP_FILE);
        let mut new_slots = GrowableMmap::create_file(&tmp_path, new_cap * SLOT_SIZE).expect("rehash slot file alloc failed");
        // The stored 32-bit hash must place a key exactly where a freshly
        // computed one would; that holds only while `cap` fits in 32 bits.
        debug_assert!(new_cap <= u32::MAX as usize, "slot capacity outgrew the 32-bit stored hash");
        let old = self.slots.as_slice();
        for i in 0..self.cap {
            let Some(s) = read_slot(old, i) else { continue };
            let mut j = (s.hash as usize) % new_cap;
            let nd = new_slots.as_mut_slice();
            while read_slot(nd, j).is_some() {
                j = (j + 1) % new_cap;
            }
            write_slot(new_slots.as_mut_slice(), j, &s);
        }
        // Publish the new table, then drop the old mapping (which frees the
        // now-unlinked inode). Order matters only for disk-space transients.
        std::fs::rename(&tmp_path, self.dir.join(SLOTS_FILE)).expect("rehash slot file rename failed");
        self.slots = new_slots;
        self.cap = new_cap;
    }
}

enum Probe {
    Found(u64),
    Empty(usize),
}

// ── Marker I/O ──────────────────────────────────────────────────────────────────

fn invalid(msg: impl std::fmt::Display) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("row map: {msg}"))
}

/// Read `(next_id, keybytes_pos)` from the marker.
///
/// Returns `None` only when the marker file is **absent** (a never-checkpointed
/// map → start fresh). A marker that is present but corrupt — truncated, wrong
/// magic, or an unsupported version — is rejected with
/// [`io::ErrorKind::InvalidData`] rather than silently treated as fresh: a fresh
/// start would reset `next_id` to 0 and reissue dense IDs that existing field-
/// index bitmaps already use, corrupting the index. The marker is written via
/// tmp+rename+fsync, so a present marker is always whole — a malformed one means
/// real corruption.
fn read_marker(path: &Path) -> io::Result<Option<(u64, usize)>> {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    if bytes.len() < MARKER_SIZE {
        return Err(invalid(format!("marker truncated ({} bytes, need {MARKER_SIZE})", bytes.len())));
    }
    let magic = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
    if magic != MARKER_MAGIC {
        return Err(invalid(format!("bad marker magic {magic:#018x}")));
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    if version != MARKER_VERSION {
        return Err(invalid(format!("unsupported marker version {version} (expected {MARKER_VERSION})")));
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

    /// Measurement, not a correctness check — run explicitly:
    /// `cargo test -p minnal_db --lib slot_table_footprint -- --ignored --nocapture`
    ///
    /// Reports anonymous RSS against the slot file's size, which is the whole
    /// point of phase 1: the same bytes, moved somewhere the kernel can reclaim.
    #[test]
    #[ignore = "measurement; run explicitly with --ignored --nocapture"]
    fn slot_table_footprint() {
        fn rss_anon_kb() -> u64 {
            // RssAnon is the unreclaimable part — file-backed pages are RssFile.
            std::fs::read_to_string("/proc/self/status")
                .unwrap_or_default()
                .lines()
                .find_map(|l| {
                    l.strip_prefix("RssAnon:")
                        .map(|v| v.trim().trim_end_matches(" kB").trim().parse::<u64>().unwrap_or(0))
                })
                .unwrap_or(0)
        }

        const N: u32 = 1_000_000;
        let dir = TempDir::new().unwrap();
        let before = rss_anon_kb();
        let mut rm = RowMap::create(dir.path()).unwrap();
        for i in 0..N {
            rm.get_or_alloc(format!("key-{i:08}").as_bytes());
        }
        let after = rss_anon_kb();
        let slot_bytes = std::fs::metadata(dir.path().join(SLOTS_FILE)).unwrap().len();

        println!(
            "keys={N} cap={} slot_file={} MiB anon_rss_delta={} MiB",
            rm.cap,
            slot_bytes / (1024 * 1024),
            (after - before) / 1024
        );
        assert_eq!(slot_bytes, (rm.cap * SLOT_SIZE) as u64);
    }

    /// Measurement, not a correctness check — run explicitly (use `--release`,
    /// a debug build is not representative):
    /// `cargo test -p minnal_db --release --lib rebuild_cost -- --ignored --nocapture`
    ///
    /// Times the `open` rebuild, which is the price phase 1 keeps paying to
    /// avoid ever adopting slot-file contents. `MINNAL_ROWMAP_KEYS` overrides
    /// the key count.
    #[test]
    #[ignore = "measurement; run explicitly with --release --ignored --nocapture"]
    fn rebuild_cost() {
        let n: u32 = std::env::var("MINNAL_ROWMAP_KEYS").ok().and_then(|v| v.parse().ok()).unwrap_or(1_000_000);
        let dir = TempDir::new().unwrap();

        let t0 = std::time::Instant::now();
        {
            let mut rm = RowMap::create(dir.path()).unwrap();
            for i in 0..n {
                rm.get_or_alloc(format!("key-{i:010}").as_bytes());
            }
            rm.flush(1).unwrap();
        }
        let populate = t0.elapsed();

        // Re-open several times so page-cache-warm cost is visible separately
        // from the first (cold-ish) open.
        let mut opens = Vec::new();
        for _ in 0..3 {
            let t = std::time::Instant::now();
            let rm = RowMap::open(dir.path()).unwrap();
            opens.push(t.elapsed());
            assert_eq!(rm.next_id(), n as u64);
        }

        let slot_mib = std::fs::metadata(dir.path().join(SLOTS_FILE)).unwrap().len() / (1024 * 1024);
        println!(
            "keys={n} slot_file={slot_mib} MiB  populate={:?}  open_rebuild={:?} {:?} {:?}  (~{:.0} ns/key)",
            populate,
            opens[0],
            opens[1],
            opens[2],
            opens[2].as_nanos() as f64 / n as f64
        );
    }

    /// Phase 1: the slot table is backed by a real file, not anonymous memory.
    /// That is the whole point — anonymous pages can be swapped but never
    /// dropped, so they are unreclaimable RSS for the life of the store.
    #[test]
    fn slot_table_is_file_backed_and_sized_to_capacity() {
        let dir = TempDir::new().unwrap();
        let slots = dir.path().join(SLOTS_FILE);

        let mut rm = RowMap::create(dir.path()).unwrap();
        assert!(slots.exists(), "the slot table must be backed by a file");
        assert_eq!(
            std::fs::metadata(&slots).unwrap().len(),
            (INITIAL_SLOT_CAP * SLOT_SIZE) as u64,
            "a fresh table is sized to the initial capacity"
        );

        // Past the 0.7 load factor the table rehashes; the file must grow with
        // it rather than the doubled copy landing back in anonymous memory.
        for i in 0u32..500 {
            rm.get_or_alloc(format!("key-{i}").as_bytes());
        }
        assert_eq!(
            std::fs::metadata(&slots).unwrap().len(),
            (rm.cap * SLOT_SIZE) as u64,
            "after rehashing, the file still backs the whole table"
        );
        assert!(
            !dir.path().join(SLOTS_TMP_FILE).exists(),
            "the rehash scratch file must be renamed into place, not left behind"
        );
    }

    /// Phase 1's load-bearing property: `open` **never adopts** slot-file
    /// contents. The file is truncated and rebuilt from the durable id array, so
    /// entries the marker never committed cannot survive to be read back.
    ///
    /// Without this, a stale `key → id` for an uncommitted id would make
    /// `get_or_alloc` hand back an id the id array does not have — silent,
    /// permanent bitmap/key divergence that WAL replay cannot heal, because
    /// replay adds entries and never removes them.
    #[test]
    fn open_rebuilds_the_slot_table_and_never_adopts_stale_entries() {
        let dir = TempDir::new().unwrap();
        {
            let mut rm = RowMap::create(dir.path()).unwrap();
            for i in 0u32..100 {
                rm.get_or_alloc(format!("doc:{i}").as_bytes());
            }
            rm.flush(1).unwrap();
            // Allocate past the committed marker, exactly as a run does between
            // checkpoints, then "crash" without flushing.
            for i in 100u32..150 {
                rm.get_or_alloc(format!("doc:{i}").as_bytes());
            }
        }

        // Find a slot the rebuild provably does not write, so the poison below
        // is a deterministic negative control rather than a coin flip: if `open`
        // ever stopped truncating, this entry would certainly survive.
        let victim = {
            let _ = RowMap::open(dir.path()).unwrap();
            let data = std::fs::read(dir.path().join(SLOTS_FILE)).unwrap();
            (0..data.len() / SLOT_SIZE)
                .find(|&i| read_slot(&data, i).is_none())
                .expect("a table under 0.7 load must have empty slots")
        };

        // Simulate the kernel having written back a slot for an uncommitted id —
        // which it may do at any time, in any order, for a MAP_SHARED mapping.
        {
            use std::os::unix::fs::FileExt;
            let f = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(dir.path().join(SLOTS_FILE))
                .unwrap();
            let mut poisoned = [0u8; SLOT_SIZE];
            write_slot(
                &mut poisoned,
                0,
                &Slot {
                    hash: hash_bytes(b"doc:142"),
                    key_off: 0,
                    key_len: 7,
                    id: 142,
                },
            );
            f.write_all_at(&poisoned, (victim * SLOT_SIZE) as u64).unwrap();
            f.sync_all().unwrap();
        }

        let rm = RowMap::open(dir.path()).unwrap();
        assert_eq!(rm.next_id(), 100, "only the committed ids survive");
        assert_eq!(
            rm.get(b"doc:142"),
            None,
            "an uncommitted entry must NOT be adopted — it would hand back an id the id array lacks"
        );
        for i in 0u32..100 {
            assert_eq!(
                rm.get(format!("doc:{i}").as_bytes()),
                Some(i as u128),
                "every committed key is rebuilt from the id array"
            );
        }

        // The strongest form of the property: the only occupied slots on disk are
        // the ones the rebuild just wrote. Anything else is residue that a future
        // lookup could reach.
        let data = std::fs::read(dir.path().join(SLOTS_FILE)).unwrap();
        let occupied = (0..data.len() / SLOT_SIZE).filter(|&i| read_slot(&data, i).is_some()).count();
        assert_eq!(occupied, 100, "the slot file must hold exactly the rebuilt entries, no residue");
    }

    #[test]
    fn get_only_does_not_allocate() {
        // Bind the TempDir: passing `TempDir::new()?.path()` directly drops the
        // directory at the end of the statement, leaving the map pointed at a
        // path that no longer exists.
        let dir = TempDir::new().unwrap();
        let mut rm = RowMap::create(dir.path()).unwrap();
        rm.get_or_alloc(b"x");
        assert_eq!(rm.get(b"x"), Some(0));
        assert_eq!(rm.get(b"missing"), None);
        assert_eq!(rm.next_id(), 1, "get must not allocate");
    }

    #[test]
    fn key_for_round_trips() {
        // Bind the TempDir: passing `TempDir::new()?.path()` directly drops the
        // directory at the end of the statement, leaving the map pointed at a
        // path that no longer exists.
        let dir = TempDir::new().unwrap();
        let mut rm = RowMap::create(dir.path()).unwrap();
        let id = rm.get_or_alloc(b"hello world");
        assert_eq!(rm.key_for(id).unwrap(), b"hello world");
        assert_eq!(rm.key_for(999), None);
    }

    #[test]
    fn survives_rehash() {
        // Bind the TempDir: passing `TempDir::new()?.path()` directly drops the
        // directory at the end of the statement, leaving the map pointed at a
        // path that no longer exists.
        let dir = TempDir::new().unwrap();
        let mut rm = RowMap::create(dir.path()).unwrap();
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

    // ── Marker / bounds validation on open ──────────────────────────────────
    //
    // A present-but-corrupt marker must be rejected with InvalidData rather than
    // silently treated as fresh (which would reset next_id and reissue IDs that
    // existing bitmaps already use) or panic with an out-of-bounds slice.

    fn seed(dir: &Path) {
        let mut rm = RowMap::create(dir).unwrap();
        rm.get_or_alloc(b"alpha");
        rm.get_or_alloc(b"beta");
        rm.flush(0).unwrap();
    }

    fn corrupt_marker(dir: &Path, f: impl FnOnce(&mut [u8])) {
        let p = dir.join(MARKER_FILE);
        let mut b = std::fs::read(&p).unwrap();
        f(&mut b);
        std::fs::write(&p, &b).unwrap();
    }

    fn assert_invalid(dir: &Path) {
        match RowMap::open(dir) {
            Err(e) => assert_eq!(e.kind(), io::ErrorKind::InvalidData, "expected InvalidData, got: {e}"),
            Ok(_) => panic!("expected open() to reject corrupt marker"),
        }
    }

    #[test]
    fn open_rejects_bad_marker_magic() {
        let dir = TempDir::new().unwrap();
        seed(dir.path());
        corrupt_marker(dir.path(), |b| b[0] ^= 0xFF);
        assert_invalid(dir.path());
    }

    #[test]
    fn open_rejects_unsupported_marker_version() {
        let dir = TempDir::new().unwrap();
        seed(dir.path());
        corrupt_marker(dir.path(), |b| b[8..12].copy_from_slice(&(MARKER_VERSION + 1).to_le_bytes()));
        assert_invalid(dir.path());
    }

    #[test]
    fn open_rejects_truncated_marker() {
        let dir = TempDir::new().unwrap();
        seed(dir.path());
        std::fs::write(dir.path().join(MARKER_FILE), [0u8; MARKER_SIZE - 1]).unwrap();
        assert_invalid(dir.path());
    }

    #[test]
    fn open_rejects_next_id_larger_than_id_array() {
        let dir = TempDir::new().unwrap();
        seed(dir.path());
        // next_id far beyond what the id-array file can hold.
        corrupt_marker(dir.path(), |b| b[16..24].copy_from_slice(&(1u64 << 40).to_le_bytes()));
        assert_invalid(dir.path());
    }

    #[test]
    fn open_rejects_keybytes_pos_past_file() {
        let dir = TempDir::new().unwrap();
        seed(dir.path());
        corrupt_marker(dir.path(), |b| b[24..32].copy_from_slice(&(1u64 << 40).to_le_bytes()));
        assert_invalid(dir.path());
    }

    #[test]
    fn open_rejects_id_entry_past_keybytes_pos() {
        let dir = TempDir::new().unwrap();
        seed(dir.path());
        // Shrink keybytes_pos so the first id's key bytes fall outside it.
        corrupt_marker(dir.path(), |b| b[24..32].copy_from_slice(&1u64.to_le_bytes()));
        assert_invalid(dir.path());
    }
}
