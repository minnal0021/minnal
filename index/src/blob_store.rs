//! Memory-mapped two-file blob store.
//!
//! Stores a `u128 → raw-bytes` map in two memory-mapped regions, using the
//! same on-disk layout as [`ContainerStore`] but with arbitrary byte blobs as
//! values instead of rkyv-serialised [`Container`] objects.
//!
//! Used by [`FieldIndex`] to hold per-distinct-value [`RoaringBitmap`] data
//! off-heap so the heap only carries the small `BTreeMap<V, u128>` ordering
//! index.
//!
//! **Key file** (`blobs.keys`):
//! - 64-byte header (magic, version, capacity, counts, value write pos)
//! - Fixed-size 48-byte slots forming an open-addressing linear-probing hash table
//!
//! **Value file** (`blobs.vals`):
//! - Append-only sequence of raw byte blobs
//! - Each slot records the byte offset and length of its blob

use std::fs::{File, OpenOptions};
use std::io;
use std::io::Write;
use std::path::{Path, PathBuf};

use memmap2::MmapMut;

// ── Layout constants ──────────────────────────────────────────────────────────

const MAGIC: u64 = 0x4D494E4E414C4253; // "MINNALBS"
const VERSION: u32 = 1;
const HEADER_SIZE: usize = 64;
const SLOT_SIZE: usize = 48;
const INITIAL_CAPACITY: usize = 16;
const INITIAL_KEY_SIZE: usize = HEADER_SIZE + INITIAL_CAPACITY * SLOT_SIZE;
const INITIAL_VAL_SIZE: usize = 4096;
const VALUE_ALIGNMENT: usize = 16;

const STATE_EMPTY: u8 = 0;
const STATE_OCCUPIED: u8 = 1;
const STATE_TOMBSTONE: u8 = 2;

// ── Header ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
struct Header {
    capacity: u64,
    count: u64,
    tombstone_count: u64,
    value_write_pos: u64,
}

fn read_header(data: &[u8]) -> Header {
    Header {
        capacity: u64::from_le_bytes(data[16..24].try_into().unwrap()),
        count: u64::from_le_bytes(data[24..32].try_into().unwrap()),
        tombstone_count: u64::from_le_bytes(data[32..40].try_into().unwrap()),
        value_write_pos: u64::from_le_bytes(data[48..56].try_into().unwrap()),
    }
}

fn write_header(data: &mut [u8], h: &Header) {
    data[0..8].copy_from_slice(&MAGIC.to_le_bytes());
    data[8..12].copy_from_slice(&VERSION.to_le_bytes());
    data[12..16].fill(0);
    data[16..24].copy_from_slice(&h.capacity.to_le_bytes());
    data[24..32].copy_from_slice(&h.count.to_le_bytes());
    data[32..40].copy_from_slice(&h.tombstone_count.to_le_bytes());
    data[40..48].fill(0); // reserved
    data[48..56].copy_from_slice(&h.value_write_pos.to_le_bytes());
    data[56..64].fill(0);
}

fn init_header(data: &mut [u8], capacity: usize) {
    write_header(
        data,
        &Header {
            capacity: capacity as u64,
            count: 0,
            tombstone_count: 0,
            value_write_pos: 0,
        },
    );
}

// ── Slot ──────────────────────────────────────────────────────────────────────
//
// Byte layout (48 bytes):
//   0       state: u8
//   1..8    pad
//   8..24   key: u128  (LE)
//   24..32  offset: u64 (LE)
//   32..36  len: u32   (LE)
//   36..48  pad

#[derive(Debug, Clone, Copy)]
struct Slot {
    state: u8,
    key: u128,
    offset: u64,
    len: u32,
}

fn slot_byte_offset(i: usize) -> usize {
    HEADER_SIZE + i * SLOT_SIZE
}

fn read_slot(data: &[u8], i: usize) -> Slot {
    let b = slot_byte_offset(i);
    Slot {
        state: data[b],
        key: u128::from_le_bytes(data[b + 8..b + 24].try_into().unwrap()),
        offset: u64::from_le_bytes(data[b + 24..b + 32].try_into().unwrap()),
        len: u32::from_le_bytes(data[b + 32..b + 36].try_into().unwrap()),
    }
}

fn write_slot(data: &mut [u8], i: usize, s: &Slot) {
    let b = slot_byte_offset(i);
    data[b] = s.state;
    data[b + 1..b + 8].fill(0);
    data[b + 8..b + 24].copy_from_slice(&s.key.to_le_bytes());
    data[b + 24..b + 32].copy_from_slice(&s.offset.to_le_bytes());
    data[b + 32..b + 36].copy_from_slice(&s.len.to_le_bytes());
    data[b + 36..b + 48].fill(0);
}

// ── Open-time validation ───────────────────────────────────────────────────────

fn invalid(msg: impl std::fmt::Display) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("blob store: {msg}"))
}

/// Validate a just-opened store's on-disk images before any read trusts them.
///
/// Without this, a corrupt or wrong file silently feeds garbage into the hot
/// path: a zero `capacity` divides by zero in the hash modulo, a `capacity`
/// larger than the key file slices out of bounds, and a slot whose `offset+len`
/// runs past the value region reads arbitrary bytes. Every failure here is a
/// controlled [`io::ErrorKind::InvalidData`], never a panic.
fn validate_open(key: &[u8], val_len: usize) -> io::Result<()> {
    if key.len() < HEADER_SIZE {
        return Err(invalid("key file smaller than header"));
    }
    let magic = u64::from_le_bytes(key[0..8].try_into().unwrap());
    if magic != MAGIC {
        return Err(invalid(format!("bad magic {magic:#018x} (not a minnal blob store)")));
    }
    let version = u32::from_le_bytes(key[8..12].try_into().unwrap());
    if version != VERSION {
        return Err(invalid(format!("unsupported on-disk version {version} (expected {VERSION})")));
    }
    let hdr = read_header(key);
    if hdr.capacity == 0 || !hdr.capacity.is_power_of_two() {
        return Err(invalid(format!("capacity {} is not a non-zero power of two", hdr.capacity)));
    }
    // The key file must hold the full header + slot array for `capacity`.
    let need = hdr
        .capacity
        .checked_mul(SLOT_SIZE as u64)
        .and_then(|s| s.checked_add(HEADER_SIZE as u64))
        .ok_or_else(|| invalid("capacity overflows key-file size"))?;
    if (key.len() as u64) < need {
        return Err(invalid(format!(
            "key file {} bytes too small for capacity {} (need {need})",
            key.len(),
            hdr.capacity
        )));
    }
    // The write cursor and every live blob must lie within the value file.
    if hdr.value_write_pos > val_len as u64 {
        return Err(invalid(format!(
            "value_write_pos {} beyond value file ({val_len} bytes)",
            hdr.value_write_pos
        )));
    }
    for i in 0..hdr.capacity as usize {
        let s = read_slot(key, i);
        if s.state == STATE_OCCUPIED {
            let end = s.offset.checked_add(s.len as u64).ok_or_else(|| invalid("slot offset+len overflows"))?;
            if end > hdr.value_write_pos {
                return Err(invalid(format!(
                    "slot {i} blob [{}, {end}) extends past value_write_pos {}",
                    s.offset, hdr.value_write_pos
                )));
            }
        }
    }
    Ok(())
}

// ── Hash ──────────────────────────────────────────────────────────────────────

fn hash_key(key: u128) -> usize {
    let lo = key as u64;
    let hi = (key >> 64) as u64;
    lo.wrapping_add(hi.rotate_left(17))
        .wrapping_mul(0x9e3779b97f4a7c15)
        .wrapping_add(hi ^ lo.rotate_right(13)) as usize
}

// ── GrowableMmap ──────────────────────────────────────────────────────────────

pub(crate) struct GrowableMmap {
    mmap: MmapMut,
    file: Option<File>,
}

impl GrowableMmap {
    pub(crate) fn new_anon(initial_size: usize) -> io::Result<Self> {
        let mmap = MmapMut::map_anon(initial_size)?;
        Ok(Self { mmap, file: None })
    }

    pub(crate) fn create_file(path: &Path, initial_size: usize) -> io::Result<Self> {
        let file = OpenOptions::new().read(true).write(true).create(true).truncate(true).open(path)?;
        file.set_len(initial_size as u64)?;
        // SAFETY: file is open for read+write, no other mmap on this file.
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        Ok(Self { mmap, file: Some(file) })
    }

    pub(crate) fn open_file(path: &Path) -> io::Result<Self> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let size = file.metadata()?.len() as usize;
        let mmap = if size > 0 {
            // SAFETY: file is open for read+write.
            unsafe { MmapMut::map_mut(&file)? }
        } else {
            MmapMut::map_anon(INITIAL_KEY_SIZE)?
        };
        Ok(Self { mmap, file: Some(file) })
    }

    pub(crate) fn ensure_capacity(&mut self, needed: usize) -> io::Result<()> {
        if needed <= self.mmap.len() {
            return Ok(());
        }
        let new_size = self.mmap.len().max(needed).next_power_of_two().max(needed);
        match &self.file {
            None => {
                let mut new_mmap = MmapMut::map_anon(new_size)?;
                new_mmap[..self.mmap.len()].copy_from_slice(&self.mmap);
                self.mmap = new_mmap;
            }
            Some(file) => {
                file.set_len(new_size as u64)?;
                // SAFETY: file is open for read+write.
                self.mmap = unsafe { MmapMut::map_mut(file)? };
            }
        }
        Ok(())
    }

    /// Shrink (or grow) the backing store to exactly `new_size` bytes,
    /// reclaiming disk for the file-backed case. Floored at 1 byte so the mmap
    /// stays valid. Contents in `0..min(new_size, old_size)` are preserved.
    fn truncate(&mut self, new_size: usize) -> io::Result<()> {
        let new_size = new_size.max(1);
        if new_size == self.mmap.len() {
            return Ok(());
        }
        match &self.file {
            None => {
                let mut new_mmap = MmapMut::map_anon(new_size)?;
                let n = new_size.min(self.mmap.len());
                new_mmap[..n].copy_from_slice(&self.mmap[..n]);
                self.mmap = new_mmap;
            }
            Some(file) => {
                self.mmap.flush()?;
                file.set_len(new_size as u64)?;
                // SAFETY: file is open for read+write.
                self.mmap = unsafe { MmapMut::map_mut(file)? };
            }
        }
        Ok(())
    }

    #[inline]
    pub(crate) fn as_slice(&self) -> &[u8] {
        &self.mmap
    }
    #[inline]
    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.mmap
    }

    pub(crate) fn flush(&self) -> io::Result<()> {
        self.mmap.flush()
    }
}

// ── BlobStore ─────────────────────────────────────────────────────────────────

/// Open-addressing hash table mapping `u128` keys to raw byte blobs stored in
/// a companion value region.
///
/// Can be backed by anonymous mmaps (transient, in-memory) or persistent files
/// (`blobs.keys` / `blobs.vals`) in a given directory.
pub(crate) struct BlobStore {
    key: GrowableMmap,
    val: GrowableMmap,
    /// Backing directory for persistent stores (`None` for anonymous ones).
    /// Used by [`compact`](BlobStore::compact) to stage and atomically swap in
    /// the rewritten `blobs.keys` / `blobs.vals` files.
    dir: Option<PathBuf>,
}

impl BlobStore {
    // ── Construction ──────────────────────────────────────────────────────────

    /// Create a new anonymous (transient) store.
    pub fn new_anon() -> Self {
        let mut key = GrowableMmap::new_anon(INITIAL_KEY_SIZE).expect("anon key mmap alloc failed");
        init_header(key.as_mut_slice(), INITIAL_CAPACITY);
        let val = GrowableMmap::new_anon(INITIAL_VAL_SIZE).expect("anon val mmap alloc failed");
        Self { key, val, dir: None }
    }

    /// Create a new persistent store in `dir`, creating `blobs.keys` and
    /// `blobs.vals`.
    pub fn create(dir: &Path) -> io::Result<Self> {
        let mut key = GrowableMmap::create_file(&dir.join("blobs.keys"), INITIAL_KEY_SIZE)?;
        init_header(key.as_mut_slice(), INITIAL_CAPACITY);
        key.flush()?;
        let val = GrowableMmap::create_file(&dir.join("blobs.vals"), INITIAL_VAL_SIZE)?;
        Ok(Self {
            key,
            val,
            dir: Some(dir.to_path_buf()),
        })
    }

    /// Open an existing persistent store from `dir`.
    pub fn open(dir: &Path) -> io::Result<Self> {
        // Complete or discard any compaction that was interrupted by a crash
        // before its file swap finished, so we always map a consistent pair.
        Self::recover_compaction(dir)?;
        let key = GrowableMmap::open_file(&dir.join("blobs.keys"))?;
        let val = GrowableMmap::open_file(&dir.join("blobs.vals"))?;
        validate_open(key.as_slice(), val.as_slice().len())?;
        Ok(Self {
            key,
            val,
            dir: Some(dir.to_path_buf()),
        })
    }

    /// Returns `true` if persistent store files exist in `dir`.
    pub fn exists(dir: &Path) -> bool {
        dir.join("blobs.keys").exists() && dir.join("blobs.vals").exists()
    }

    // ── Header accessors ──────────────────────────────────────────────────────

    fn header(&self) -> Header {
        read_header(self.key.as_slice())
    }

    fn set_header(&mut self, h: &Header) {
        write_header(self.key.as_mut_slice(), h);
    }

    // ── Public metrics ────────────────────────────────────────────────────────

    /// Number of live (occupied) entries.
    pub fn count(&self) -> usize {
        self.header().count as usize
    }

    /// Size the value region would shrink to if compacted now: each live blob's
    /// length rounded up to `VALUE_ALIGNMENT` (matching [`compact`]'s layout).
    ///
    /// [`compact`]: BlobStore::compact
    fn compacted_value_bytes(&self) -> u64 {
        let cap = self.header().capacity as usize;
        let kdata = self.key.as_slice();
        (0..cap)
            .map(|i| read_slot(kdata, i))
            .filter(|s| s.state == STATE_OCCUPIED)
            .map(|s| align_up(s.len as usize, VALUE_ALIGNMENT) as u64)
            .sum()
    }

    /// Fraction (`0.0..1.0`) of the value region that is reclaimable dead space
    /// left by overwritten or removed blobs. Returns `0.0` for an empty store.
    ///
    /// `upsert` appends every new blob and never reclaims the old bytes, so a
    /// key re-written N times leaves N−1 stale copies behind. This ratio drives
    /// the compaction trigger; it reads back to ≈0 immediately after [`compact`]
    /// (alignment padding is counted as live, not waste).
    ///
    /// [`compact`]: BlobStore::compact
    pub fn waste_ratio(&self) -> f64 {
        let total = self.header().value_write_pos;
        if total == 0 {
            return 0.0;
        }
        let compacted = self.compacted_value_bytes();
        total.saturating_sub(compacted) as f64 / total as f64
    }

    // ── Probe ─────────────────────────────────────────────────────────────────

    fn probe(&self, key: u128) -> (usize, bool) {
        let cap = self.header().capacity as usize;
        let start = hash_key(key) % cap;
        let kdata = self.key.as_slice();
        let mut first_tombstone: Option<usize> = None;
        let mut i = start;
        loop {
            let s = read_slot(kdata, i);
            match s.state {
                STATE_EMPTY => {
                    return (first_tombstone.unwrap_or(i), false);
                }
                STATE_OCCUPIED if s.key == key => {
                    return (i, true);
                }
                STATE_TOMBSTONE if first_tombstone.is_none() => {
                    first_tombstone = Some(i);
                }
                _ => {}
            }
            i = (i + 1) % cap;
            if i == start {
                return (first_tombstone.unwrap_or(start), false);
            }
        }
    }

    // ── Rehash ────────────────────────────────────────────────────────────────

    fn needs_rehash(&self) -> bool {
        let hdr = self.header();
        let used = hdr.count + hdr.tombstone_count;
        used * 10 > hdr.capacity * 7 // > 70% load
    }

    fn rehash(&mut self) {
        let hdr = self.header();
        let old_cap = hdr.capacity as usize;
        let new_cap = old_cap * 2;
        let new_key_size = HEADER_SIZE + new_cap * SLOT_SIZE;

        let entries: Vec<Slot> = (0..old_cap)
            .map(|i| read_slot(self.key.as_slice(), i))
            .filter(|s| s.state == STATE_OCCUPIED)
            .collect();

        self.key.ensure_capacity(new_key_size).expect("rehash key mmap grow failed");

        let kdata = self.key.as_mut_slice();
        kdata[HEADER_SIZE..HEADER_SIZE + new_cap * SLOT_SIZE].fill(0);
        write_header(
            kdata,
            &Header {
                capacity: new_cap as u64,
                count: 0,
                tombstone_count: 0,
                value_write_pos: hdr.value_write_pos,
            },
        );

        let live = entries.len() as u64;
        for slot in &entries {
            let start = hash_key(slot.key) % new_cap;
            let kdata = self.key.as_mut_slice();
            let mut i = start;
            loop {
                if read_slot(kdata, i).state == STATE_EMPTY {
                    write_slot(kdata, i, slot);
                    break;
                }
                i = (i + 1) % new_cap;
            }
        }

        let kdata = self.key.as_mut_slice();
        let mut h = read_header(kdata);
        h.count = live;
        write_header(kdata, &h);
    }

    // ── Mutation ──────────────────────────────────────────────────────────────

    /// Insert or replace the blob for `key`.
    ///
    /// Returns `true` if this was a new key (false if an existing one was updated).
    pub fn upsert(&mut self, key: u128, blob: &[u8]) -> bool {
        if self.needs_rehash() {
            self.rehash();
        }

        let hdr = self.header();
        let raw_offset = hdr.value_write_pos as usize;
        let aligned_offset = align_up(raw_offset, VALUE_ALIGNMENT);
        let new_val_end = aligned_offset + blob.len();

        self.val.ensure_capacity(new_val_end).expect("val mmap grow failed");

        self.val.as_mut_slice()[aligned_offset..aligned_offset + blob.len()].copy_from_slice(blob);

        let (idx, found) = self.probe(key);

        write_slot(
            self.key.as_mut_slice(),
            idx,
            &Slot {
                state: STATE_OCCUPIED,
                key,
                offset: aligned_offset as u64,
                len: blob.len() as u32,
            },
        );

        let mut h = self.header();
        h.value_write_pos = new_val_end as u64;
        if !found {
            h.count += 1;
        }
        self.set_header(&h);

        !found
    }

    /// Mark the entry for `key` as a tombstone.
    pub fn remove_key(&mut self, key: u128) {
        let (idx, found) = self.probe(key);
        if !found {
            return;
        }
        let kdata = self.key.as_mut_slice();
        kdata[slot_byte_offset(idx)] = STATE_TOMBSTONE;

        let mut h = read_header(kdata);
        h.count -= 1;
        h.tombstone_count += 1;
        write_header(kdata, &h);
    }

    // ── Query ─────────────────────────────────────────────────────────────────

    /// Return the blob for `key`, or `None` if the key is absent.
    pub fn get(&self, key: u128) -> Option<Vec<u8>> {
        let (idx, found) = self.probe(key);
        if !found {
            return None;
        }
        let s = read_slot(self.key.as_slice(), idx);
        let bytes = &self.val.as_slice()[s.offset as usize..s.offset as usize + s.len as usize];
        Some(bytes.to_vec())
    }

    /// Return all live `(key, blob)` pairs.
    pub fn iter_entries(&self) -> Vec<(u128, Vec<u8>)> {
        let cap = self.header().capacity as usize;
        (0..cap)
            .map(|i| read_slot(self.key.as_slice(), i))
            .filter(|s| s.state == STATE_OCCUPIED)
            .map(|s| {
                let bytes = self.val.as_slice()[s.offset as usize..s.offset as usize + s.len as usize].to_vec();
                (s.key, bytes)
            })
            .collect()
    }

    #[cfg(test)]
    fn live_keys(&self) -> Vec<u128> {
        self.iter_entries().into_iter().map(|(k, _)| k).collect()
    }

    // ── Compaction ──────────────────────────────────────────────────────────

    /// Reclaim dead space in the value region.
    ///
    /// Rewrites only the live blobs into a compact value region (dropping the
    /// bytes orphaned by overwrites and removals), rebuilds the key table to
    /// clear tombstones, and shrinks the value file. Live `key -> blob`
    /// mappings are preserved exactly. Returns the number of value-region bytes
    /// reclaimed.
    ///
    /// **Crash safety (persistent stores).** The compacted `blobs.keys` /
    /// `blobs.vals` are staged in `*.new` files (leaving the live pair
    /// untouched), fsynced, and only then swapped in under cover of a
    /// `compact.commit` marker. A crash at any point leaves the on-disk pair
    /// either fully old or fully new — never a torn mix of new offsets against
    /// an old value region. [`open`](BlobStore::open) finishes or rolls back an
    /// interrupted swap. The caller must hold the field write lock (no
    /// concurrent reader/writer touches the mmaps while they are remapped).
    pub fn compact(&mut self) -> io::Result<u64> {
        let before = self.header().value_write_pos;
        let cap = self.header().capacity as usize;

        // Build the compacted value region + slot placements directly from the
        // mmap, so the only large heap allocation is the single compacted
        // `val_buf` (≈ the live index size). No owned copy of the live blobs is
        // materialised, and the value region is never copied again to pad it.
        let mut val_buf: Vec<u8> = Vec::new();
        let mut placements: Vec<Slot> = Vec::new();
        let mut write_pos = 0usize;
        {
            let kdata = self.key.as_slice();
            let vdata = self.val.as_slice();
            for i in 0..cap {
                let s = read_slot(kdata, i);
                if s.state != STATE_OCCUPIED {
                    continue;
                }
                let blob = &vdata[s.offset as usize..s.offset as usize + s.len as usize];
                let aligned = align_up(write_pos, VALUE_ALIGNMENT);
                val_buf.resize(aligned, 0); // honour VALUE_ALIGNMENT padding
                val_buf.extend_from_slice(blob);
                placements.push(Slot {
                    state: STATE_OCCUPIED,
                    key: s.key,
                    offset: aligned as u64,
                    len: s.len,
                });
                write_pos = aligned + blob.len();
            }
        }
        let key_buf = build_key_table(cap, write_pos as u64, &placements);

        match self.dir.clone() {
            // Persistent: stage the new pair and swap it in atomically.
            Some(dir) => self.commit_compaction(&dir, &key_buf, &val_buf)?,
            // Anonymous (transient) store: rewrite the in-memory mmaps in place
            // — there is nothing on disk to make crash-safe.
            None => self.rewrite_in_place(&key_buf, &val_buf)?,
        }

        Ok(before.saturating_sub(write_pos as u64))
    }

    /// Overwrite the in-memory mmaps with a freshly built key table and value
    /// region (anonymous stores only).
    fn rewrite_in_place(&mut self, key_buf: &[u8], val_buf: &[u8]) -> io::Result<()> {
        self.key.as_mut_slice()[..key_buf.len()].copy_from_slice(key_buf);
        self.val.truncate(val_buf.len().max(INITIAL_VAL_SIZE))?;
        self.val.as_mut_slice()[..val_buf.len()].copy_from_slice(val_buf);
        Ok(())
    }

    /// Atomically swap the staged compacted files into place for a persistent
    /// store, then remap onto them. See [`compact`](BlobStore::compact) for the
    /// crash-safety contract.
    fn commit_compaction(&mut self, dir: &Path, key_buf: &[u8], val_buf: &[u8]) -> io::Result<()> {
        let keys = dir.join("blobs.keys");
        let vals = dir.join("blobs.vals");
        let keys_new = dir.join("blobs.keys.new");
        let vals_new = dir.join("blobs.vals.new");
        let commit = dir.join("compact.commit");

        // 1. Stage the new pair (the live files stay untouched) and fsync it.
        //    Write val_buf directly and zero-extend the file to the minimum size
        //    (so the reopened mmap is non-empty) rather than copying val_buf to
        //    pad it — the header's value_write_pos marks the real end, so the
        //    zero tail is just slack.
        {
            let mut f = File::create(&vals_new)?;
            f.write_all(val_buf)?;
            if (val_buf.len() as u64) < INITIAL_VAL_SIZE as u64 {
                f.set_len(INITIAL_VAL_SIZE as u64)?;
            }
            f.sync_all()?;
        }
        write_file_sync(&keys_new, key_buf)?;
        fsync_dir(dir)?;

        // 2. Commit point: once this marker is durable, recovery *completes* the
        //    swap; before it, recovery discards the staged files.
        write_file_sync(&commit, &[])?;
        fsync_dir(dir)?;

        // 3. Swap both files into place (rename is atomic per file), then drop
        //    the marker. An interrupt here is finished idempotently on open.
        std::fs::rename(&vals_new, &vals)?;
        std::fs::rename(&keys_new, &keys)?;
        fsync_dir(dir)?;
        std::fs::remove_file(&commit)?;
        fsync_dir(dir)?;

        // 4. Remap onto the new files (safe: the caller holds the write lock).
        self.val = GrowableMmap::open_file(&vals)?;
        self.key = GrowableMmap::open_file(&keys)?;
        Ok(())
    }

    /// Finish or roll back a compaction interrupted before its swap completed.
    ///
    /// If the `compact.commit` marker is present the staged `*.new` files are
    /// complete and fsynced, so the swap is replayed idempotently. Otherwise any
    /// staged files are partial and are discarded, leaving the live pair intact.
    fn recover_compaction(dir: &Path) -> io::Result<()> {
        let keys = dir.join("blobs.keys");
        let vals = dir.join("blobs.vals");
        let keys_new = dir.join("blobs.keys.new");
        let vals_new = dir.join("blobs.vals.new");
        let commit = dir.join("compact.commit");

        if commit.exists() {
            if vals_new.exists() {
                std::fs::rename(&vals_new, &vals)?;
            }
            if keys_new.exists() {
                std::fs::rename(&keys_new, &keys)?;
            }
            fsync_dir(dir)?;
            std::fs::remove_file(&commit)?;
            fsync_dir(dir)?;
        } else {
            let mut changed = false;
            for stale in [&keys_new, &vals_new] {
                if stale.exists() {
                    std::fs::remove_file(stale)?;
                    changed = true;
                }
            }
            if changed {
                fsync_dir(dir)?;
            }
        }
        Ok(())
    }

    // ── Persistence ───────────────────────────────────────────────────────────

    /// Flush both mmaps to disk (no-op for anonymous stores).
    pub fn flush(&self) -> io::Result<()> {
        self.key.flush()?;
        self.val.flush()
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

#[inline]
fn align_up(offset: usize, align: usize) -> usize {
    (offset + align - 1) & !(align - 1)
}

/// Build a fresh key-table image (header + `capacity` slots) holding exactly
/// `placements`, with no tombstones. Used to stage a compacted `blobs.keys`.
fn build_key_table(capacity: usize, value_write_pos: u64, placements: &[Slot]) -> Vec<u8> {
    let mut buf = vec![0u8; HEADER_SIZE + capacity * SLOT_SIZE];
    write_header(
        &mut buf,
        &Header {
            capacity: capacity as u64,
            count: placements.len() as u64,
            tombstone_count: 0,
            value_write_pos,
        },
    );
    // Live count never exceeds the 70% rehash load factor, so an EMPTY slot
    // always exists and the probe terminates.
    for slot in placements {
        let mut i = hash_key(slot.key) % capacity;
        loop {
            if read_slot(&buf, i).state == STATE_EMPTY {
                write_slot(&mut buf, i, slot);
                break;
            }
            i = (i + 1) % capacity;
        }
    }
    buf
}

/// Write `bytes` to `path` (truncating any existing file) and fsync the file.
fn write_file_sync(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut f = File::create(path)?;
    f.write_all(bytes)?;
    f.sync_all()
}

/// fsync a directory so a contained rename/create/unlink is durable.
fn fsync_dir(dir: &Path) -> io::Result<()> {
    File::open(dir)?.sync_all()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_get_roundtrip() {
        let mut store = BlobStore::new_anon();
        assert!(store.upsert(42, b"hello world"));
        assert_eq!(store.get(42).unwrap(), b"hello world");
        assert_eq!(store.count(), 1);
    }

    #[test]
    fn update_existing_overwrites() {
        let mut store = BlobStore::new_anon();
        assert!(store.upsert(10, b"first"));
        assert!(!store.upsert(10, b"second"));
        assert_eq!(store.get(10).unwrap(), b"second");
        assert_eq!(store.count(), 1);
    }

    #[test]
    fn remove_marks_tombstone() {
        let mut store = BlobStore::new_anon();
        store.upsert(1, b"data");
        store.upsert(2, b"other");
        store.remove_key(1);
        assert_eq!(store.count(), 1);
        assert!(store.get(1).is_none());
        assert!(store.get(2).is_some());
    }

    #[test]
    fn live_keys_returns_active_entries() {
        let mut store = BlobStore::new_anon();
        store.upsert(100, b"a");
        store.upsert(200, b"b");
        store.upsert(300, b"c");
        store.remove_key(200);
        let mut keys = store.live_keys();
        keys.sort();
        assert_eq!(keys, vec![100, 300]);
    }

    #[test]
    fn rehash_preserves_entries() {
        let mut store = BlobStore::new_anon();
        for i in 0u128..20 {
            store.upsert(i * 0x1_0000, format!("value_{}", i).as_bytes());
        }
        assert_eq!(store.count(), 20);
        for i in 0u128..20 {
            let got = store.get(i * 0x1_0000).unwrap();
            assert_eq!(got, format!("value_{}", i).as_bytes());
        }
    }

    #[test]
    fn waste_ratio_grows_with_overwrites_and_resets_after_compact() {
        let mut store = BlobStore::new_anon();
        // One key rewritten many times with a growing blob — the append-only
        // value region accumulates dead copies.
        for n in 1..=50usize {
            store.upsert(7, &vec![0xABu8; n * 8]);
        }
        assert_eq!(store.count(), 1, "still a single live key");
        assert!(
            store.waste_ratio() > 0.8,
            "heavy overwrite must show high waste, got {}",
            store.waste_ratio()
        );

        let live_before = store.get(7).unwrap();
        let reclaimed = store.compact().unwrap();
        assert!(reclaimed > 0, "compaction must reclaim bytes");

        assert_eq!(store.count(), 1);
        assert_eq!(store.get(7).unwrap(), live_before, "live blob must survive compaction byte-for-byte");
        assert!(
            store.waste_ratio() < 0.01,
            "waste must read ≈0 after compaction, got {}",
            store.waste_ratio()
        );
    }

    #[test]
    fn compact_preserves_all_live_entries_and_drops_tombstones() {
        let mut store = BlobStore::new_anon();
        for i in 0u128..30 {
            store.upsert(i, format!("v{i}").as_bytes());
        }
        // Overwrite half (dead copies) and remove a third (tombstones).
        for i in 0u128..15 {
            store.upsert(i, format!("v{i}-updated-{}", "x".repeat(20)).as_bytes());
        }
        for i in 20u128..30 {
            store.remove_key(i);
        }
        let expected: Vec<(u128, Vec<u8>)> = {
            let mut e = store.iter_entries();
            e.sort_by_key(|(k, _)| *k);
            e
        };

        store.compact().unwrap();

        let mut got = store.iter_entries();
        got.sort_by_key(|(k, _)| *k);
        assert_eq!(got, expected, "live set must be identical after compaction");
        assert_eq!(store.count(), 20);
        // The removed keys stay gone, and a fresh insert still works post-compact.
        assert!(store.get(25).is_none());
        assert!(store.upsert(99, b"fresh"));
        assert_eq!(store.get(99).unwrap(), b"fresh");
    }

    #[test]
    fn compact_persistent_shrinks_value_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut store = BlobStore::create(dir.path()).unwrap();
        for n in 1..=200usize {
            store.upsert(1, &vec![0x5Au8; n]);
            store.upsert(2, &vec![0x3Cu8; n]);
        }
        store.flush().unwrap();
        let bloated = std::fs::metadata(dir.path().join("blobs.vals")).unwrap().len();

        store.compact().unwrap();
        let compacted = std::fs::metadata(dir.path().join("blobs.vals")).unwrap().len();
        assert!(compacted < bloated, "value file must shrink on disk: {compacted} !< {bloated}");

        // Reopen and confirm the live data survived the shrink.
        drop(store);
        let store = BlobStore::open(dir.path()).unwrap();
        assert_eq!(store.get(1).unwrap(), vec![0x5Au8; 200]);
        assert_eq!(store.get(2).unwrap(), vec![0x3Cu8; 200]);
    }

    // Simulate a crash *after* the commit marker is durable but *before* the
    // file swap finishes (staged files still present). open() must complete it.
    #[test]
    fn open_completes_interrupted_swap_after_commit_marker() {
        let dir = tempfile::TempDir::new().unwrap();
        {
            let mut store = BlobStore::create(dir.path()).unwrap();
            store.upsert(1, b"old-one");
            store.upsert(2, b"old-two");
            store.flush().unwrap();
        }
        // Hand-stage a newer compacted pair plus the commit marker, leaving the
        // *old* live files in place (mid-swap state).
        let mut newstore = BlobStore::create(dir.path()).unwrap();
        newstore.upsert(1, b"NEW-one");
        newstore.upsert(2, b"NEW-two");
        newstore.upsert(3, b"NEW-three");
        let key_img = std::fs::read(dir.path().join("blobs.keys")).unwrap();
        let val_img = std::fs::read(dir.path().join("blobs.vals")).unwrap();
        drop(newstore);
        // Restore the OLD live pair, then stage the NEW pair as *.new + marker.
        let mut old = BlobStore::create(dir.path()).unwrap();
        old.upsert(1, b"old-one");
        old.upsert(2, b"old-two");
        old.flush().unwrap();
        drop(old);
        std::fs::write(dir.path().join("blobs.keys.new"), &key_img).unwrap();
        std::fs::write(dir.path().join("blobs.vals.new"), &val_img).unwrap();
        std::fs::write(dir.path().join("compact.commit"), []).unwrap();

        let store = BlobStore::open(dir.path()).unwrap();
        assert_eq!(store.get(1).unwrap(), b"NEW-one", "swap must be completed forward");
        assert_eq!(store.get(2).unwrap(), b"NEW-two");
        assert_eq!(store.get(3).unwrap(), b"NEW-three");
        assert!(!dir.path().join("compact.commit").exists(), "marker cleared");
        assert!(!dir.path().join("blobs.keys.new").exists(), "staged file consumed");
    }

    // Simulate a crash *before* the commit marker (staged files may be partial).
    // open() must discard them and keep the intact live pair.
    #[test]
    fn open_discards_staged_files_without_commit_marker() {
        let dir = tempfile::TempDir::new().unwrap();
        {
            let mut store = BlobStore::create(dir.path()).unwrap();
            store.upsert(1, b"live-one");
            store.flush().unwrap();
        }
        // Garbage half-written staging files, no commit marker.
        std::fs::write(dir.path().join("blobs.keys.new"), b"partial-garbage").unwrap();
        std::fs::write(dir.path().join("blobs.vals.new"), b"partial").unwrap();

        let store = BlobStore::open(dir.path()).unwrap();
        assert_eq!(store.get(1).unwrap(), b"live-one", "live pair must be untouched");
        assert!(!dir.path().join("blobs.keys.new").exists(), "stale staging discarded");
        assert!(!dir.path().join("blobs.vals.new").exists());
    }

    #[test]
    fn persistent_roundtrip() {
        let dir = tempfile::TempDir::new().unwrap();
        {
            let mut store = BlobStore::create(dir.path()).unwrap();
            store.upsert(1, b"one");
            store.upsert(2, b"two");
            store.flush().unwrap();
        }
        let store = BlobStore::open(dir.path()).unwrap();
        assert_eq!(store.count(), 2);
        assert_eq!(store.get(1).unwrap(), b"one");
        assert_eq!(store.get(2).unwrap(), b"two");
    }

    // ── Header / bounds validation on open ──────────────────────────────────
    //
    // A corrupt or wrong-version key/value file must be rejected with
    // io::ErrorKind::InvalidData, never opened (which would later panic or read
    // out of bounds). Each test seeds a valid store, then mutates the on-disk
    // image and asserts open() fails.

    fn seed(dir: &Path) {
        let mut store = BlobStore::create(dir).unwrap();
        store.upsert(1, b"one");
        store.upsert(2, b"two");
        store.flush().unwrap();
    }

    fn assert_invalid(dir: &Path) {
        let err = BlobStore::open(dir).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData, "expected InvalidData, got: {err}");
    }

    #[test]
    fn open_rejects_bad_magic() {
        let dir = tempfile::TempDir::new().unwrap();
        seed(dir.path());
        let mut key = std::fs::read(dir.path().join("blobs.keys")).unwrap();
        key[0] ^= 0xFF; // corrupt the magic
        std::fs::write(dir.path().join("blobs.keys"), &key).unwrap();
        assert_invalid(dir.path());
    }

    #[test]
    fn open_rejects_unsupported_version() {
        let dir = tempfile::TempDir::new().unwrap();
        seed(dir.path());
        let mut key = std::fs::read(dir.path().join("blobs.keys")).unwrap();
        key[8..12].copy_from_slice(&(VERSION + 1).to_le_bytes());
        std::fs::write(dir.path().join("blobs.keys"), &key).unwrap();
        assert_invalid(dir.path());
    }

    #[test]
    fn open_rejects_truncated_header() {
        let dir = tempfile::TempDir::new().unwrap();
        seed(dir.path());
        std::fs::write(dir.path().join("blobs.keys"), [0u8; HEADER_SIZE - 1]).unwrap();
        assert_invalid(dir.path());
    }

    #[test]
    fn open_rejects_zero_capacity() {
        let dir = tempfile::TempDir::new().unwrap();
        seed(dir.path());
        let mut key = std::fs::read(dir.path().join("blobs.keys")).unwrap();
        key[16..24].copy_from_slice(&0u64.to_le_bytes()); // capacity = 0
        std::fs::write(dir.path().join("blobs.keys"), &key).unwrap();
        assert_invalid(dir.path());
    }

    #[test]
    fn open_rejects_capacity_larger_than_key_file() {
        let dir = tempfile::TempDir::new().unwrap();
        seed(dir.path());
        let mut key = std::fs::read(dir.path().join("blobs.keys")).unwrap();
        key[16..24].copy_from_slice(&(1u64 << 40).to_le_bytes()); // huge pow2 capacity
        std::fs::write(dir.path().join("blobs.keys"), &key).unwrap();
        assert_invalid(dir.path());
    }

    #[test]
    fn open_rejects_slot_blob_past_value_region() {
        let dir = tempfile::TempDir::new().unwrap();
        seed(dir.path());
        // Inflate value_write_pos to a sane bound first so the slot bound is the
        // failing check, then point slot 0's blob past it.
        let mut key = std::fs::read(dir.path().join("blobs.keys")).unwrap();
        // Find an occupied slot and corrupt its offset to exceed value_write_pos.
        let vwp = u64::from_le_bytes(key[48..56].try_into().unwrap());
        for i in 0..INITIAL_CAPACITY {
            let b = slot_byte_offset(i);
            if key[b] == STATE_OCCUPIED {
                key[b + 24..b + 32].copy_from_slice(&(vwp + 1).to_le_bytes()); // offset past vwp
                break;
            }
        }
        std::fs::write(dir.path().join("blobs.keys"), &key).unwrap();
        assert_invalid(dir.path());
    }

    #[test]
    fn open_rejects_value_write_pos_past_value_file() {
        let dir = tempfile::TempDir::new().unwrap();
        seed(dir.path());
        let val_len = std::fs::metadata(dir.path().join("blobs.vals")).unwrap().len();
        let mut key = std::fs::read(dir.path().join("blobs.keys")).unwrap();
        key[48..56].copy_from_slice(&(val_len + 1).to_le_bytes()); // vwp beyond value file
        std::fs::write(dir.path().join("blobs.keys"), &key).unwrap();
        assert_invalid(dir.path());
    }
}
