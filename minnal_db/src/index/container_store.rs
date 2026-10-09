//! Container store: the `u128 → Container` map behind a [`RoaringBitmap`].
//!
//! Two backings, one API:
//!
//! - **Heap** (`ContainerStore::new_anon`): a sorted map of containers,
//!   changed in place. Every transient bitmap uses it — loaded copies, query
//!   results, replay batches. Its memory is the bitmap's real size.
//! - **Mapped** (`ContainerStore::create` / `ContainerStore::open`): the
//!   two-file layout below, for persistent bitmaps.
//!
//! The heap backing exists because the mapped layout is append-only: every
//! change writes a new copy of the container and nothing is reclaimed until the
//! store is dropped. Backing a transient bitmap with it (as an anonymous map)
//! made 65,536 inserts into one container cost 514 MiB of memory for an 8 KB
//! bitmap. Do not put transient bitmaps on the mapped backing.
//!
//! [`RoaringBitmap`]: crate::index::RoaringBitmap
//!
//! The mapped layout stores the map in two memory-mapped regions:
//!
//! **Key file** (`containers.keys`):
//! - 64-byte header (magic, version, capacity, counts, cardinality, value write pos)
//! - Fixed-size 48-byte slots forming an open-addressing linear-probing hash table
//!
//! **Value file** (`containers.vals`):
//! - Append-only sequence of rkyv-serialized [`Container`] blobs
//! - Each slot in the key file records the byte offset and length of its blob
//!
//! Growth is handled transparently by remapping.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;

use memmap2::MmapMut;
use rkyv::api::high::{HighDeserializer, HighValidator};
use rkyv::ser::allocator::ArenaHandle;
use rkyv::util::AlignedVec;
use rkyv::{Archive, Deserialize, Serialize, rancor};

use crate::index::container::Container;

// ── Layout constants ─────────────────────────────────────────────────────────

const MAGIC: u64 = 0x4D494E4E414C4249; // "MINNALBI"
const VERSION: u32 = 1;
pub(crate) const HEADER_SIZE: usize = 64;
pub(crate) const SLOT_SIZE: usize = 48;
const INITIAL_CAPACITY: usize = 16;
pub(crate) const INITIAL_KEY_SIZE: usize = HEADER_SIZE + INITIAL_CAPACITY * SLOT_SIZE;
const INITIAL_VAL_SIZE: usize = 4096;
const VALUE_ALIGNMENT: usize = 16;

// Slot state flags
const STATE_EMPTY: u8 = 0;
const STATE_OCCUPIED: u8 = 1;
const STATE_TOMBSTONE: u8 = 2;

// ── Header ────────────────────────────────────────────────────────────────────

/// In-memory copy of the 64-byte key-file header.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Header {
    pub capacity: u64,        // number of hash slots (power of 2)
    pub count: u64,           // live (occupied) entries
    pub tombstone_count: u64, // tombstone entries
    pub cardinality: u64,     // total bit count across all containers
    pub value_write_pos: u64, // next write offset in the value file
}

pub(crate) fn read_header(data: &[u8]) -> Header {
    Header {
        capacity: u64::from_le_bytes(data[16..24].try_into().unwrap()),
        count: u64::from_le_bytes(data[24..32].try_into().unwrap()),
        tombstone_count: u64::from_le_bytes(data[32..40].try_into().unwrap()),
        cardinality: u64::from_le_bytes(data[40..48].try_into().unwrap()),
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
    data[40..48].copy_from_slice(&h.cardinality.to_le_bytes());
    data[48..56].copy_from_slice(&h.value_write_pos.to_le_bytes());
    data[56..64].fill(0); // reserved
}

fn init_header(data: &mut [u8], capacity: usize) {
    write_header(
        data,
        &Header {
            capacity: capacity as u64,
            count: 0,
            tombstone_count: 0,
            cardinality: 0,
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
//   24..32  offset: u64 (LE)  — byte offset in value file
//   32..36  len: u32   (LE)  — byte length of serialized container
//   36..40  card: u32  (LE)  — container cardinality (cached for O(1) delta)
//   40..48  pad

#[derive(Debug, Clone, Copy)]
pub(crate) struct Slot {
    pub state: u8,
    pub key: u128,
    pub offset: u64,
    pub len: u32,
    pub card: u32,
}

pub(crate) fn slot_byte_offset(i: usize) -> usize {
    HEADER_SIZE + i * SLOT_SIZE
}

fn read_slot(data: &[u8], i: usize) -> Slot {
    let b = slot_byte_offset(i);
    Slot {
        state: data[b],
        key: u128::from_le_bytes(data[b + 8..b + 24].try_into().unwrap()),
        offset: u64::from_le_bytes(data[b + 24..b + 32].try_into().unwrap()),
        len: u32::from_le_bytes(data[b + 32..b + 36].try_into().unwrap()),
        card: u32::from_le_bytes(data[b + 36..b + 40].try_into().unwrap()),
    }
}

fn write_slot(data: &mut [u8], i: usize, s: &Slot) {
    let b = slot_byte_offset(i);
    data[b] = s.state;
    data[b + 1..b + 8].fill(0);
    data[b + 8..b + 24].copy_from_slice(&s.key.to_le_bytes());
    data[b + 24..b + 32].copy_from_slice(&s.offset.to_le_bytes());
    data[b + 32..b + 36].copy_from_slice(&s.len.to_le_bytes());
    data[b + 36..b + 40].copy_from_slice(&s.card.to_le_bytes());
    data[b + 40..b + 48].fill(0);
}

// ── Open-time validation ───────────────────────────────────────────────────────

fn invalid(msg: impl std::fmt::Display) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("container store: {msg}"))
}

/// Validate a just-opened store's on-disk images before any read trusts them.
///
/// Guards the hot path against corrupt/wrong files: a zero `capacity` would
/// divide by zero in the hash modulo, a `capacity` larger than the key file
/// slices out of bounds, and a slot whose `offset+len` runs past the value
/// region reads arbitrary bytes. Every failure is a controlled
/// [`io::ErrorKind::InvalidData`], never a panic.
fn validate_open(key: &[u8], val_len: usize) -> io::Result<()> {
    if key.len() < HEADER_SIZE {
        return Err(invalid("key file smaller than header"));
    }
    let magic = u64::from_le_bytes(key[0..8].try_into().unwrap());
    if magic != MAGIC {
        return Err(invalid(format!("bad magic {magic:#018x} (not a minnal container store)")));
    }
    let version = u32::from_le_bytes(key[8..12].try_into().unwrap());
    if version != VERSION {
        return Err(invalid(format!("unsupported on-disk version {version} (expected {VERSION})")));
    }
    let hdr = read_header(key);
    if hdr.capacity == 0 || !hdr.capacity.is_power_of_two() {
        return Err(invalid(format!("capacity {} is not a non-zero power of two", hdr.capacity)));
    }
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

// ── Container serialisation ───────────────────────────────────────────────────

fn ser_container(c: &Container) -> AlignedVec
where
    Container: for<'a> Serialize<rkyv::api::high::HighSerializer<AlignedVec, ArenaHandle<'a>, rancor::Error>>,
{
    rkyv::to_bytes::<rancor::Error>(c).expect("container serialize failed")
}

fn deser_container(bytes: &[u8]) -> Container
where
    <Container as Archive>::Archived:
        Deserialize<Container, HighDeserializer<rancor::Error>> + for<'a> rkyv::bytecheck::CheckBytes<HighValidator<'a, rancor::Error>>,
{
    // Copy into an AlignedVec to guarantee rkyv's alignment requirement.
    let mut buf = AlignedVec::<16>::new();
    buf.extend_from_slice(bytes);
    let archived = rkyv::access::<rkyv::Archived<Container>, rancor::Error>(&buf).expect("container access failed");
    rkyv::deserialize::<Container, rancor::Error>(archived).expect("container deserialize failed")
}

// ── GrowableMmap ─────────────────────────────────────────────────────────────

struct GrowableMmap {
    mmap: MmapMut,
    file: Option<File>,
}

impl GrowableMmap {
    fn create_file(path: &Path, initial_size: usize) -> io::Result<Self> {
        let file = OpenOptions::new().read(true).write(true).create(true).truncate(true).open(path)?;
        file.set_len(initial_size as u64)?;
        // SAFETY: file is open for read+write, no other mmap on this file.
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        Ok(Self { mmap, file: Some(file) })
    }

    fn open_file(path: &Path) -> io::Result<Self> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let size = file.metadata()?.len() as usize;
        let mmap = if size > 0 {
            // SAFETY: file is open for read+write.
            unsafe { MmapMut::map_mut(&file)? }
        } else {
            // Empty file: return anon mmap (will never be written; caller checks)
            MmapMut::map_anon(INITIAL_KEY_SIZE)?
        };
        Ok(Self { mmap, file: Some(file) })
    }

    /// Ensure the mmap is at least `needed` bytes. Grows by doubling.
    fn ensure_capacity(&mut self, needed: usize) -> io::Result<()> {
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

    #[inline]
    fn as_slice(&self) -> &[u8] {
        &self.mmap
    }
    #[inline]
    fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.mmap
    }
    #[allow(dead_code)]
    #[inline]
    fn len(&self) -> usize {
        self.mmap.len()
    }

    fn flush(&self) -> io::Result<()> {
        self.mmap.flush()
    }
}

// ── ContainerStore ─────────────────────────────────────────────────────────────────

/// The mapped backing: an open-addressing hash table mapping `u128` container
/// keys to serialized [`Container`] blobs stored in a companion value region.
struct MappedStore {
    key: GrowableMmap,
    val: GrowableMmap,
}

impl MappedStore {
    // ── Construction ─────────────────────────────────────────────────────────

    /// Create a new persistent store, creating both files under `dir`.
    pub fn create(dir: &Path) -> io::Result<Self> {
        let mut key = GrowableMmap::create_file(&dir.join("containers.keys"), INITIAL_KEY_SIZE)?;
        init_header(key.as_mut_slice(), INITIAL_CAPACITY);
        key.flush()?;
        let val = GrowableMmap::create_file(&dir.join("containers.vals"), INITIAL_VAL_SIZE)?;
        Ok(Self { key, val })
    }

    /// Open an existing persistent store from `dir`.
    pub fn open(dir: &Path) -> io::Result<Self> {
        let key = GrowableMmap::open_file(&dir.join("containers.keys"))?;
        let val = GrowableMmap::open_file(&dir.join("containers.vals"))?;
        validate_open(key.as_slice(), val.as_slice().len())?;
        Ok(Self { key, val })
    }

    // ── Header accessors ─────────────────────────────────────────────────────

    pub fn header(&self) -> Header {
        read_header(self.key.as_slice())
    }

    fn set_header(&mut self, h: &Header) {
        write_header(self.key.as_mut_slice(), h);
    }

    // ── Public metrics ───────────────────────────────────────────────────────

    /// Cached total bit count across all containers.
    pub fn cardinality(&self) -> usize {
        self.header().cardinality as usize
    }

    /// Number of live (occupied) entries.
    pub fn count(&self) -> usize {
        self.header().count as usize
    }

    // ── Probe helpers ────────────────────────────────────────────────────────

    /// Find the slot index for `key`.
    ///
    /// Returns `(slot_index, found)` where `found` is true if the key is
    /// present and `slot_index` points to either the occupied slot or the
    /// first suitable empty/tombstone slot for insertion.
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
                // Full table with only tombstones (shouldn't happen with load-factor guard).
                return (first_tombstone.unwrap_or(start), false);
            }
        }
    }

    // ── Rehash ───────────────────────────────────────────────────────────────

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

        // Collect all live slots before growing (borrows key mmap).
        let entries: Vec<Slot> = (0..old_cap)
            .map(|i| read_slot(self.key.as_slice(), i))
            .filter(|s| s.state == STATE_OCCUPIED)
            .collect();

        self.key.ensure_capacity(new_key_size).expect("rehash key mmap grow failed");

        // Zero all slot space and re-write header with new capacity.
        let kdata = self.key.as_mut_slice();
        kdata[HEADER_SIZE..HEADER_SIZE + new_cap * SLOT_SIZE].fill(0);
        write_header(
            kdata,
            &Header {
                capacity: new_cap as u64,
                count: 0,
                tombstone_count: 0,
                cardinality: hdr.cardinality,
                value_write_pos: hdr.value_write_pos,
            },
        );

        // Re-insert live entries (no value file writes — reuse existing offsets).
        let live = entries.len() as u64;
        for slot in &entries {
            let cap = new_cap;
            let start = hash_key(slot.key) % cap;
            let kdata = self.key.as_mut_slice();
            let mut i = start;
            loop {
                if read_slot(kdata, i).state == STATE_EMPTY {
                    write_slot(kdata, i, slot);
                    break;
                }
                i = (i + 1) % cap;
            }
        }

        // Fix count in header.
        let kdata = self.key.as_mut_slice();
        let mut h = read_header(kdata);
        h.count = live;
        write_header(kdata, &h);
    }

    // ── Mutation ─────────────────────────────────────────────────────────────

    /// Insert or replace the container for `key`.
    ///
    /// Returns `true` if this was a new key (false if an existing one was updated).
    pub fn upsert(&mut self, key: u128, container: &Container) -> bool {
        if self.needs_rehash() {
            self.rehash();
        }

        let blob = ser_container(container);
        let new_card = container.cardinality() as u32;

        // Align value write position.
        let hdr = self.header();
        let raw_offset = hdr.value_write_pos as usize;
        let aligned_offset = align_up(raw_offset, VALUE_ALIGNMENT);
        let new_val_end = aligned_offset + blob.len();

        // Grow value region if needed.
        self.val.ensure_capacity(new_val_end).expect("val mmap grow failed");

        // Write blob.
        self.val.as_mut_slice()[aligned_offset..aligned_offset + blob.len()].copy_from_slice(&blob);

        // Find slot.
        let (idx, found) = self.probe(key);
        let old_card = if found { read_slot(self.key.as_slice(), idx).card } else { 0 };

        // Write slot.
        write_slot(
            self.key.as_mut_slice(),
            idx,
            &Slot {
                state: STATE_OCCUPIED,
                key,
                offset: aligned_offset as u64,
                len: crate::index::blob_store::u32_len(blob.len(), "container blob"),
                card: new_card,
            },
        );

        // Update header.
        let mut h = self.header();
        h.value_write_pos = new_val_end as u64;
        if !found {
            h.count += 1;
        }
        h.cardinality = (h.cardinality as i64 + new_card as i64 - old_card as i64) as u64;
        self.set_header(&h);

        !found
    }

    /// Remove the container for `key`. Returns the removed container's
    /// cardinality, or 0 if the key was not present.
    pub fn remove_key(&mut self, key: u128) -> u32 {
        let (idx, found) = self.probe(key);
        if !found {
            return 0;
        }
        let old_card = read_slot(self.key.as_slice(), idx).card;

        // Mark tombstone.
        let kdata = self.key.as_mut_slice();
        kdata[slot_byte_offset(idx)] = STATE_TOMBSTONE;

        let mut h = read_header(kdata);
        h.count -= 1;
        h.tombstone_count += 1;
        h.cardinality = h.cardinality.saturating_sub(old_card as u64);
        write_header(kdata, &h);

        old_card
    }

    /// Remove all entries and reset both regions.
    pub fn clear(&mut self) {
        let cap = self.header().capacity as usize;
        // Zero all slots.
        let key_size = HEADER_SIZE + cap * SLOT_SIZE;
        self.key.as_mut_slice()[..key_size].fill(0);
        init_header(self.key.as_mut_slice(), cap);
    }

    // ── Query ────────────────────────────────────────────────────────────────

    /// Deserialise and return the container for `key`, or `None`.
    pub fn get(&self, key: u128) -> Option<Container> {
        let (idx, found) = self.probe(key);
        if !found {
            return None;
        }
        let s = read_slot(self.key.as_slice(), idx);
        let bytes = &self.val.as_slice()[s.offset as usize..s.offset as usize + s.len as usize];
        Some(deser_container(bytes))
    }

    /// True if the key has an entry.
    #[allow(dead_code)]
    pub fn contains_key(&self, key: u128) -> bool {
        self.probe(key).1
    }

    // ── Sorted access ────────────────────────────────────────────────────────

    /// Return `(key, card)` for every live entry, sorted by key.
    ///
    /// No container deserialisation — reads only the key file.
    pub fn sorted_key_cards(&self) -> Vec<(u128, u32)> {
        let cap = self.header().capacity as usize;
        let mut out: Vec<(u128, u32)> = (0..cap)
            .map(|i| read_slot(self.key.as_slice(), i))
            .filter(|s| s.state == STATE_OCCUPIED)
            .map(|s| (s.key, s.card))
            .collect();
        out.sort_unstable_by_key(|&(k, _)| k);
        out
    }

    /// Return all live keys sorted.
    pub fn sorted_keys(&self) -> Vec<u128> {
        let cap = self.header().capacity as usize;
        let mut keys: Vec<u128> = (0..cap)
            .map(|i| read_slot(self.key.as_slice(), i))
            .filter(|s| s.state == STATE_OCCUPIED)
            .map(|s| s.key)
            .collect();
        keys.sort_unstable();
        keys
    }

    /// Deserialise and return all `(key, container)` pairs, sorted by key.
    pub fn sorted_entries(&self) -> Vec<(u128, Container)> {
        let cap = self.header().capacity as usize;
        let mut slots: Vec<Slot> = (0..cap)
            .map(|i| read_slot(self.key.as_slice(), i))
            .filter(|s| s.state == STATE_OCCUPIED)
            .collect();
        slots.sort_unstable_by_key(|s| s.key);
        slots
            .into_iter()
            .map(|s| {
                let bytes = &self.val.as_slice()[s.offset as usize..s.offset as usize + s.len as usize];
                (s.key, deser_container(bytes))
            })
            .collect()
    }

    // ── Persistence ──────────────────────────────────────────────────────────

    /// Flush both mmaps to disk (no-op for anonymous stores).
    pub fn flush(&self) -> io::Result<()> {
        self.key.flush()?;
        self.val.flush()
    }
}

// ── HeapStore ────────────────────────────────────────────────────────────────

/// Fixed per-container cost of a heap-backed store beyond the container's own
/// buffer: the map entry (key + `Container` enum) plus B-tree node overhead.
const HEAP_ENTRY_OVERHEAD: usize = std::mem::size_of::<(u128, Container)>() + 16;

/// Heap bytes one stored container accounts for.
fn heap_cost(c: &Container) -> usize {
    c.heap_bytes() + HEAP_ENTRY_OVERHEAD
}

/// The heap backing: containers in a sorted map, changed in place.
#[derive(Default)]
struct HeapStore {
    map: BTreeMap<u128, Container>,
    /// Total bit count across all containers, kept current on every change.
    cardinality: u64,
    /// Heap bytes held, kept current on every change (see [`heap_cost`]).
    heap_bytes: usize,
}

impl HeapStore {
    fn upsert(&mut self, key: u128, container: &Container) -> bool {
        let new_card = container.cardinality() as u64;
        self.heap_bytes += heap_cost(container);
        match self.map.insert(key, container.clone()) {
            Some(old) => {
                self.cardinality = self.cardinality - old.cardinality() as u64 + new_card;
                self.heap_bytes -= heap_cost(&old);
                false
            }
            None => {
                self.cardinality += new_card;
                true
            }
        }
    }

    #[allow(dead_code)]
    fn remove_key(&mut self, key: u128) -> u32 {
        match self.map.remove(&key) {
            Some(old) => {
                let card = old.cardinality() as u32;
                self.cardinality -= card as u64;
                self.heap_bytes -= heap_cost(&old);
                card
            }
            None => 0,
        }
    }
}

// ── ContainerStore ───────────────────────────────────────────────────────────

/// The `u128 → Container` map behind a [`RoaringBitmap`](crate::index::RoaringBitmap):
/// heap-backed for transient bitmaps, file-backed (mapped) for persistent ones.
/// See the module docs for why transient bitmaps must not use the mapped layout.
pub(crate) struct ContainerStore {
    backing: Backing,
}

enum Backing {
    Heap(HeapStore),
    Mapped(MappedStore),
}

impl ContainerStore {
    // ── Construction ─────────────────────────────────────────────────────────

    /// Create a new empty transient store on the heap.
    pub fn new_anon() -> Self {
        Self {
            backing: Backing::Heap(HeapStore::default()),
        }
    }

    /// Create a new persistent store, creating both files under `dir`.
    pub fn create(dir: &Path) -> io::Result<Self> {
        Ok(Self {
            backing: Backing::Mapped(MappedStore::create(dir)?),
        })
    }

    /// Open an existing persistent store from `dir`.
    pub fn open(dir: &Path) -> io::Result<Self> {
        Ok(Self {
            backing: Backing::Mapped(MappedStore::open(dir)?),
        })
    }

    // ── Metrics ──────────────────────────────────────────────────────────────

    /// Cached total bit count across all containers.
    pub fn cardinality(&self) -> usize {
        match &self.backing {
            Backing::Heap(h) => h.cardinality as usize,
            Backing::Mapped(m) => m.cardinality(),
        }
    }

    /// Number of live (occupied) entries.
    pub fn count(&self) -> usize {
        match &self.backing {
            Backing::Heap(h) => h.map.len(),
            Backing::Mapped(m) => m.count(),
        }
    }

    // ── Mutation ─────────────────────────────────────────────────────────────

    /// Insert or replace the container for `key`.
    ///
    /// Returns `true` if this was a new key (false if an existing one was updated).
    pub fn upsert(&mut self, key: u128, container: &Container) -> bool {
        match &mut self.backing {
            Backing::Heap(h) => h.upsert(key, container),
            Backing::Mapped(m) => m.upsert(key, container),
        }
    }

    /// Insert or replace the container for `key`, taking ownership (no copy
    /// on the heap backing).
    pub fn insert_owned(&mut self, key: u128, container: Container) {
        match &mut self.backing {
            Backing::Heap(h) => {
                let (card, cost) = (container.cardinality() as u64, heap_cost(&container));
                h.cardinality += card;
                h.heap_bytes += cost;
                if let Some(old) = h.map.insert(key, container) {
                    h.cardinality -= old.cardinality() as u64;
                    h.heap_bytes -= heap_cost(&old);
                }
            }
            Backing::Mapped(m) => {
                m.upsert(key, &container);
            }
        }
    }

    /// Remove the container for `key`. Returns the removed container's
    /// cardinality, or 0 if the key was not present.
    #[allow(dead_code)]
    pub fn remove_key(&mut self, key: u128) -> u32 {
        match &mut self.backing {
            Backing::Heap(h) => h.remove_key(key),
            Backing::Mapped(m) => m.remove_key(key),
        }
    }

    /// Change the container for `key` in place.
    ///
    /// `f` gets the container (a new empty array container if `key` is absent
    /// and `create` is true) and returns whether it changed anything. A changed
    /// container that ends up empty is removed. Returns `f`'s result, or
    /// `false` without calling `f` when `key` is absent and `create` is false.
    ///
    /// On the heap this touches only the one container; the mapped backing
    /// still writes a new copy, as [`upsert`](Self::upsert) does.
    pub fn modify(&mut self, key: u128, create: bool, f: impl FnOnce(&mut Container) -> bool) -> bool {
        match &mut self.backing {
            Backing::Heap(h) => {
                // (changed, old card, new card, old cost, new cost, now empty)
                let (changed, old_card, new_card, old_cost, new_cost, empty) = match h.map.get_mut(&key) {
                    Some(c) => {
                        let (old_card, old_cost) = (c.cardinality() as u64, heap_cost(c));
                        let changed = f(c);
                        (changed, old_card, c.cardinality() as u64, old_cost, heap_cost(c), c.is_empty())
                    }
                    None if create => {
                        let mut c = Container::new_array();
                        let changed = f(&mut c);
                        let (card, cost, empty) = (c.cardinality() as u64, heap_cost(&c), c.is_empty());
                        if changed && !empty {
                            h.map.insert(key, c);
                        }
                        (changed, 0, card, 0, if changed && !empty { cost } else { 0 }, empty)
                    }
                    None => return false,
                };
                h.heap_bytes = h.heap_bytes - old_cost + new_cost;
                if changed {
                    h.cardinality = h.cardinality - old_card + new_card;
                    if empty && h.map.remove(&key).is_some() {
                        h.heap_bytes -= new_cost;
                    }
                }
                changed
            }
            Backing::Mapped(m) => {
                let mut c = match m.get(key) {
                    Some(c) => c,
                    None if create => Container::new_array(),
                    None => return false,
                };
                let changed = f(&mut c);
                if changed {
                    if c.is_empty() {
                        m.remove_key(key);
                    } else {
                        m.upsert(key, &c);
                    }
                }
                changed
            }
        }
    }

    /// Remove all entries.
    pub fn clear(&mut self) {
        match &mut self.backing {
            Backing::Heap(h) => *h = HeapStore::default(),
            Backing::Mapped(m) => m.clear(),
        }
    }

    // ── Query ────────────────────────────────────────────────────────────────

    /// Return a copy of the container for `key`, or `None`.
    ///
    /// Prefer [`with_container`](Self::with_container) for reads: on the heap
    /// it borrows instead of copying.
    pub fn get(&self, key: u128) -> Option<Container> {
        match &self.backing {
            Backing::Heap(h) => h.map.get(&key).cloned(),
            Backing::Mapped(m) => m.get(key),
        }
    }

    /// Run `f` on the container for `key` without copying it (heap) or after
    /// deserialising it (mapped). `None` if `key` is absent.
    pub fn with_container<R>(&self, key: u128, f: impl FnOnce(&Container) -> R) -> Option<R> {
        match &self.backing {
            Backing::Heap(h) => h.map.get(&key).map(f),
            Backing::Mapped(m) => m.get(key).as_ref().map(f),
        }
    }

    /// True if the key has an entry.
    #[allow(dead_code)]
    pub fn contains_key(&self, key: u128) -> bool {
        match &self.backing {
            Backing::Heap(h) => h.map.contains_key(&key),
            Backing::Mapped(m) => m.contains_key(key),
        }
    }

    /// Return `(key, card)` for every live entry, sorted by key.
    pub fn sorted_key_cards(&self) -> Vec<(u128, u32)> {
        match &self.backing {
            Backing::Heap(h) => h.map.iter().map(|(&k, c)| (k, c.cardinality() as u32)).collect(),
            Backing::Mapped(m) => m.sorted_key_cards(),
        }
    }

    /// Return all live keys sorted.
    pub fn sorted_keys(&self) -> Vec<u128> {
        match &self.backing {
            Backing::Heap(h) => h.map.keys().copied().collect(),
            Backing::Mapped(m) => m.sorted_keys(),
        }
    }

    /// Return all `(key, container)` pairs, sorted by key.
    pub fn sorted_entries(&self) -> Vec<(u128, Container)> {
        match &self.backing {
            Backing::Heap(h) => h.map.iter().map(|(&k, c)| (k, c.clone())).collect(),
            Backing::Mapped(m) => m.sorted_entries(),
        }
    }

    /// Flush to disk (no-op for the heap backing).
    pub fn flush(&self) -> io::Result<()> {
        match &self.backing {
            Backing::Heap(_) => Ok(()),
            Backing::Mapped(m) => m.flush(),
        }
    }

    /// Heap bytes held: each container's buffer plus a fixed per-entry cost,
    /// kept current on every change, so this is O(1). Zero for the mapped
    /// backing, whose data lives in file pages the kernel can evict.
    pub fn heap_bytes(&self) -> usize {
        match &self.backing {
            Backing::Heap(h) => h.heap_bytes,
            Backing::Mapped(_) => 0,
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

#[inline]
fn align_up(offset: usize, align: usize) -> usize {
    (offset + align - 1) & !(align - 1)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::container::Container;
    use crate::index::container::array::ArrayContainer;

    fn make_array(vals: &[u16]) -> Container {
        Container::Array(ArrayContainer::from_sorted(vals.to_vec()))
    }

    #[test]
    fn insert_get_roundtrip() {
        let mut store = ContainerStore::new_anon();
        let c = make_array(&[1, 2, 3]);
        assert!(store.upsert(42, &c));
        let got = store.get(42).unwrap();
        assert_eq!(got.to_values(), vec![1, 2, 3]);
        assert_eq!(store.count(), 1);
        assert_eq!(store.cardinality(), 3);
    }

    #[test]
    fn update_existing_adjusts_cardinality() {
        let mut store = ContainerStore::new_anon();
        let c1 = make_array(&[1, 2, 3]);
        let c2 = make_array(&[1, 2, 3, 4, 5]);
        assert!(store.upsert(10, &c1));
        assert!(!store.upsert(10, &c2)); // update
        assert_eq!(store.cardinality(), 5);
        assert_eq!(store.count(), 1);
    }

    #[test]
    fn remove_adjusts_cardinality() {
        let mut store = ContainerStore::new_anon();
        store.upsert(1, &make_array(&[1, 2]));
        store.upsert(2, &make_array(&[3, 4, 5]));
        assert_eq!(store.cardinality(), 5);
        let removed_card = store.remove_key(1);
        assert_eq!(removed_card, 2);
        assert_eq!(store.cardinality(), 3);
        assert_eq!(store.count(), 1);
        assert!(store.get(1).is_none());
    }

    #[test]
    fn sorted_key_cards_order() {
        let mut store = ContainerStore::new_anon();
        store.upsert(300, &make_array(&[1]));
        store.upsert(100, &make_array(&[1, 2]));
        store.upsert(200, &make_array(&[1, 2, 3]));
        let kc = store.sorted_key_cards();
        assert_eq!(kc, vec![(100, 2), (200, 3), (300, 1)]);
    }

    #[test]
    fn rehash_preserves_entries() {
        let mut store = ContainerStore::new_anon();
        // Insert enough to trigger rehash (>70% of initial 16 slots = ~12 entries)
        for i in 0u128..20 {
            store.upsert(i * 0x1_0000, &make_array(&[i as u16]));
        }
        assert_eq!(store.count(), 20);
        // Verify all entries survived rehash
        for i in 0u128..20 {
            let c = store.get(i * 0x1_0000).unwrap();
            assert_eq!(c.to_values(), vec![i as u16]);
        }
    }

    #[test]
    fn clear_resets_store() {
        let mut store = ContainerStore::new_anon();
        store.upsert(1, &make_array(&[1, 2, 3]));
        store.clear();
        assert_eq!(store.count(), 0);
        assert_eq!(store.cardinality(), 0);
        assert!(store.get(1).is_none());
    }

    #[test]
    fn sorted_entries_correct() {
        let mut store = ContainerStore::new_anon();
        store.upsert(200, &make_array(&[5, 6]));
        store.upsert(100, &make_array(&[1, 2, 3]));
        let entries = store.sorted_entries();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].0, 100);
        assert_eq!(entries[1].0, 200);
    }

    #[test]
    fn transient_store_memory_is_the_bitmaps_size_not_its_history() {
        // 65,536 single-value changes to one container. The mapped layout
        // appends a copy per change (this used to cost 514 MiB of memory for an
        // 8 KB bitmap); the heap store changes the container in place.
        let mut store = ContainerStore::new_anon();
        for low in 0..=u16::MAX {
            assert!(store.modify(0, true, |c| c.insert(low)));
        }
        assert_eq!(store.cardinality(), 65_536);
        assert_eq!(store.count(), 1);
        let bytes = store.heap_bytes();
        assert!(bytes <= 16 * 1024, "one full container should hold ~8 KB, holds {bytes} bytes");
    }

    #[test]
    fn modify_keeps_cardinality_and_removes_emptied_containers() {
        let dir = tempfile::TempDir::new().unwrap();
        for mut store in [ContainerStore::new_anon(), ContainerStore::create(dir.path()).unwrap()] {
            assert!(store.modify(7, true, |c| c.insert(1)));
            assert!(store.modify(7, true, |c| c.insert(2)));
            assert!(!store.modify(7, true, |c| c.insert(2)), "re-inserting is no change");
            assert_eq!(store.cardinality(), 2);
            assert!(!store.modify(8, false, |c| c.insert(1)), "absent key without create is untouched");
            assert!(!store.contains_key(8));
            assert!(store.modify(7, false, |c| c.remove(1)));
            assert!(store.modify(7, false, |c| c.remove(2)));
            assert_eq!(store.cardinality(), 0);
            assert!(!store.contains_key(7), "an emptied container is removed");
            assert_eq!(store.with_container(7, |c| c.cardinality()), None);
        }
    }

    // ── Header / bounds validation on open ──────────────────────────────────

    fn seed(dir: &Path) {
        let mut store = ContainerStore::create(dir).unwrap();
        store.upsert(1, &make_array(&[1, 2, 3]));
        store.upsert(2, &make_array(&[4, 5]));
        store.flush().unwrap();
    }

    fn assert_invalid(dir: &Path) {
        match ContainerStore::open(dir) {
            Err(e) => assert_eq!(e.kind(), io::ErrorKind::InvalidData, "expected InvalidData, got: {e}"),
            Ok(_) => panic!("expected open() to reject corrupt store"),
        }
    }

    #[test]
    fn persistent_roundtrip_still_opens() {
        let dir = tempfile::TempDir::new().unwrap();
        seed(dir.path());
        let store = ContainerStore::open(dir.path()).unwrap();
        assert_eq!(store.count(), 2);
        assert_eq!(store.get(1).unwrap().to_values(), vec![1, 2, 3]);
    }

    #[test]
    fn open_rejects_bad_magic() {
        let dir = tempfile::TempDir::new().unwrap();
        seed(dir.path());
        let mut key = std::fs::read(dir.path().join("containers.keys")).unwrap();
        key[0] ^= 0xFF;
        std::fs::write(dir.path().join("containers.keys"), &key).unwrap();
        assert_invalid(dir.path());
    }

    #[test]
    fn open_rejects_unsupported_version() {
        let dir = tempfile::TempDir::new().unwrap();
        seed(dir.path());
        let mut key = std::fs::read(dir.path().join("containers.keys")).unwrap();
        key[8..12].copy_from_slice(&(VERSION + 1).to_le_bytes());
        std::fs::write(dir.path().join("containers.keys"), &key).unwrap();
        assert_invalid(dir.path());
    }

    #[test]
    fn open_rejects_truncated_header() {
        let dir = tempfile::TempDir::new().unwrap();
        seed(dir.path());
        std::fs::write(dir.path().join("containers.keys"), [0u8; HEADER_SIZE - 1]).unwrap();
        assert_invalid(dir.path());
    }

    #[test]
    fn open_rejects_zero_capacity() {
        let dir = tempfile::TempDir::new().unwrap();
        seed(dir.path());
        let mut key = std::fs::read(dir.path().join("containers.keys")).unwrap();
        key[16..24].copy_from_slice(&0u64.to_le_bytes());
        std::fs::write(dir.path().join("containers.keys"), &key).unwrap();
        assert_invalid(dir.path());
    }

    #[test]
    fn open_rejects_capacity_larger_than_key_file() {
        let dir = tempfile::TempDir::new().unwrap();
        seed(dir.path());
        let mut key = std::fs::read(dir.path().join("containers.keys")).unwrap();
        key[16..24].copy_from_slice(&(1u64 << 40).to_le_bytes());
        std::fs::write(dir.path().join("containers.keys"), &key).unwrap();
        assert_invalid(dir.path());
    }

    #[test]
    fn open_rejects_value_write_pos_past_value_file() {
        let dir = tempfile::TempDir::new().unwrap();
        seed(dir.path());
        let val_len = std::fs::metadata(dir.path().join("containers.vals")).unwrap().len();
        let mut key = std::fs::read(dir.path().join("containers.keys")).unwrap();
        key[48..56].copy_from_slice(&(val_len + 1).to_le_bytes());
        std::fs::write(dir.path().join("containers.keys"), &key).unwrap();
        assert_invalid(dir.path());
    }

    #[test]
    fn open_rejects_slot_blob_past_value_region() {
        let dir = tempfile::TempDir::new().unwrap();
        seed(dir.path());
        let mut key = std::fs::read(dir.path().join("containers.keys")).unwrap();
        let vwp = u64::from_le_bytes(key[48..56].try_into().unwrap());
        for i in 0..INITIAL_CAPACITY {
            let b = slot_byte_offset(i);
            if key[b] == STATE_OCCUPIED {
                key[b + 24..b + 32].copy_from_slice(&(vwp + 1).to_le_bytes());
                break;
            }
        }
        std::fs::write(dir.path().join("containers.keys"), &key).unwrap();
        assert_invalid(dir.path());
    }
}
