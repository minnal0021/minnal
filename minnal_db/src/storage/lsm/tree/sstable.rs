//! SSTable files, Level 0 and Level 1 alike: the on-disk format (header, CRC-framed
//! entries), the buffered frame reader, and reading entries and lookups from a file.

use super::*;

/// The header bytes written at offset 0 of a non-empty SSTable file.
pub(super) fn sstable_header_bytes() -> [u8; SSTABLE_DATA_START as usize] {
    let mut header = [0u8; SSTABLE_DATA_START as usize];
    header[0..4].copy_from_slice(&SSTABLE_MAGIC);
    header[4..8].copy_from_slice(&SSTABLE_FORMAT_VERSION.to_le_bytes());
    header
}

/// Verify that `file` begins with a header this build can read.
///
/// Returns [`LSMError::Corruption`] for a truncated header, a bad magic, or a
/// format version this build does not understand.
pub(super) fn validate_sstable_header(file: &File, path: &Path) -> Result<()> {
    let mut header = [0u8; SSTABLE_DATA_START as usize];
    file.read_exact_at(&mut header, 0)
        .map_err(|e| LSMError::Corruption(format!("SSTable {} has no readable header: {e}", path.display())))?;
    if header[0..4] != SSTABLE_MAGIC {
        return Err(LSMError::Corruption(format!(
            "SSTable {} is not a recognised SSTable (bad magic {:02x?}). A database written before SSTable \
             headers existed used a narrower write sequence and cannot be read by this build; recreate it.",
            path.display(),
            &header[0..4]
        )));
    }
    let version = u32::from_le_bytes(header[4..8].try_into().unwrap());
    if version != SSTABLE_FORMAT_VERSION {
        return Err(LSMError::Corruption(format!(
            "SSTable {} has format version {version}, but this build reads version {SSTABLE_FORMAT_VERSION}",
            path.display()
        )));
    }
    Ok(())
}

/// Open an SSTable for a head-of-file scan, **validating its header**, and leave
/// the read cursor at [`SSTABLE_DATA_START`] so sequential readers can keep using
/// `read_exact` unchanged.
///
/// This is the *first-touch* opener: use it where the file has not been vetted
/// yet — the L1 files and on-disk L0 files inspected during [`LSMTree::open`].
/// Files already in the Level-0 registry are opened with
/// [`open_registered_l0_for_scan`] instead; see it for why re-validating is
/// redundant there.
pub(super) fn open_sstable_for_scan(path: &Path) -> Result<File> {
    let mut file = File::open(path)?;
    validate_sstable_header(&file, path)?;
    file.seek(SeekFrom::Start(SSTABLE_DATA_START))?;
    Ok(file)
}

/// Reject a frame whose declared payload length cannot fit in the file holding it,
/// **before** that length is used to allocate.
///
/// The 4-byte length prefix sits *outside* the entry CRC — which covers only the
/// body — so a flipped bit there is undetectable until after the payload has been
/// read, and allocating for it first would let a corrupt length request up to
/// 4 GiB from a file of a few hundred bytes. A frame can never be longer than its
/// file, so that is the bound (`valid_scan_start` applies the same bound to a
/// sparse-index hint).
///
/// Zero is rejected too: a body is `crc32(4) + rkyv payload`, so it is never empty.
#[inline]
pub(super) fn check_frame_len(size: u64, file_len: u64) -> Result<()> {
    if size == 0 || size > file_len {
        return Err(LSMError::Corruption(format!(
            "SSTable frame declares a {size}-byte payload in a {file_len}-byte file"
        )));
    }
    Ok(())
}

impl<'a> FrameReader<'a> {
    pub(super) const MIN_READ: usize = 8 * 1024;
    pub(super) const MAX_READ: usize = 256 * 1024;

    pub(super) fn new(file: &'a File, offset: u64) -> Self {
        Self {
            file,
            buf: Vec::new(),
            pos: 0,
            len: 0,
            next_offset: offset,
            read_size: Self::MIN_READ,
            bytes_read: 0,
        }
    }

    /// File offset of the next frame.
    pub(super) fn offset(&self) -> u64 {
        self.next_offset - (self.len - self.pos) as u64
    }

    /// Continue reading at `offset`, a frame boundary at or past
    /// [`offset`](Self::offset). Skips within the buffer when the target is
    /// already in it; otherwise drops the buffer and restarts small read-ahead.
    pub(super) fn seek(&mut self, offset: u64) {
        let here = self.offset();
        debug_assert!(offset >= here, "FrameReader only seeks forward");
        if offset >= here && offset < self.next_offset {
            self.pos += (offset - here) as usize;
            return;
        }
        self.pos = 0;
        self.len = 0;
        self.next_offset = offset;
        self.read_size = Self::MIN_READ;
    }

    /// Buffer at least `n` unconsumed bytes, reading more only if needed. Returns
    /// the number of unconsumed bytes, which is below `n` only at end of file.
    pub(super) fn fill(&mut self, n: usize) -> Result<usize> {
        if self.len - self.pos >= n {
            return Ok(self.len - self.pos);
        }
        self.buf.copy_within(self.pos..self.len, 0);
        self.len -= self.pos;
        self.pos = 0;
        let want = self.len + n.max(self.read_size);
        if self.buf.len() < want {
            self.buf.resize(want, 0);
        }
        self.read_size = (self.read_size * 2).min(Self::MAX_READ);
        while self.len < n {
            match self.file.read_at(&mut self.buf[self.len..], self.next_offset) {
                Ok(0) => break,
                Ok(read) => {
                    self.len += read;
                    self.next_offset += read as u64;
                    self.bytes_read += read as u64;
                }
                Err(err) if err.kind() == std::io::ErrorKind::Interrupted => {}
                Err(err) => return Err(LSMError::Io(err)),
            }
        }
        Ok(self.len - self.pos)
    }

    /// The next frame's body (`crc32 ‖ payload`), or `None` at a clean EOF.
    pub(super) fn next_frame(&mut self, file_len: u64) -> Result<Option<&[u8]>> {
        let unexpected_eof = || LSMError::Io(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "Unexpected EOF"));
        match self.fill(4)? {
            0 => return Ok(None),
            1..=3 => return Err(unexpected_eof()),
            _ => {}
        }
        let size = u32::from_le_bytes(self.buf[self.pos..self.pos + 4].try_into().unwrap()) as usize;
        check_frame_len(size as u64, file_len)?;
        self.pos += 4;
        if self.fill(size)? < size {
            return Err(unexpected_eof());
        }
        let body = &self.buf[self.pos..self.pos + size];
        self.pos += size;
        Ok(Some(body))
    }
}

/// Open a **registered** Level-0 file — one reachable from `level0_files` — for a
/// positional reader, skipping header validation.
///
/// # Why skipping is sound
///
/// A file's format version is fixed when it is written and cannot change
/// afterwards: L0 files are write-once (never mutated in place) and are unlinked
/// only once `readers() == 0`. So the header is a property to check when a file
/// *enters* the registry, not on every read. Both entry points establish it:
/// [`LSMTree::open`] validates every on-disk L0 file it loads
/// (`load_existing_level0_files` → `level0_file_summary`), and
/// [`register_level0_file`](LSMTree::register_level0_file) only ever receives a
/// file this process just wrote with a header. Re-validating on each read would
/// cost a `pread` per L0 file on every point get, since `search_level0_files`
/// opens every L0 file in the bucket its summary cannot rule out.
///
/// **Precondition:** `path` comes from an `L0FileEntry`. Opening an unvetted file
/// with this would parse a foreign layout as entries — use
/// [`open_sstable_for_scan`] there.
pub(super) fn open_registered_l0(path: &Path) -> Result<File> {
    Ok(File::open(path)?)
}

/// [`open_registered_l0`] for a sequential reader: leaves the cursor at
/// [`SSTABLE_DATA_START`]. Same precondition — the caller's path must come from
/// an `L0FileEntry`.
pub(super) fn open_registered_l0_for_scan(path: &Path) -> Result<File> {
    let mut file = open_registered_l0(path)?;
    file.seek(SeekFrom::Start(SSTABLE_DATA_START))?;
    Ok(file)
}

/// On disk, each SSTable entry is framed as:
///
/// ```text
/// [u32 payload_len][u32 crc32(body)][body = rkyv-serialized SStableEntry]
/// ```
///
/// `payload_len` counts the CRC word plus the rkyv `body`. The CRC guards
/// against silent on-disk corruption (bit rot, torn writes): it is verified
/// before the body is handed to the *unchecked* rkyv accessor, which assumes
/// well-formed input and would otherwise interpret garbage as a valid entry.
pub(super) fn encode_sstable_entry(entry: &SStableEntry) -> Result<Vec<u8>> {
    let body = rkyv::to_bytes::<rkyv::rancor::Error>(entry).map_err(|e| LSMError::Serialization(format!("Failed to serialize entry: {}", e)))?;
    let crc = crc32fast::hash(&body);
    let mut payload = Vec::with_capacity(4 + body.len());
    payload.extend_from_slice(&crc.to_le_bytes());
    payload.extend_from_slice(&body);
    Ok(payload)
}

/// Verify a framed SSTable entry payload and return its rkyv `body` slice.
///
/// `framed` is the `payload_len` bytes that follow the length prefix:
/// `[u32 crc32(body)][body]`. Returns [`LSMError::Corruption`] if the payload
/// is truncated or the CRC does not match, so a corrupt entry never reaches
/// `rkyv::access_unchecked`.
pub(super) fn verify_sstable_payload(framed: &[u8]) -> Result<&[u8]> {
    if framed.len() < 4 {
        return Err(LSMError::Corruption(format!("SSTable entry payload too small: {} bytes", framed.len())));
    }
    let stored = u32::from_le_bytes(framed[0..4].try_into().unwrap());
    let body = &framed[4..];
    let actual = crc32fast::hash(body);
    if actual != stored {
        return Err(LSMError::Corruption(format!(
            "SSTable entry CRC mismatch: stored {:#010x}, computed {:#010x}",
            stored, actual
        )));
    }
    Ok(body)
}

impl LSMTree {
    pub(super) fn read_exact_at(file: &File, buf: &mut [u8], offset: &mut u64) -> Result<bool> {
        let mut read_total = 0usize;
        while read_total < buf.len() {
            match file.read_at(&mut buf[read_total..], *offset + read_total as u64) {
                Ok(0) => {
                    if read_total == 0 {
                        return Ok(false);
                    }
                    return Err(LSMError::Io(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "Unexpected EOF")));
                }
                Ok(n) => {
                    read_total += n;
                }
                Err(err) => return Err(LSMError::Io(err)),
            }
        }
        *offset += buf.len() as u64;
        Ok(true)
    }

    /// Scan an SSTable file for `key`, starting at `start_offset` (a sparse-index
    /// hint, or 0 for the whole file).
    ///
    /// The result is tri-state, and the distinction between
    /// [`SsLookup::Deleted`] and [`SsLookup::Missing`] is load-bearing: a
    /// tombstone found in a newer layer (e.g. an L0 file) must *shadow* a live
    /// value in an older layer (L1). Collapsing both into `None` would let `get`
    /// fall through an L0 tombstone to a stale L1 value and resurrect a deleted key.
    ///
    /// The hint is validated against the file first: if it does not point at a
    /// complete, CRC-valid frame whose key is `<= key` — e.g. a concurrent
    /// compaction swapped the file under the in-memory index, leaving the offset
    /// stale — the scan restarts from 0. Entries are key-sorted, so beginning at
    /// *any* valid frame boundary with key `<= key` still finds `key` if present;
    /// this makes a stale hint a performance issue, never a correctness one.
    pub(super) fn lookup_in_sstable_file_from(&self, file: &File, key: &[u8], start_offset: u64) -> Result<SsLookup> {
        Ok(self.lookup_in_sstable_file_hinted(file, key, start_offset)?.0)
    }

    /// [`lookup_in_sstable_file_from`](Self::lookup_in_sstable_file_from), also
    /// reporting whether a non-zero `start_offset` was rejected and the scan fell
    /// back to the head of the file.
    ///
    /// A rejection is never a wrong answer, only a slow one, which is exactly why
    /// it is counted (`sparse_hint_rejects`): it is otherwise invisible. For L1 it
    /// happens legitimately while a compaction swaps the file under the index; for
    /// an immutable L0 file it means the index's offsets are wrong, and the L0
    /// caller asserts on it in debug builds.
    pub(super) fn lookup_in_sstable_file_hinted(&self, file: &File, key: &[u8], start_offset: u64) -> Result<(SsLookup, bool)> {
        // Also the bound for `check_frame_len` below: a frame can never be longer
        // than its file.
        let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
        let hint_rejected = start_offset != 0 && !Self::valid_scan_start(file, start_offset, file_len, key);
        if hint_rejected && let Some(m) = self.metrics() {
            Metrics::bump(&m.sparse_hint_rejects);
        }
        let offset = if start_offset != 0 && !hint_rejected {
            start_offset
        } else {
            SSTABLE_DATA_START
        };
        Ok((Self::scan_sstable_file_for(file, key, offset, file_len)?, hint_rejected))
    }

    /// The forward scan behind [`lookup_in_sstable_file_hinted`](Self::lookup_in_sstable_file_hinted),
    /// from `offset` (a validated frame boundary) until `key` is found or passed.
    pub(super) fn scan_sstable_file_for(file: &File, key: &[u8], mut offset: u64, file_len: u64) -> Result<SsLookup> {
        loop {
            let mut size_buf = [0u8; 4];
            if !Self::read_exact_at(file, &mut size_buf, &mut offset)? {
                break;
            }

            let size = u32::from_le_bytes(size_buf);
            check_frame_len(u64::from(size), file_len)?;
            let mut entry_bytes = vec![0u8; size as usize];
            Self::read_exact_at(file, &mut entry_bytes, &mut offset)?;

            let archived = unsafe { rkyv::access_unchecked::<ArchivedSStableEntry>(verify_sstable_payload(&entry_bytes)?) };

            if archived.key.as_slice() == key {
                if archived.tombstone {
                    return Ok(SsLookup::Deleted(archived.seq.to_native()));
                }
                return Ok(SsLookup::Found(archived.value.to_native(), archived.seq.to_native()));
            }

            if archived.key.as_slice() > key {
                break;
            }
        }

        Ok(SsLookup::Missing)
    }

    /// True if `offset` begins a complete, CRC-valid frame whose key is `<= key`
    /// — a safe place to start a forward scan. The size is bounds-checked against
    /// `file_len` so a stale offset cannot trigger a huge allocation, and the CRC
    /// catches an offset that lands mid-frame or in a different (swapped) file.
    pub(super) fn valid_scan_start(file: &File, offset: u64, file_len: u64, key: &[u8]) -> bool {
        // Saturating arithmetic: a stale hint may be an arbitrarily large offset,
        // and `offset + 4` must not overflow (it would panic in debug).
        if offset.saturating_add(4) > file_len {
            return false;
        }
        let mut size_buf = [0u8; 4];
        if file.read_at(&mut size_buf, offset).is_err() {
            return false;
        }
        let size = u32::from_le_bytes(size_buf) as u64;
        if size == 0 || offset.saturating_add(4).saturating_add(size) > file_len {
            return false;
        }
        let mut buf = vec![0u8; size as usize];
        if file.read_at(&mut buf, offset + 4).is_err() {
            return false;
        }
        match verify_sstable_payload(&buf) {
            Ok(body) => {
                let archived = unsafe { rkyv::access_unchecked::<ArchivedSStableEntry>(body) };
                archived.key.as_slice() <= key
            }
            Err(_) => false,
        }
    }

    pub(super) fn read_sstable_entries_from_file(&self, file: &File) -> Result<Vec<SStableEntry>> {
        let mut entries = Vec::new();
        // Bound for `check_frame_len`: a frame can never be longer than its file.
        let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
        let mut offset = SSTABLE_DATA_START;

        loop {
            let mut size_buf = [0u8; 4];
            if !Self::read_exact_at(file, &mut size_buf, &mut offset)? {
                break;
            }

            let size = u32::from_le_bytes(size_buf);
            check_frame_len(u64::from(size), file_len)?;
            let mut entry_bytes = vec![0u8; size as usize];
            Self::read_exact_at(file, &mut entry_bytes, &mut offset)?;

            let archived = unsafe { rkyv::access_unchecked::<ArchivedSStableEntry>(verify_sstable_payload(&entry_bytes)?) };
            let key = archived.key.to_vec();
            let entry = SStableEntry {
                key_prefix: key_prefix_of(&key),
                key,
                value: archived.value.to_native(),
                tombstone: archived.tombstone,
                seq: archived.seq.to_native(),
            };

            entries.push(entry);
        }

        Ok(entries)
    }
    /// Read all entries from an existing SSTable file
    pub(super) fn read_sstable_entries(&self, bucket: u32) -> Result<Vec<SStableEntry>> {
        let file = self.sstable_files[bucket as usize].read().clone();
        self.read_sstable_entries_from_file(&file)
    }

    /// Read SSTable entries from a specific file path.
    ///
    /// Called by the L0→L1 merge with a registered `L0FileEntry`'s path, so it uses
    /// the registered-file opener (see [`open_registered_l0`]).
    pub(super) fn read_sstable_entries_from_path(&self, path: &Path) -> Result<Vec<SStableEntry>> {
        if !path.exists() {
            return Ok(Vec::new());
        }

        let mut entries = Vec::new();
        let mut file = open_registered_l0_for_scan(path)?;
        // Bound for `check_frame_len`: a frame can never be longer than its file.
        let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);

        loop {
            let mut size_buf = [0u8; 4];
            match file.read_exact(&mut size_buf) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(LSMError::Io(e)),
            }

            let size = u32::from_le_bytes(size_buf);
            check_frame_len(u64::from(size), file_len)?;
            let mut entry_bytes = vec![0u8; size as usize];
            file.read_exact(&mut entry_bytes)?;

            let archived = unsafe { rkyv::access_unchecked::<ArchivedSStableEntry>(verify_sstable_payload(&entry_bytes)?) };
            let key = archived.key.to_vec();
            let entry = SStableEntry {
                key_prefix: key_prefix_of(&key),
                key,
                value: archived.value.to_native(),
                tombstone: archived.tombstone,
                seq: archived.seq.to_native(),
            };

            entries.push(entry);
        }

        Ok(entries)
    }

    /// Tri-state lookup of a key in a specific SSTable file (by path).
    pub(super) fn lookup_in_sstable_file_path(&self, path: std::path::PathBuf, key: &[u8], start_offset: u64) -> Result<SsLookup> {
        // The hot path: `search_level0_files` calls this for every L0 file its
        // summary cannot rule out. Reads below are positional, so no cursor seek,
        // and the file is registered, so no header re-validation.
        let file = open_registered_l0(&path)?;
        let (hit, hint_rejected) = self.lookup_in_sstable_file_hinted(&file, key, start_offset)?;
        // An L0 file is never rewritten, so its sparse-index hint has no
        // legitimate way to go stale: a rejection means the index's offsets
        // disagree with the file. Release builds survive it with a full scan (and
        // count it in `sparse_hint_rejects`); debug builds fail, so every test
        // that reaches an L0 lookup guards the offsets, not just the dedicated one.
        debug_assert!(
            !hint_rejected,
            "L0 sparse-index hint {start_offset} is not a valid scan start in {}",
            path.display()
        );
        Ok(hit)
    }

    /// Load metadata from an existing SSTable file by scanning it, and build the
    /// L1 Bloom filter from the same pass (no extra I/O). Returns `None` for the
    /// bloom when the file has no entries.
    pub(super) fn load_metadata_from_file(path: &Path) -> Result<LoadedSstable> {
        let mut file = open_sstable_for_scan(path)?;
        // Bound for `check_frame_len`: a frame can never be longer than its file.
        let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);

        let mut min_key: Option<Vec<u8>> = None;
        let mut max_key: Option<Vec<u8>> = None;
        let mut entry_count = 0u64;
        let mut data_size = 0u64;
        let mut keys: Vec<Vec<u8>> = Vec::new();
        let mut index = SparseIndex::new();
        let mut max_seq: Option<u64> = None;

        loop {
            // `data_size` counts entry bytes only; the frame's absolute file
            // offset is that plus the header the scan started after.
            let frame_offset = SSTABLE_DATA_START + data_size;

            let mut size_buf = [0u8; 4];
            match file.read_exact(&mut size_buf) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(LSMError::Io(e)),
            }

            let size = u32::from_le_bytes(size_buf);
            check_frame_len(u64::from(size), file_len)?;
            let mut entry_bytes = vec![0u8; size as usize];
            file.read_exact(&mut entry_bytes)?;

            let archived = unsafe { rkyv::access_unchecked::<ArchivedSStableEntry>(verify_sstable_payload(&entry_bytes)?) };
            let key = archived.key.to_vec();

            let seq = archived.seq.to_native();
            if max_seq.is_none_or(|m| seq_newer(seq, m)) {
                max_seq = Some(seq);
            }

            if min_key.is_none() {
                min_key = Some(key.clone());
            }
            max_key = Some(key.clone());
            if entry_count.is_multiple_of(SparseIndex::SAMPLE_INTERVAL) {
                index.push(key.clone(), frame_offset);
            }
            keys.push(key);

            entry_count += 1;
            data_size += 4 + size as u64;
        }

        let bloom = (!keys.is_empty()).then(|| BloomFilter::build(keys.iter().map(|k| k.as_slice()), keys.len()));
        let index = (!index.is_empty()).then_some(index);

        let min_key = min_key.unwrap_or_default();
        let max_key = max_key.unwrap_or_default();

        Ok((
            SStableMetadata {
                min_key,
                max_key,
                entry_count,
                max_seq,
            },
            bloom,
            index,
            max_seq,
        ))
    }
}
