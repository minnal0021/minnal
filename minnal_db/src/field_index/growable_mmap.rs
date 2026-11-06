//! A memory-mapped byte region that can grow, used by every mapped store in the
//! field index: [`BlobStore`](crate::field_index::blob_store::BlobStore), the
//! container store's mapped backing and [`RowMap`](crate::field_index::RowMap).

use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;

use memmap2::MmapMut;

/// Size of the anonymous map [`GrowableMmap::open_file`] returns for an empty
/// file, which has nothing to map. Callers check what they read from it.
const EMPTY_FILE_MAP_SIZE: usize = 832;

/// A byte region backed by an anonymous map or a file, grown by doubling.
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
            MmapMut::map_anon(EMPTY_FILE_MAP_SIZE)?
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
    pub(crate) fn truncate(&mut self, new_size: usize) -> io::Result<()> {
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
