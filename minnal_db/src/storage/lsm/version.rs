//! A copy of a key's version: which write made it, and how many times the
//! engine has rewritten it since.
//!
//! Every user write (put, delete, merge, and the logged and no-WAL writes) draws
//! a fresh **user seq** from the global counter (`Database::next_seq`) and is at
//! system seq 0. A **system action** — value-log GC relocating a record, TTL
//! expiry deleting one — makes no new user-visible write: it keeps the user seq
//! of the copy it replaces and raises the **system seq** by one. Versions
//! compare by user seq, then system seq, so a system action's copy is newer than
//! the copy it replaced and older than every user write that comes after it.
//! That lets GC and TTL act under the bucket lock alone: a user write that drew
//! its seq before them and is still waiting to apply is newer, whenever it lands
//! (`docs/Layer-Order-Design.md`).

/// A key copy's version; see the module docs. The derived order is the
/// version order: user seq first, then system seq.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Version {
    /// The global write sequence of the user write that made this copy.
    pub user: u64,
    /// How many system actions (GC relocations, TTL expiry) have rewritten
    /// that write since.
    pub system: u32,
}

impl Version {
    /// A user write's version.
    pub const fn user(seq: u64) -> Self {
        Self { user: seq, system: 0 }
    }

    /// The version a system action gives the copy it writes in place of this
    /// one, or `None` once the system seq is exhausted (about four billion
    /// rewrites of one unchanged write).
    pub fn next_system(self) -> Option<Self> {
        Some(Self {
            user: self.user,
            system: self.system.checked_add(1)?,
        })
    }
}

impl From<u64> for Version {
    fn from(seq: u64) -> Self {
        Self::user(seq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions_order_by_user_seq_then_system_seq() {
        let write = Version::user(10);
        let moved = write.next_system().unwrap();
        assert!(moved > write, "a system action's copy is newer than what it replaced");
        assert!(Version::user(11) > moved, "and older than the next user write");
        assert!(Version { user: 11, system: 0 } > Version { user: 10, system: u32::MAX });
        assert_eq!(Version { user: 1, system: u32::MAX }.next_system(), None);
    }
}
