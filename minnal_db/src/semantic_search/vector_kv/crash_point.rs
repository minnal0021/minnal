//! Test-only crash injection for the crash audit (`Dynamic-RaBitQ-Index-Design.md` → M0-2).
//!
//! [`crash_point!`] sits before every vector-index write in `vector_kv`. A test
//! arms a countdown of N writes; the next check after N fails with a simulated
//! crash, and so does every check after it, as if the process had died. The
//! test then drops the database without shutting it down (losing every unflushed
//! no-WAL write, as a real crash does), reopens it and checks what survived.
//! Thread-local, so it only affects the (current-thread) test that armed it.

use std::cell::Cell;

thread_local! {
    static REMAINING: Cell<Option<usize>> = const { Cell::new(None) };
}

/// Let `writes` more writes through, then fail every later one.
pub(crate) fn arm(writes: usize) {
    REMAINING.with(|r| r.set(Some(writes)));
}

/// Stop injecting; returns whether the crash fired.
pub(crate) fn disarm() -> bool {
    REMAINING.with(|r| r.replace(None) == Some(0))
}

pub(crate) fn check() -> Result<(), crate::KVError> {
    REMAINING.with(|r| match r.get() {
        Some(0) => Err(crate::KVError::Io(std::io::Error::other("simulated crash"))),
        Some(n) => {
            r.set(Some(n - 1));
            Ok(())
        }
        None => Ok(()),
    })
}
