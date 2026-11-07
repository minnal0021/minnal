//! Commit rounds: concurrent callers share one fsync of a log.
//!
//! Both logs use this. The main WAL calls [`CommitRounds::commit`] once per
//! write; the vector index WAL calls it from `sync()`, once per batch. A caller
//! that needs everything below `end` durable either finds it already is, waits
//! for the round that is running, or — if none is — leads one: it runs the
//! log's own `round` step (snapshot the tail and active file under the log's
//! lock, fsync outside it) and publishes the position that fsync covered. One
//! fsync then covers every caller whose bytes were written before it started,
//! and nobody returns before an fsync that started after its bytes were written.
//!
//! A failed round **poisons** the rounds, and with them the log: the kernel may
//! have dropped the dirty pages, and a later fsync that succeeds says nothing
//! about them. Every caller waiting on that round, and every later one, fails
//! until the database is reopened. Positions are each log's own; they only
//! need to grow. Design: `minnal_db/docs/Main-WAL-Rework-Design.md` §6, §16.

use parking_lot::{Condvar, Mutex};
use std::time::Duration;

#[derive(Debug, Default)]
struct State {
    /// Every byte below this position is fsynced.
    durable: u64,
    /// A leader is fsyncing; others wait for it to finish.
    running: bool,
    /// Callers waiting for a round (not counting a running leader).
    waiting: usize,
}

/// See the module docs.
#[derive(Debug, Default)]
pub(crate) struct CommitRounds {
    state: Mutex<State>,
    /// Signalled when a round ends (or the rounds are poisoned).
    round_done: Condvar,
    /// Set by the first failed round; see the module docs.
    poisoned: Mutex<Option<String>>,
    /// Test hook: while set, callers entering [`commit`](Self::commit) wait
    /// there; the count is how many are waiting.
    #[cfg(test)]
    frozen: Mutex<(bool, usize)>,
    #[cfg(test)]
    thawed: Condvar,
    /// Test hook: runs in a failed round's leader after it has woken the
    /// waiters and released the state lock, before it returns.
    #[cfg(test)]
    after_failed_round: Mutex<Option<TestHook>>,
}

/// A test hook closure (`Debug` so `CommitRounds` can derive it).
#[cfg(test)]
struct TestHook(Box<dyn Fn() + Send + Sync>);

#[cfg(test)]
impl std::fmt::Debug for TestHook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("TestHook")
    }
}

impl CommitRounds {
    /// Rounds whose log is durable below `durable` (what survived a reopen).
    pub(crate) fn new(durable: u64) -> Self {
        let rounds = Self::default();
        rounds.state.lock().durable = durable;
        rounds
    }

    /// Every byte below this position is fsynced.
    pub(crate) fn durable(&self) -> u64 {
        self.state.lock().durable
    }

    /// Record that everything below `position` is durable (a sealed segment,
    /// the state found at open).
    pub(crate) fn advance(&self, position: u64) {
        let mut st = self.state.lock();
        st.durable = st.durable.max(position);
    }

    /// `Err(reason)` once the rounds are poisoned.
    pub(crate) fn check_poisoned(&self) -> Result<(), String> {
        match self.poisoned.lock().as_ref() {
            None => Ok(()),
            Some(why) => Err(why.clone()),
        }
    }

    /// Stop the log: every waiting and later commit fails until reopen.
    /// Returns the reason to report (the first one recorded).
    pub(crate) fn poison(&self, why: String) -> String {
        let why = self.poisoned.lock().get_or_insert(why).clone();
        let _st = self.state.lock();
        self.round_done.notify_all();
        why
    }

    /// The reason the rounds are poisoned, if they are.
    pub(crate) fn poisoned(&self) -> Option<String> {
        self.poisoned.lock().clone()
    }

    /// Wait until every byte below `end` is durable, leading a round if none
    /// is running. `round` is the log's step: take its tail and active file
    /// together under its own lock, fsync the file outside every lock, and
    /// return the position the fsync covered (at least `end` when the caller's
    /// bytes are in it). A round's error poisons the rounds.
    ///
    /// `delay`: a leader with others already waiting first sleeps this long,
    /// so more callers join its round. A lone caller never waits.
    ///
    /// Returns `Ok(true)` if this caller led a round (issued an fsync).
    pub(crate) fn commit(&self, end: u64, delay: Duration, mut round: impl FnMut() -> Result<u64, String>) -> Result<bool, String> {
        #[cfg(test)]
        self.wait_while_frozen();
        let mut st = self.state.lock();
        loop {
            self.check_poisoned()?;
            if st.durable >= end {
                return Ok(false);
            }
            if st.running {
                st.waiting += 1;
                self.round_done.wait(&mut st);
                st.waiting -= 1;
                continue;
            }
            st.running = true;
            let others_waiting = st.waiting > 0;
            drop(st);
            if others_waiting && !delay.is_zero() {
                std::thread::sleep(delay);
            }
            // A failed round is recorded as poison BEFORE `running` is cleared: a
            // waiter woken by this round would otherwise find no poison, lead the
            // next round, and have its fsync succeed over pages the kernel
            // dropped. A round that succeeded is not trusted either if the log
            // was poisoned while it ran (a failed rotation fsync of the file it
            // synced).
            let covered = round()
                .and_then(|position| self.check_poisoned().map(|()| position))
                .map_err(|why| self.poisoned.lock().get_or_insert(why).clone());
            st = self.state.lock();
            st.running = false;
            self.round_done.notify_all();
            match covered {
                Ok(position) => st.durable = st.durable.max(position),
                Err(why) => {
                    drop(st);
                    #[cfg(test)]
                    if let Some(hook) = self.after_failed_round.lock().as_ref() {
                        (hook.0)();
                    }
                    return Err(why);
                }
            }
            if st.durable >= end {
                return Ok(true);
            }
        }
    }

    /// Test hook: while `on`, every caller reaching [`commit`](Self::commit)
    /// waits there, its bytes written but not yet fsynced — the state a power
    /// loss would catch it in. Turning it off releases them.
    #[cfg(test)]
    pub(crate) fn freeze(&self, on: bool) {
        self.frozen.lock().0 = on;
        if !on {
            self.thawed.notify_all();
        }
    }

    /// Test hook: how many callers are waiting in a frozen commit.
    #[cfg(test)]
    pub(crate) fn frozen_callers(&self) -> usize {
        self.frozen.lock().1
    }

    #[cfg(test)]
    fn wait_while_frozen(&self) {
        let mut f = self.frozen.lock();
        if f.0 {
            f.1 += 1;
            while f.0 {
                self.thawed.wait(&mut f);
            }
            f.1 -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Concurrent callers share rounds, and each returns only once a round
    /// covering its end has finished.
    #[test]
    fn concurrent_callers_share_rounds() {
        let rounds = Arc::new(CommitRounds::new(0));
        let written = Arc::new(AtomicU64::new(0));
        let fsyncs = Arc::new(AtomicU64::new(0));
        std::thread::scope(|s| {
            for _ in 0..16 {
                let (rounds, written, fsyncs) = (&rounds, &written, &fsyncs);
                s.spawn(move || {
                    for _ in 0..50 {
                        let end = written.fetch_add(1, Ordering::SeqCst) + 1;
                        rounds
                            .commit(end, Duration::ZERO, || {
                                let covered = written.load(Ordering::SeqCst);
                                std::thread::sleep(Duration::from_micros(200));
                                fsyncs.fetch_add(1, Ordering::SeqCst);
                                Ok(covered)
                            })
                            .unwrap();
                        assert!(rounds.durable() >= end);
                    }
                });
            }
        });
        let n = fsyncs.load(Ordering::SeqCst);
        assert!(n < 800, "16 concurrent callers never shared a round ({n} fsyncs for 800 commits)");
    }

    /// A caller waiting on a round that fails gets the error; it never leads a
    /// round of its own, whose fsync could succeed over dropped pages. The
    /// failed leader is held after waking the waiters (where the poison used
    /// to be set) until the waiter has finished, so a wrong order always shows.
    #[test]
    fn a_caller_waiting_on_a_failed_round_never_leads_another() {
        let rounds = Arc::new(CommitRounds::new(0));
        let follower_done = Arc::new((Mutex::new(false), Condvar::new()));
        {
            let follower_done = Arc::clone(&follower_done);
            *rounds.after_failed_round.lock() = Some(TestHook(Box::new(move || {
                let (done, cv) = &*follower_done;
                let mut done = done.lock();
                if !*done {
                    cv.wait_for(&mut done, Duration::from_secs(5));
                }
            })));
        }
        let led_after_failure = AtomicU64::new(0);
        std::thread::scope(|s| {
            let leader = s.spawn(|| {
                rounds.commit(1, Duration::ZERO, || {
                    // Hold the round until the follower is waiting on it.
                    while rounds.state.lock().waiting == 0 {
                        std::thread::yield_now();
                    }
                    Err("disk".into())
                })
            });
            while !rounds.state.lock().running {
                std::thread::yield_now();
            }
            let follower = rounds.commit(2, Duration::ZERO, || {
                led_after_failure.fetch_add(1, Ordering::SeqCst);
                Ok(2)
            });
            {
                let (done, cv) = &*follower_done;
                *done.lock() = true;
                cv.notify_all();
            }
            assert_eq!(follower, Err("disk".into()));
            assert_eq!(leader.join().unwrap(), Err("disk".into()));
        });
        assert_eq!(led_after_failure.load(Ordering::SeqCst), 0, "a round ran after a failed one");
    }

    /// A round that succeeds while the log is poisoned (a failed rotation
    /// fsync of the same file) acknowledges nothing.
    #[test]
    fn a_round_that_finishes_after_the_log_is_poisoned_fails() {
        let rounds = CommitRounds::new(0);
        let result = rounds.commit(1, Duration::ZERO, || {
            rounds.poison("rotation".into());
            Ok(1)
        });
        assert_eq!(result, Err("rotation".into()));
        assert_eq!(rounds.durable(), 0);
    }

    /// A failed round fails its leader and every later caller.
    #[test]
    fn a_failed_round_poisons() {
        let rounds = CommitRounds::new(0);
        assert_eq!(rounds.commit(1, Duration::ZERO, || Err("disk".into())), Err("disk".into()));
        assert_eq!(rounds.commit(1, Duration::ZERO, || Ok(1)), Err("disk".into()));
        assert_eq!(rounds.poisoned().as_deref(), Some("disk"));
    }
}
