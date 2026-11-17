//! Where the CPU went during a cell's measured stretch, and whether threads
//! waited for a core: the test for "searches are starved of CPU".
//!
//! Four sources, read before and after the stretch:
//!
//! - **The process** (`getrusage(RUSAGE_SELF)`): user and system CPU of every
//!   thread, including threads that exited during the stretch, and the context
//!   switches. An *involuntary* switch is the kernel taking a core away from a
//!   thread that could still run.
//! - **Each live thread** (`/proc/self/task/*/schedstat`): time it ran and time
//!   it was **runnable but waiting for a core** (on the run queue), grouped by
//!   thread name with any trailing number dropped (`rayon-7` → `rayon`). The
//!   rayon pool is named in `main.rs` so search scoring is its own group; the
//!   field-index worker (`minnal-field-index`), the open-loop dispatcher and the
//!   WAL segment preparer name themselves; Tokio's workers and blocking pool
//!   share one name. A thread that exits during the stretch takes its time with
//!   it: that part shows only in the process total (`exited_or_unseen_s`).
//!   Much of a search's and a scan's work runs on such threads: the LSM and
//!   value-log reads fan out over `std::thread::scope` threads made per call.
//! - **CPU pressure** (`/proc/pressure/cpu`, PSI): time in which at least one
//!   runnable task on the host waited for a core. It covers every thread,
//!   short-lived ones included, which the per-thread wait cannot.
//! - **The host** (`/proc/stat`): busy time of every core, so CPU taken by
//!   other processes on the machine is visible too (and is in the pressure).
//!
//! Wait time is what tells starvation from queueing: a search that waits at the
//! search gate (a Tokio semaphore) is asleep, not runnable, so only a search
//! thread that could run and got no core adds to its group's wait.

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use serde::Serialize;

/// One reading of the process, its threads and the host.
pub(crate) struct CpuSnapshot {
    at: Instant,
    user_us: u64,
    sys_us: u64,
    voluntary_switches: u64,
    involuntary_switches: u64,
    /// Thread id → (group, ran ns, waited ns).
    threads: HashMap<u32, (String, u64, u64)>,
    /// Host jiffies: (busy, all).
    host: (u64, u64),
    /// PSI `some` total, microseconds; `None` without `/proc/pressure`.
    pressure_us: Option<u64>,
}

/// One thread group's share of the stretch.
#[derive(Serialize, Clone, Default)]
pub(crate) struct GroupUse {
    /// Threads of the group alive at the end of the stretch.
    pub threads: usize,
    /// CPU time the group's threads ran.
    pub run_s: f64,
    /// Time they were runnable but had no core.
    pub wait_s: f64,
    /// `run_s` over the stretch: cores the group kept busy on average.
    pub cores: f64,
    /// `wait_s / run_s`: time waiting for a core per second run.
    pub wait_per_run: f64,
}

/// What a cell's measured stretch cost in CPU.
#[derive(Serialize, Clone)]
pub(crate) struct CpuUse {
    pub elapsed_s: f64,
    pub user_s: f64,
    pub sys_s: f64,
    /// `(user_s + sys_s) / elapsed_s`: cores this process kept busy.
    pub process_cores: f64,
    /// Cores busy on the whole host, this process included.
    pub host_busy_cores: f64,
    /// `host_busy_cores - process_cores`: other processes on the machine.
    pub other_processes_cores: f64,
    pub host_cores: usize,
    /// Share of the stretch, in percent, in which some runnable task on the
    /// host waited for a core (PSI `some`); `None` where the kernel has no PSI.
    pub cpu_pressure_some_pct: Option<f64>,
    pub voluntary_switches: u64,
    pub involuntary_switches: u64,
    /// Process CPU per completed operation (measured and trickle).
    pub cpu_ms_per_op: f64,
    /// The `rayon` group's run time per completed search (plain and filtered):
    /// the scoring a search does on the pool.
    pub rayon_ms_per_search: f64,
    pub groups: BTreeMap<String, GroupUse>,
    /// Process CPU no live thread accounts for: threads that exited during the
    /// stretch (Tokio's idle blocking threads, scoped threads).
    pub exited_or_unseen_s: f64,
}

impl CpuSnapshot {
    pub(crate) fn take() -> Self {
        let (user_us, sys_us, voluntary_switches, involuntary_switches) = rusage();
        Self {
            at: Instant::now(),
            user_us,
            sys_us,
            voluntary_switches,
            involuntary_switches,
            threads: threads(),
            host: host_jiffies(),
            pressure_us: cpu_pressure_us(),
        }
    }

    /// The stretch from `before` to `self`. `ops` is every operation completed
    /// in it, `searches` the plain and filtered searches among them.
    pub(crate) fn since(&self, before: &CpuSnapshot, ops: u64, searches: u64) -> CpuUse {
        let elapsed_s = self.at.duration_since(before.at).as_secs_f64().max(1e-9);
        let user_s = self.user_us.saturating_sub(before.user_us) as f64 / 1e6;
        let sys_s = self.sys_us.saturating_sub(before.sys_us) as f64 / 1e6;
        let mut groups: BTreeMap<String, GroupUse> = BTreeMap::new();
        for (tid, (group, ran, waited)) in &self.threads {
            // A thread born during the stretch counts from zero.
            let (ran0, waited0) = before.threads.get(tid).map_or((0, 0), |(_, r, w)| (*r, *w));
            let g = groups.entry(group.clone()).or_default();
            g.threads += 1;
            g.run_s += ran.saturating_sub(ran0) as f64 / 1e9;
            g.wait_s += waited.saturating_sub(waited0) as f64 / 1e9;
        }
        for g in groups.values_mut() {
            g.cores = g.run_s / elapsed_s;
            g.wait_per_run = if g.run_s > 0.0 { g.wait_s / g.run_s } else { 0.0 };
        }
        let seen: f64 = groups.values().map(|g| g.run_s).sum();
        let (busy, all) = (self.host.0.saturating_sub(before.host.0), self.host.1.saturating_sub(before.host.1));
        let host_cores = std::thread::available_parallelism().map_or(1, |n| n.get());
        let host_busy_cores = if all > 0 { busy as f64 / all as f64 * host_cores as f64 } else { 0.0 };
        let process_cores = (user_s + sys_s) / elapsed_s;
        let rayon_s = groups.get("rayon").map_or(0.0, |g| g.run_s);
        CpuUse {
            elapsed_s,
            user_s,
            sys_s,
            process_cores,
            host_busy_cores,
            other_processes_cores: (host_busy_cores - process_cores).max(0.0),
            host_cores,
            cpu_pressure_some_pct: match (self.pressure_us, before.pressure_us) {
                (Some(a), Some(b)) => Some(a.saturating_sub(b) as f64 / 1e4 / elapsed_s),
                _ => None,
            },
            voluntary_switches: self.voluntary_switches.saturating_sub(before.voluntary_switches),
            involuntary_switches: self.involuntary_switches.saturating_sub(before.involuntary_switches),
            cpu_ms_per_op: if ops > 0 { (user_s + sys_s) * 1e3 / ops as f64 } else { 0.0 },
            rayon_ms_per_search: if searches > 0 { rayon_s * 1e3 / searches as f64 } else { 0.0 },
            groups,
            exited_or_unseen_s: (user_s + sys_s - seen).max(0.0),
        }
    }
}

fn rusage() -> (u64, u64, u64, u64) {
    // SAFETY: getrusage only writes the struct it is given.
    let mut r: libc::rusage = unsafe { std::mem::zeroed() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut r) } != 0 {
        return (0, 0, 0, 0);
    }
    let us = |t: libc::timeval| t.tv_sec as u64 * 1_000_000 + t.tv_usec as u64;
    (us(r.ru_utime), us(r.ru_stime), r.ru_nvcsw as u64, r.ru_nivcsw as u64)
}

/// Every live thread: its group (name without a trailing number) and its
/// schedstat run and wait times. Empty where `/proc` has no schedstat (macOS,
/// or a kernel without `CONFIG_SCHED_INFO`).
fn threads() -> HashMap<u32, (String, u64, u64)> {
    let mut out = HashMap::new();
    let Ok(dir) = std::fs::read_dir("/proc/self/task") else { return out };
    for entry in dir.flatten() {
        let Some(tid) = entry.file_name().to_str().and_then(|s| s.parse::<u32>().ok()) else {
            continue;
        };
        let path = entry.path();
        let Ok(stat) = std::fs::read_to_string(path.join("schedstat")) else {
            continue;
        };
        let mut f = stat.split_whitespace().map(|v| v.parse::<u64>().unwrap_or(0));
        let (ran, waited) = (f.next().unwrap_or(0), f.next().unwrap_or(0));
        let name = std::fs::read_to_string(path.join("comm")).unwrap_or_default();
        out.insert(tid, (group_of(name.trim()), ran, waited));
    }
    out
}

/// `rayon-12` → `rayon`, `tokio-runtime-w` unchanged. Thread names are cut to 15
/// bytes by the kernel, so `minnal-field-index` reads `minnal-field-in`.
fn group_of(name: &str) -> String {
    let g = name.trim_end_matches(|c: char| c.is_ascii_digit()).trim_end_matches(['-', '_']);
    if g.is_empty() { name.to_string() } else { g.to_string() }
}

/// Host-wide (busy, all) jiffies from the first line of `/proc/stat`; idle and
/// iowait are not busy.
fn host_jiffies() -> (u64, u64) {
    let stat = std::fs::read_to_string("/proc/stat").unwrap_or_default();
    let Some(line) = stat.lines().next() else { return (0, 0) };
    let v: Vec<u64> = line.split_whitespace().skip(1).map(|x| x.parse().unwrap_or(0)).collect();
    let all: u64 = v.iter().sum();
    let idle = v.get(3).copied().unwrap_or(0) + v.get(4).copied().unwrap_or(0);
    (all.saturating_sub(idle), all)
}

/// The `some` line's `total=` from `/proc/pressure/cpu`: microseconds in which
/// at least one runnable task waited for a core.
fn cpu_pressure_us() -> Option<u64> {
    let p = std::fs::read_to_string("/proc/pressure/cpu").ok()?;
    let line = p.lines().find(|l| l.starts_with("some"))?;
    line.split_whitespace().find_map(|f| f.strip_prefix("total=")?.parse().ok())
}
