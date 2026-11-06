//! Latency recording and summaries. Each client records into its own
//! [`Recording`]; they are merged after the cell, so nothing on the measured
//! path is shared.

use std::collections::BTreeMap;
use std::time::Duration;

use serde::Serialize;

use crate::workload::CLASSES;

/// Histogram bucket width: 2% (bucket `b` holds latencies around 1.02^b µs).
const BUCKET_GROWTH: f64 = 1.02;
/// Error messages kept per class.
const ERROR_SAMPLES: usize = 5;

#[derive(Default, Clone)]
pub struct ClassRecording {
    latencies_us: Vec<u32>,
    errors: u64,
    not_found: u64,
    /// An operation of this class was due but none was available (a delete
    /// that would pass the inserts).
    pub unavailable: u64,
    error_samples: Vec<String>,
}

impl ClassRecording {
    pub fn ok(&mut self, took: Duration) {
        self.latencies_us.push(took.as_micros().min(u32::MAX as u128) as u32);
    }
    pub fn not_found(&mut self, took: Duration) {
        self.ok(took);
        self.not_found += 1;
    }
    pub fn error(&mut self, took: Duration, e: String) {
        self.ok(took);
        self.errors += 1;
        if self.error_samples.len() < ERROR_SAMPLES {
            self.error_samples.push(e);
        }
    }
    fn merge(&mut self, o: &ClassRecording) {
        self.latencies_us.extend_from_slice(&o.latencies_us);
        self.errors += o.errors;
        self.not_found += o.not_found;
        self.unavailable += o.unavailable;
        for e in &o.error_samples {
            if self.error_samples.len() < ERROR_SAMPLES {
                self.error_samples.push(e.clone());
            }
        }
    }
}

/// One client's recording: per class.
#[derive(Default, Clone)]
pub struct Recording(pub [ClassRecording; 8]);

impl Recording {
    pub fn merge(&mut self, o: &Recording) {
        for (a, b) in self.0.iter_mut().zip(&o.0) {
            a.merge(b);
        }
    }
    pub fn errors(&self) -> u64 {
        self.0.iter().map(|c| c.errors).sum()
    }
    pub fn unavailable(&self) -> u64 {
        self.0.iter().map(|c| c.unavailable).sum()
    }
    pub fn completed(&self) -> u64 {
        self.0.iter().map(|c| c.latencies_us.len() as u64).sum()
    }
    pub fn count(&self, class: usize) -> u64 {
        self.0[class].latencies_us.len() as u64
    }
    pub fn error_samples(&self) -> Vec<String> {
        self.0
            .iter()
            .zip(CLASSES)
            .flat_map(|(c, n)| c.error_samples.iter().map(move |e| format!("{n}: {e}")))
            .collect()
    }

    /// Per-class summaries over `measured` (classes with no operation left out).
    pub fn summary(&self, measured: Duration) -> BTreeMap<&'static str, ClassSummary> {
        self.0
            .iter()
            .zip(CLASSES)
            .filter(|(c, _)| !c.latencies_us.is_empty() || c.unavailable > 0)
            .map(|(c, name)| (name, ClassSummary::of(c, measured)))
            .collect()
    }
}

#[derive(Serialize)]
pub struct ClassSummary {
    pub count: u64,
    pub ops_per_s: f64,
    pub mean_ms: f64,
    pub p50_ms: f64,
    pub p75_ms: f64,
    pub p90_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub p999_ms: f64,
    pub max_ms: f64,
    pub errors: u64,
    pub not_found: u64,
    pub unavailable: u64,
    /// `(bucket, count)`: bucket `b` holds latencies in `[1.02^b, 1.02^(b+1))`
    /// µs, so repeats can be merged without the raw samples.
    pub histogram: Vec<(u32, u64)>,
}

impl ClassSummary {
    fn of(c: &ClassRecording, measured: Duration) -> Self {
        let mut l = c.latencies_us.clone();
        l.sort_unstable();
        let q = |p: f64| {
            if l.is_empty() {
                return 0.0;
            }
            // Nearest rank.
            let i = ((p * l.len() as f64).ceil() as usize).clamp(1, l.len()) - 1;
            l[i] as f64 / 1e3
        };
        let mut hist: BTreeMap<u32, u64> = BTreeMap::new();
        for &us in &l {
            *hist.entry(((us.max(1) as f64).ln() / BUCKET_GROWTH.ln()) as u32).or_default() += 1;
        }
        let n = l.len() as u64;
        Self {
            count: n,
            ops_per_s: n as f64 / measured.as_secs_f64(),
            mean_ms: if n == 0 {
                0.0
            } else {
                l.iter().map(|&x| x as f64).sum::<f64>() / n as f64 / 1e3
            },
            p50_ms: q(0.50),
            p75_ms: q(0.75),
            p90_ms: q(0.90),
            p95_ms: q(0.95),
            p99_ms: q(0.99),
            p999_ms: q(0.999),
            max_ms: l.last().map_or(0.0, |&x| x as f64 / 1e3),
            errors: c.errors,
            not_found: c.not_found,
            unavailable: c.unavailable,
            histogram: hist.into_iter().collect(),
        }
    }
}
