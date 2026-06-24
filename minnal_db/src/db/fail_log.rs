//! Recovery fail-log writer.
//!
//! When a WAL entry cannot be applied after one retry during recovery, it is
//! written to a timestamped JSON file under the configured `fail_log_dir`.
//!
//! Customers can inspect these files to understand what data was not applied
//! and take corrective action (replay, delete, or ignore).

use std::path::{Path, PathBuf};

use log::error;
use serde::Serialize;

use crate::db::wal::{WalEntry, WalOperationType};

// ── JSON types ────────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct FailLog {
    pub recovery_timestamp: String,
    pub db_path: String,
    pub failed_operations: Vec<FailLogOperation>,
}

#[derive(Serialize)]
pub struct FailLogOperation {
    pub name: String,
    pub operation: &'static str,
    pub namespace_id: u32,
    pub key: serde_json::Value,
    pub value: serde_json::Value,
    pub error: String,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Write a fail-log file for WAL entries that could not be applied during
/// recovery.  Each item is `(entry, apply_error_message)`.
pub fn write_fail_log(fail_log_dir: &Path, db_path: &Path, failures: &[(&WalEntry, String)]) {
    if failures.is_empty() {
        return;
    }

    if let Err(e) = std::fs::create_dir_all(fail_log_dir) {
        error!("[FAIL-LOG] Cannot create fail_log directory '{}': {}", fail_log_dir.display(), e);
        return;
    }

    let now = chrono_timestamp();
    let path = fail_log_path(fail_log_dir, &now);

    let log = FailLog {
        recovery_timestamp: now,
        db_path: db_path.display().to_string(),
        failed_operations: failures.iter().map(|(entry, err)| build_operation(entry, err)).collect(),
    };

    match serde_json::to_string_pretty(&log) {
        Ok(json) => {
            if let Err(e) = std::fs::write(&path, json.as_bytes()) {
                error!("[FAIL-LOG] Failed to write '{}': {}", path.display(), e);
            } else {
                error!(
                    "[FAIL-LOG] {} operation(s) could not be applied. \
                     Details written to '{}'. \
                     Inspect the file and replay, delete, or ignore affected records.",
                    failures.len(),
                    path.display()
                );
            }
        }
        Err(e) => {
            error!("[FAIL-LOG] Failed to serialize fail log: {}", e);
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn build_operation(entry: &WalEntry, error: &str) -> FailLogOperation {
    let op_str = match entry.operation {
        WalOperationType::Upsert => "Put",
        WalOperationType::Delete => "Delete",
    };

    let key_json = bytes_to_json(entry.key.as_slice());
    let value_json = entry.value.as_deref().map(bytes_to_json).unwrap_or(serde_json::Value::Null);

    FailLogOperation {
        name: if entry.op_name.is_empty() {
            format!("{}_{}", op_str.to_lowercase(), entry.namespace_id)
        } else {
            entry.op_name.clone()
        },
        operation: op_str,
        namespace_id: entry.namespace_id,
        key: key_json,
        value: value_json,
        error: error.to_owned(),
    }
}

/// Try to represent bytes as a JSON value:
/// 1. Valid UTF-8 JSON  → embed as nested JSON
/// 2. Valid UTF-8 text  → embed as JSON string
/// 3. Binary           → hex string prefixed with `"hex:"`
fn bytes_to_json(bytes: &[u8]) -> serde_json::Value {
    if let Ok(s) = std::str::from_utf8(bytes) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(s) {
            return v;
        }
        return serde_json::Value::String(s.to_owned());
    }
    let hex: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
    serde_json::Value::String(format!("hex:{}", hex))
}

fn chrono_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    // Format as YYYY-MM-DDTHH-MM-SS (safe for filenames on all platforms)
    let s = secs;
    let sec = s % 60;
    let min = (s / 60) % 60;
    let hour = (s / 3600) % 24;
    let days = s / 86400;
    // Rough Gregorian from epoch (good enough for a filename)
    let (year, month, day) = days_to_ymd(days);
    format!("{:04}-{:02}-{:02}T{:02}-{:02}-{:02}", year, month, day, hour, min, sec)
}

fn fail_log_path(dir: &Path, timestamp: &str) -> PathBuf {
    dir.join(format!("fail_log_{}.json", timestamp))
}

/// Convert days-since-Unix-epoch to (year, month, day).  Gregorian calendar.
fn days_to_ymd(mut days: u64) -> (u64, u64, u64) {
    let mut year = 1970u64;
    loop {
        let leap = is_leap(year);
        let days_in_year = if leap { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }
    let leap = is_leap(year);
    let months = [31u64, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut month = 1u64;
    for &m in &months {
        if days < m {
            break;
        }
        days -= m;
        month += 1;
    }
    (year, month, days + 1)
}

fn is_leap(y: u64) -> bool {
    (y.is_multiple_of(4) && !y.is_multiple_of(100)) || y.is_multiple_of(400)
}
