//! Startup configuration report.
//!
//! Logs a table of every effective configuration value alongside its **source** —
//! whether it was read from the config file or fell back to a built-in default — so
//! a misbehaving deployment can be debugged from the log alone, without guessing
//! which knob actually took effect. See [`log_config_table`].

use crate::config::DocStoreApiConfig;
use tracing::info;

/// Whether `[section] key` is explicitly present in the parsed TOML.
///
/// `raw` is the file parsed as a plain table; `None` means no file was given (so
/// every value is a default). `section` may be a dotted path to a nested table
/// (`semantic_search.maintenance`). A key that is absent from the file took its
/// default, even if that default happens to equal what a user might have typed.
fn is_from_file(raw: Option<&toml::Table>, section: &str, key: &str) -> bool {
    let mut table = raw;
    for part in section.split('.') {
        table = table.and_then(|t| t.get(part)).and_then(|v| v.as_table());
    }
    table.is_some_and(|t| t.contains_key(key))
}

/// Human-friendly byte size, e.g. `67108864 (64 MiB)`.
fn bytes(n: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = KIB * 1024;
    const GIB: u64 = MIB * 1024;
    let pretty = if n >= GIB && n.is_multiple_of(GIB) {
        format!("{} GiB", n / GIB)
    } else if n >= MIB {
        format!("{:.0} MiB", n as f64 / MIB as f64)
    } else if n >= KIB {
        format!("{:.0} KiB", n as f64 / KIB as f64)
    } else {
        format!("{n} B")
    };
    format!("{n} ({pretty})")
}

struct Report<'a> {
    raw: Option<&'a toml::Table>,
    rows: Vec<[String; 4]>,
    overridden: Vec<String>,
}

impl<'a> Report<'a> {
    fn new(raw: Option<&'a toml::Table>) -> Self {
        Self {
            raw,
            rows: Vec::new(),
            overridden: Vec::new(),
        }
    }

    fn add(&mut self, section: &str, key: &str, value: impl std::fmt::Display) {
        let from_file = is_from_file(self.raw, section, key);
        let source = if from_file { "config" } else { "default" };
        if from_file {
            self.overridden.push(format!("{section}.{key}"));
        }
        self.rows
            .push([section.to_string(), key.to_string(), value.to_string(), source.to_string()]);
    }

    /// Render the collected rows as a fixed-width table.
    fn render(&self) -> String {
        const HEADERS: [&str; 4] = ["SECTION", "KEY", "VALUE", "SOURCE"];
        let mut width = HEADERS.map(str::len);
        for r in &self.rows {
            for i in 0..4 {
                width[i] = width[i].max(r[i].len());
            }
        }
        let line = |cells: &[String; 4]| {
            format!(
                "| {:<sw$} | {:<kw$} | {:<vw$} | {:<ow$} |",
                cells[0],
                cells[1],
                cells[2],
                cells[3],
                sw = width[0],
                kw = width[1],
                vw = width[2],
                ow = width[3],
            )
        };
        let sep = format!(
            "+-{}-+-{}-+-{}-+-{}-+",
            "-".repeat(width[0]),
            "-".repeat(width[1]),
            "-".repeat(width[2]),
            "-".repeat(width[3])
        );

        let mut out = String::new();
        out.push_str("effective configuration (SOURCE: 'config' = set in file, 'default' = built-in)\n");
        out.push_str(&sep);
        out.push('\n');
        out.push_str(&line(&HEADERS.map(String::from)));
        out.push('\n');
        out.push_str(&sep);
        out.push('\n');
        for r in &self.rows {
            out.push_str(&line(r));
            out.push('\n');
        }
        out.push_str(&sep);
        out
    }
}

/// Log the full effective configuration as a table, marking each value's source.
///
/// `raw` is the config file parsed as a plain [`toml::Table`] (or `None` when no
/// file was supplied and everything is a default); it is what distinguishes a value
/// set in the file from one that fell back to its default.
pub fn log_config_table(cfg: &DocStoreApiConfig, raw: Option<&toml::Table>) {
    let r = build_report(cfg, raw);
    info!("\n{}", r.render());
    if raw.is_none() {
        info!("configuration source: no config file — all values are built-in defaults");
    } else if r.overridden.is_empty() {
        info!("configuration source: a file was loaded but sets no recognised keys — all values are defaults");
    } else {
        info!(
            "configuration: {} value(s) set from the config file: {}",
            r.overridden.len(),
            r.overridden.join(", ")
        );
    }
}

/// Collect every effective config value into a [`Report`], tagging each with its
/// source. Separated from [`log_config_table`] so the table can be tested without a
/// tracing subscriber.
fn build_report<'a>(cfg: &DocStoreApiConfig, raw: Option<&'a toml::Table>) -> Report<'a> {
    let mut r = Report::new(raw);
    let f = &cfg.file;

    r.add("storage", "db_path", &f.storage.db_path);
    r.add("storage", "log_dir", &f.storage.log_dir);

    r.add("api", "listen_addr", &cfg.api.listen_addr);
    r.add("logging", "level", &cfg.logging.level);

    r.add("memtable", "max_capacity", f.memtable.max_capacity);
    r.add("sharding", "num_buckets", f.sharding.num_buckets);
    r.add("lsm", "compaction_threshold_percent", f.lsm.compaction_threshold_percent);
    r.add("sync", "records_per_sync", f.sync.records_per_sync);
    r.add("sync", "commit_delay_us", f.sync.commit_delay_us);

    let t = &f.thresholds;
    r.add("thresholds", "value_log_waste_threshold", t.value_log_waste_threshold);
    r.add("thresholds", "segment_gc_threshold", t.segment_gc_threshold);
    let tail = match t.tail_gc_min_garbage_pct {
        Some(v) => format!("{v}"),
        None => format!("none (tracks trigger = {})", t.value_log_waste_threshold),
    };
    r.add("thresholds", "tail_gc_min_garbage_pct", tail);
    r.add("thresholds", "index_blob_waste_threshold", t.index_blob_waste_threshold);
    r.add("thresholds", "index_blob_backpressure_bytes", bytes(t.index_blob_backpressure_bytes));
    r.add("thresholds", "index_overlay_soft_bytes", bytes(t.index_overlay_soft_bytes));
    r.add("thresholds", "index_overlay_hard_bytes", bytes(t.index_overlay_hard_bytes));
    r.add("thresholds", "max_pinned_wal_segments", t.max_pinned_wal_segments);

    let s = &f.scheduled_tasks;
    r.add("scheduled_tasks", "value_log_gc_interval_secs", s.value_log_gc_interval_secs);
    r.add("scheduled_tasks", "wal_gc_interval_secs", s.wal_gc_interval_secs);
    r.add("scheduled_tasks", "lsm_compaction_interval_secs", s.lsm_compaction_interval_secs);
    r.add("scheduled_tasks", "ttl_cleanup_interval_secs", s.ttl_cleanup_interval_secs);
    r.add("scheduled_tasks", "index_checkpoint_interval_ms", s.index_checkpoint_interval_ms);

    r.add("wal", "segment_size_bytes", bytes(f.wal.segment_size_bytes));
    r.add("value_log", "segment_size_bytes", bytes(f.value_log.segment_size_bytes));
    r.add("value_log", "verify_checksums_on_read", f.value_log.verify_checksums_on_read);
    let fail_logs = match &f.recovery.fail_log_dir {
        Some(dir) => dir.clone(),
        None => format!("{}/fail_logs", f.storage.db_path),
    };
    r.add("recovery", "fail_log_dir", fail_logs);

    let l = &f.lifecycle;
    r.add("lifecycle", "max_restarts", l.max_restarts);
    r.add("lifecycle", "restart_backoff_ms", l.restart_backoff_ms);
    r.add("lifecycle", "audit_retention_days", l.audit_retention_days);

    let ss = &f.semantic_search;
    r.add("semantic_search", "embedding_service_url", &ss.embedding_service_url);
    let models = if ss.supported_models.is_empty() {
        "any the embedding service serves".to_string()
    } else {
        ss.supported_models
            .iter()
            .map(|m| format!("{} ({}-d)", m.name, m.dimension))
            .collect::<Vec<_>>()
            .join(", ")
    };
    r.add("semantic_search", "supported_models", models);
    r.add("semantic_search", "query_embedding_cache_ttl_secs", ss.query_embedding_cache_ttl_secs);
    r.add("semantic_search", "partition_history_ttl_secs", ss.partition_history_ttl_secs);
    r.add("semantic_search", "embedding_request_timeout_secs", ss.embedding_request_timeout_secs);
    r.add("semantic_search", "embedding_connect_timeout_secs", ss.embedding_connect_timeout_secs);
    let m = &ss.maintenance;
    r.add("semantic_search.maintenance", "threads", m.threads);
    r.add("semantic_search.maintenance", "rebuild_threads", m.rebuild_threads);
    r.add("semantic_search.maintenance", "target_posting_size", m.target_posting_size);
    r.add("semantic_search.maintenance", "reassign_range", m.reassign_range);
    r.add("semantic_search.maintenance", "merge_ratio", format!("{:.4}", m.merge_ratio));
    r.add("semantic_search.maintenance", "skip_uncertain_moves", m.skip_uncertain_moves);
    r.add("semantic_search.maintenance", "split_samples", m.split_samples);
    r.add("semantic_search.maintenance", "split_init_trials", m.split_init_trials);
    r.add("semantic_search.maintenance", "split_max_iters", m.split_max_iters);
    r.add("semantic_search.maintenance", "split_lambda_factor", m.split_lambda_factor);
    let q = &ss.search_defaults;
    r.add("semantic_search.search_defaults", "min_probes", q.min_probes);
    r.add("semantic_search.search_defaults", "max_probes", q.max_probes);
    r.add("semantic_search.search_defaults", "first_pass_top_k", q.first_pass_top_k);
    r.add("semantic_search.search_defaults", "top_k", q.top_k);

    let v = &f.vector_index;
    r.add("vector_index", "retry_wait_secs", v.retry_wait_secs);
    r.add("vector_index", "max_retries", v.max_retries);
    r.add("vector_index", "concurrency", v.concurrency);

    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_is_config_only_for_keys_present_in_the_file() {
        // A file that sets exactly three keys, one of them in a nested table.
        let raw: toml::Table = toml::from_str(
            r#"
            [sharding]
            num_buckets = 4
            [thresholds]
            segment_gc_threshold = 7.5
            [semantic_search.maintenance]
            threads = 3
        "#,
        )
        .unwrap();

        let cfg = DocStoreApiConfig::default();
        let report = build_report(&cfg, Some(&raw));

        // Exactly the present keys are marked as coming from the file.
        assert_eq!(
            report.overridden,
            vec![
                "sharding.num_buckets".to_string(),
                "thresholds.segment_gc_threshold".to_string(),
                "semantic_search.maintenance.threads".to_string(),
            ]
        );

        let find = |section: &str, key: &str| {
            report
                .rows
                .iter()
                .find(|r| r[0] == section && r[1] == key)
                .unwrap_or_else(|| panic!("row {section}.{key} missing from the table"))
        };
        assert_eq!(find("sharding", "num_buckets")[3], "config");
        assert_eq!(find("thresholds", "segment_gc_threshold")[3], "config");
        // A key of a nested table.
        assert_eq!(find("semantic_search.maintenance", "threads")[3], "config");
        assert_eq!(find("semantic_search.maintenance", "rebuild_threads")[3], "default");
        // A sibling key in a section that WAS present, but itself absent, is a default.
        assert_eq!(find("thresholds", "value_log_waste_threshold")[3], "default");
        // A key whose whole section is absent is a default.
        assert_eq!(find("value_log", "segment_size_bytes")[3], "default");

        // The rendered table is present and self-describing.
        let table = report.render();
        assert!(table.contains("SECTION") && table.contains("SOURCE"));
        assert!(table.lines().count() > 20, "expected a big table, got:\n{table}");
    }

    /// Every key of the reference config has a row, so a key added to the
    /// file cannot be left out of the startup table.
    #[test]
    fn every_key_of_the_sample_config_is_reported() {
        fn keys(table: &toml::Table, section: &str, out: &mut Vec<(String, String)>) {
            for (key, value) in table {
                match value.as_table() {
                    Some(inner) => {
                        let path = if section.is_empty() { key.clone() } else { format!("{section}.{key}") };
                        keys(inner, &path, out);
                    }
                    None => out.push((section.to_string(), key.clone())),
                }
            }
        }
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../config/sample.toml");
        let raw: toml::Table = std::fs::read_to_string(&path).unwrap().parse().unwrap();
        let cfg = DocStoreApiConfig::from_file(&path).unwrap();
        let report = build_report(&cfg, Some(&raw));
        let mut all = Vec::new();
        keys(&raw, "", &mut all);
        assert!(all.len() > 30, "{all:?}");
        for (section, key) in all {
            assert!(
                report.rows.iter().any(|r| r[0] == section && r[1] == key && r[3] == "config"),
                "{section}.{key} is not reported as set"
            );
        }
    }

    #[test]
    fn no_file_means_everything_is_default() {
        let cfg = DocStoreApiConfig::default();
        let report = build_report(&cfg, None);
        assert!(report.overridden.is_empty());
        assert!(report.rows.iter().all(|r| r[3] == "default"));
    }
}
