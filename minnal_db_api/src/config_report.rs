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
/// every value is a default). A key that is absent from the file took its default,
/// even if that default happens to equal what a user might have typed.
fn is_from_file(raw: Option<&toml::Table>, section: &str, key: &str) -> bool {
    raw.and_then(|t| t.get(section))
        .and_then(|v| v.as_table())
        .is_some_and(|s| s.contains_key(key))
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
        self.rows.push([section.to_string(), key.to_string(), value.to_string(), source.to_string()]);
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
        let sep = format!("+-{}-+-{}-+-{}-+-{}-+", "-".repeat(width[0]), "-".repeat(width[1]), "-".repeat(width[2]), "-".repeat(width[3]));

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
        info!("configuration: {} value(s) set from the config file: {}", r.overridden.len(), r.overridden.join(", "));
    }
}

/// Collect every effective config value into a [`Report`], tagging each with its
/// source. Separated from [`log_config_table`] so the table can be tested without a
/// tracing subscriber.
fn build_report<'a>(cfg: &DocStoreApiConfig, raw: Option<&'a toml::Table>) -> Report<'a> {
    let mut r = Report::new(raw);

    r.add("storage", "db_path", &cfg.storage.db_path);
    r.add("storage", "schema_dir", &cfg.storage.schema_dir);
    r.add("storage", "log_dir", &cfg.storage.log_dir);

    r.add("api", "listen_addr", &cfg.api.listen_addr);
    r.add("logging", "level", &cfg.logging.level);

    r.add("memtable", "max_capacity", cfg.memtable.max_capacity);
    r.add("sharding", "num_buckets", cfg.sharding.num_buckets);
    r.add("lsm", "compaction_threshold_percent", cfg.lsm.compaction_threshold_percent);
    r.add("sync", "records_per_sync", cfg.sync.records_per_sync);

    let t = &cfg.thresholds;
    r.add("thresholds", "value_log_waste_threshold", t.value_log_waste_threshold);
    r.add("thresholds", "segment_gc_threshold", t.segment_gc_threshold);
    let tail = match t.tail_gc_min_garbage_pct {
        Some(v) => format!("{v}"),
        None => format!("none (tracks trigger = {})", t.value_log_waste_threshold),
    };
    r.add("thresholds", "tail_gc_min_garbage_pct", tail);
    r.add("thresholds", "index_blob_waste_threshold", t.index_blob_waste_threshold);
    r.add("thresholds", "index_blob_backpressure_bytes", bytes(t.index_blob_backpressure_bytes));

    let s = &cfg.scheduled_tasks;
    r.add("scheduled_tasks", "value_log_gc_interval_secs", s.value_log_gc_interval_secs);
    r.add("scheduled_tasks", "wal_gc_interval_secs", s.wal_gc_interval_secs);
    r.add("scheduled_tasks", "lsm_compaction_interval_secs", s.lsm_compaction_interval_secs);
    r.add("scheduled_tasks", "ttl_cleanup_interval_secs", s.ttl_cleanup_interval_secs);

    r.add("wal", "segment_size_bytes", bytes(cfg.wal.segment_size_bytes));
    r.add("value_log", "segment_size_bytes", bytes(cfg.value_log.segment_size_bytes));
    r.add("value_log", "verify_checksums_on_read", cfg.value_log.verify_checksums_on_read);

    let ss = &cfg.semantic_search;
    r.add("semantic_search", "number_of_bits_for_dense_quantisation", ss.number_of_bits_for_dense_quantisation);
    r.add("semantic_search", "n_probes", ss.n_probes);
    r.add("semantic_search", "embedding_dim", ss.embedding_dim);
    r.add("semantic_search", "first_pass_sparse_search_top_k", ss.first_pass_sparse_search_top_k);
    r.add("semantic_search", "window_size", ss.window_size);
    r.add("semantic_search", "sliding_size", ss.sliding_size);
    r.add("semantic_search", "top_k_results", ss.top_k_results);
    r.add("semantic_search", "embedding_service_url", &ss.embedding_service_url);
    r.add("semantic_search", "model", &ss.model);
    let cluster = ss.cluster_path.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "none (bundled default)".to_string());
    r.add("semantic_search", "cluster_path", cluster);
    r.add("semantic_search", "supported_models", format!("{} entry(ies)", ss.supported_models.len()));
    r.add("semantic_search", "query_embedding_cache_ttl_secs", ss.query_embedding_cache_ttl_secs);
    r.add("semantic_search", "embedding_request_timeout_secs", ss.embedding_request_timeout_secs);
    r.add("semantic_search", "embedding_connect_timeout_secs", ss.embedding_connect_timeout_secs);

    let v = &cfg.vector_index;
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
        // A file that sets exactly two keys: one in [thresholds], one in [sharding].
        let raw: toml::Table = toml::from_str(
            r#"
            [sharding]
            num_buckets = 4
            [thresholds]
            segment_gc_threshold = 7.5
        "#,
        )
        .unwrap();

        let cfg = DocStoreApiConfig::default();
        let report = build_report(&cfg, Some(&raw));

        // Exactly the two present keys are marked as coming from the file.
        assert_eq!(
            report.overridden,
            vec!["sharding.num_buckets".to_string(), "thresholds.segment_gc_threshold".to_string()]
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
        // A sibling key in a section that WAS present, but itself absent, is a default.
        assert_eq!(find("thresholds", "value_log_waste_threshold")[3], "default");
        // A key whose whole section is absent is a default.
        assert_eq!(find("value_log", "segment_size_bytes")[3], "default");

        // The rendered table is present and self-describing.
        let table = report.render();
        assert!(table.contains("SECTION") && table.contains("SOURCE"));
        assert!(table.lines().count() > 20, "expected a big table, got:\n{table}");
    }

    #[test]
    fn no_file_means_everything_is_default() {
        let cfg = DocStoreApiConfig::default();
        let report = build_report(&cfg, None);
        assert!(report.overridden.is_empty());
        assert!(report.rows.iter().all(|r| r[3] == "default"));
    }
}
