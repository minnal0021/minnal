//! Shared fixtures for the `db` module's unit tests.
//!
//! These used to live inside `database.rs`'s test module, which is why every
//! test needing one had to live there too — that is what kept a 3,300-line test
//! block attached to a file whose code had already been split up.
//!
//! The whole module is `#[cfg(test)]`, so none of it reaches a release build.

use std::sync::Arc;
use std::time::Duration;

use crate::db::config::{DbConfig, ScheduledTaskConfig, SyncConfig, ThresholdConfig};
use crate::db::database::Database;
use crate::db::index_manager::{GapCause, GapRecord, RepairMode};
use crate::db::namespace::FieldId;
use crate::db::namespace_index::ExtractorFn;
use crate::index::{IndexValue, IndexValueType};
use crate::store::lsm::lsm_tree::LSMConfig;

pub(crate) fn create_db_config() -> DbConfig {
    let gc_interval = Duration::from_secs(5);
    let wal_gc_interval = Duration::from_secs(5);
    let lsm_compaction_interval = Duration::from_secs(5);

    let sync_config = SyncConfig::default();
    let threshold_config = ThresholdConfig::new(2.5);
    let scheduled_task_config = ScheduledTaskConfig::new(gc_interval, wal_gc_interval, lsm_compaction_interval);
    let lsm_config = LSMConfig::default();
    let mut config = DbConfig::new(threshold_config, scheduled_task_config, sync_config, lsm_config);
    config.num_buckets = crate::support::TEST_NUM_BUCKETS;
    config
}

/// Config shared by the crash child and the recounting parent — both must
/// open the same database with the same shape.
///
/// Deliberately tuned so a crash is *destructive* if the durability rules are
/// wrong. `records_per_sync` is low so the value-log fsync path runs
/// constantly (the F1 trigger). `skip_list_capacity` sits between the two key
/// counts so `busy` flushes to level 0 several times while `quiet` never does
/// (the F2 trigger). The background workers are pushed out of reach so the
/// outcome depends only on the write path, never on a worker happening to
/// tick before the kill.
pub(crate) fn crash_test_config() -> DbConfig {
    let far_future = Duration::from_secs(3600);
    let scheduled_task_config = ScheduledTaskConfig::new(far_future, far_future, far_future);
    let lsm_config = LSMConfig::default();
    let mut config = DbConfig::new(ThresholdConfig::new(2.5), scheduled_task_config, SyncConfig::new(50), lsm_config);
    config.num_buckets = crate::support::TEST_NUM_BUCKETS;
    config.lsm_config.skip_list_capacity = 500;
    config
}

/// Build a namespace with one activated `status` field index and a couple of
/// documents. Returns the field id.
pub(crate) fn activate_status_index(db: &Database, ns: u32) -> FieldId {
    let field_id = db.register_index_field(ns, "status", IndexValueType::Str).unwrap();
    let extractor: ExtractorFn = Arc::new(|bytes: &[u8]| {
        let s = std::str::from_utf8(bytes).ok()?;
        let v: serde_json::Value = serde_json::from_str(s).ok()?;
        Some(IndexValue::Str(v["status"].as_str()?.to_string()))
    });
    db.activate_field_index(ns, field_id, IndexValueType::Str, extractor).unwrap();
    field_id
}

/// Activate an index over the named JSON string field. Returns the field id.
pub(crate) fn activate_named_index(db: &Database, ns: u32, field: &'static str) -> FieldId {
    let field_id = db.register_index_field(ns, field, IndexValueType::Str).unwrap();
    let extractor: ExtractorFn = Arc::new(move |bytes: &[u8]| {
        let s = std::str::from_utf8(bytes).ok()?;
        let v: serde_json::Value = serde_json::from_str(s).ok()?;
        Some(IndexValue::Str(v[field].as_str()?.to_string()))
    });
    db.activate_field_index(ns, field_id, IndexValueType::Str, extractor).unwrap();
    field_id
}

/// Record a minimal gap so a field reads as degraded.
pub(crate) fn record_test_gap(db: &Database, ns: u32, field_id: FieldId) {
    db.index_manager
        .record_gap(
            GapRecord {
                namespace_id: ns,
                field_id,
                cause: GapCause::BackstopReclaim,
                from: 0,
                to: 0,
                missing_segments: vec![],
                detected_at_ms: 1,
                repair: RepairMode::FullRebuild,
            },
            1000,
        )
        .unwrap();
}
