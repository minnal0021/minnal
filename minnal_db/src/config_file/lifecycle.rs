//! `[lifecycle]`: how the lifecycle governor runs a store's create, drop and
//! index jobs again after a failure, and how long its audit log is kept.

use serde::Deserialize;

use crate::defaults;

/// `[lifecycle]`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LifecycleSection {
    /// Runs again of a failed task before it is given up on (the store is
    /// left as it is, with a CRITICAL log, until the next start).
    pub max_restarts: u32,
    /// Wait before each run again, in milliseconds.
    pub restart_backoff_ms: u64,
    /// Days a lifecycle audit record is kept; at least 1.
    pub audit_retention_days: u64,
}

impl Default for LifecycleSection {
    fn default() -> Self {
        Self {
            max_restarts: defaults::lifecycle::MAX_RESTARTS,
            restart_backoff_ms: defaults::lifecycle::RESTART_BACKOFF.as_millis() as u64,
            audit_retention_days: defaults::lifecycle::AUDIT_RETENTION.as_secs() / 86_400,
        }
    }
}

impl LifecycleSection {
    /// Check what a typed value cannot rule out.
    pub(super) fn validate(&self) -> Result<(), String> {
        if self.audit_retention_days == 0 {
            return Err("lifecycle.audit_retention_days must be at least 1".into());
        }
        Ok(())
    }
}

#[cfg(feature = "doc-store")]
impl super::ConfigFile {
    /// The lifecycle governor's configuration.
    pub fn lifecycle_config(&self) -> crate::namespace_lifecycle::LifecycleConfig {
        crate::namespace_lifecycle::LifecycleConfig {
            max_restarts: self.lifecycle.max_restarts,
            restart_backoff: std::time::Duration::from_millis(self.lifecycle.restart_backoff_ms),
            restart_window: defaults::lifecycle::RESTART_WINDOW,
        }
    }

    /// How long a lifecycle audit record is kept.
    pub fn audit_retention(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.lifecycle.audit_retention_days * 86_400)
    }
}
