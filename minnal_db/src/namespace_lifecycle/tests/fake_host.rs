//! `FakeHost`: a [`LifecycleHost`] that keeps its namespaces' states in a map
//! and whose tasks do nothing but what a test scripts: fail, panic, or wait
//! for the test to let them go.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use tempfile::TempDir;
use tokio::sync::Semaphore;

use crate::AsyncDb;
use crate::namespace::NamespaceId;
use crate::namespace_lifecycle::{JobKey, JobKind, Jobs, LifecycleError, LifecycleHost, MaintenanceStore, NamespaceState, Task};

/// What a scripted task does on its next runs.
#[derive(Default, Clone)]
pub(super) struct Script {
    /// Runs left that fail.
    pub fail: u32,
    /// Whether a failing run panics instead of returning an error.
    pub panic: bool,
    /// When set, every run waits for a permit first.
    pub gate: Option<Arc<Semaphore>>,
}

pub(super) struct FakeHost {
    pub states: Mutex<HashMap<String, (NamespaceState, NamespaceId)>>,
    pub scripts: Mutex<HashMap<Task, Script>>,
    pub runs: Mutex<HashMap<Task, u32>>,
    pub jobs: Arc<Jobs>,
    /// Kept for the records' namespace.
    _db: AsyncDb,
    _dir: TempDir,
}

impl FakeHost {
    /// A host with no namespaces, its records in a fresh database.
    pub async fn new() -> Arc<Self> {
        let dir = TempDir::new().unwrap();
        let db = AsyncDb::open_with_config(dir.path().to_path_buf(), crate::support::test_db_config())
            .await
            .unwrap();
        let records = Arc::new(MaintenanceStore::open(&db).await.unwrap());
        Arc::new(Self {
            states: Mutex::default(),
            scripts: Mutex::default(),
            runs: Mutex::default(),
            jobs: Arc::new(Jobs::new(
                records,
                Arc::new(
                    crate::namespace_lifecycle::AuditLog::open(&db, crate::namespace_lifecycle::DEFAULT_AUDIT_RETENTION)
                        .await
                        .unwrap(),
                ),
            )),
            _db: db,
            _dir: dir,
        })
    }

    /// Add namespace `name` in `state`; returns its id.
    pub fn add(&self, name: &str, state: NamespaceState) -> NamespaceId {
        let id = NamespaceId::random().unwrap();
        self.states.lock().insert(name.to_owned(), (state, id));
        id
    }

    pub fn set_state(&self, name: &str, state: NamespaceState) {
        self.states.lock().get_mut(name).unwrap().0 = state;
    }

    pub fn script(&self, task: Task, script: Script) {
        self.scripts.lock().insert(task, script);
    }

    pub fn runs(&self, task: Task) -> u32 {
        self.runs.lock().get(&task).copied().unwrap_or(0)
    }

    /// Start job `kind` on `name` through the registry.
    pub fn start(&self, name: &str, kind: JobKind) -> Result<JobKey, LifecycleError> {
        let key = JobKey::new(self.namespace_id(name).unwrap(), kind);
        self.jobs.start(self, name, key, None, serde_json::Value::Null)?;
        Ok(key)
    }

    /// Wait up to 10 s for `check` to hold.
    pub async fn wait_until(&self, what: &str, check: impl Fn(&Self) -> bool) {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        while !check(self) {
            assert!(tokio::time::Instant::now() < deadline, "never: {what}");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }
}

impl LifecycleHost for FakeHost {
    type Error = LifecycleError;

    fn states(&self) -> Vec<(String, NamespaceState)> {
        let mut states: Vec<_> = self.states.lock().iter().map(|(n, (s, _))| (n.clone(), *s)).collect();
        states.sort_by(|a, b| a.0.cmp(&b.0));
        states
    }

    fn state(&self, namespace: &str) -> Option<NamespaceState> {
        self.states.lock().get(namespace).map(|(s, _)| *s)
    }

    fn namespace_id(&self, namespace: &str) -> Option<NamespaceId> {
        self.states.lock().get(namespace).map(|(_, id)| *id)
    }

    fn transition(&self, namespace: &str, from: &[NamespaceState], to: NamespaceState) -> Result<NamespaceState, LifecycleError> {
        let mut states = self.states.lock();
        let (state, _) = states.get_mut(namespace).ok_or_else(|| LifecycleError::NotFound {
            namespace: namespace.to_owned(),
        })?;
        if !from.contains(state) {
            return Err(state.refusal(namespace, to));
        }
        Ok(std::mem::replace(state, to))
    }

    async fn run(&self, namespace: &str, task: Task) -> Result<(), LifecycleError> {
        *self.runs.lock().entry(task).or_default() += 1;
        let script = self.scripts.lock().get(&task).cloned().unwrap_or_default();
        if let Some(gate) = &script.gate {
            gate.acquire().await.unwrap().forget();
        }
        if script.fail > 0 {
            self.scripts.lock().get_mut(&task).unwrap().fail -= 1;
            if script.panic {
                panic!("scripted panic in {task}");
            }
            return Err(LifecycleError::Record(format!("scripted failure in {task}")));
        }
        match task {
            Task::Provision => {
                self.transition(namespace, &[NamespaceState::Creating], NamespaceState::Created)?;
            }
            Task::Teardown => {
                self.states.lock().remove(namespace);
            }
            // A job's body: nothing to do, or to undo when cancelled.
            Task::Job(_) => {}
        }
        Ok(())
    }
}
