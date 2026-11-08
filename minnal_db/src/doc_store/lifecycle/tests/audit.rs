//! Tests of the lifecycle audit log.

use crate::doc_store::schema::IndexType;
use crate::doc_store::tests::test_support::*;
use crate::namespace_lifecycle::AuditEvent;

/// A store's whole life is in the audit log, newest first, and stays there
/// after the store is gone; one store's events can be read alone, a page at a
/// time.
#[tokio::test]
async fn a_stores_life_is_in_the_audit_log_after_it_is_gone() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;
    store.create(make_schema("a", vec![])).await.unwrap();
    store.create(make_schema("b", vec![])).await.unwrap();
    let spec = IndexSpec {
        field: "status".to_owned(),
        index_type: IndexType::Str,
    };
    store.add_index("a", spec).await.unwrap();
    store.remove("a").await.unwrap();

    let (all, next) = store.lifecycle_audit(None, None, 100).await.unwrap();
    assert!(next.is_none());
    let events: Vec<(&str, AuditEvent)> = all.iter().map(|r| (r.namespace.as_str(), r.event)).collect();
    assert_eq!(
        events,
        vec![
            ("a", AuditEvent::Deleted),
            ("a", AuditEvent::Deleting),
            ("a", AuditEvent::JobCompleted),
            ("a", AuditEvent::JobStarted),
            ("b", AuditEvent::Created),
            ("b", AuditEvent::Creating),
            ("a", AuditEvent::Created),
            ("a", AuditEvent::Creating),
        ]
    );
    let started = &all[3];
    assert_eq!(
        (started.job.as_deref(), started.subject.as_deref()),
        (Some("field_build"), Some("status"))
    );
    assert!(all.windows(2).all(|w| w[0].at_ms >= w[1].at_ms), "newest first");

    // One store, two at a time.
    let (page, next) = store.lifecycle_audit(Some("a"), None, 2).await.unwrap();
    assert_eq!(
        page.iter().map(|r| r.event).collect::<Vec<_>>(),
        vec![AuditEvent::Deleted, AuditEvent::Deleting]
    );
    let (rest, _) = store.lifecycle_audit(Some("a"), next, 100).await.unwrap();
    assert_eq!(rest.len(), 4);
    assert!(rest.iter().all(|r| r.namespace == "a"));

    // It survives a restart.
    drop(store);
    let store = open_fresh(db_dir.path()).await;
    assert_eq!(store.lifecycle_audit(None, None, 100).await.unwrap().0, all);
}
