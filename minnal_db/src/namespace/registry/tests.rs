//! The registry's changes and the catalog records they write, checked against
//! an in-memory catalog: every test applies a registry's writes to a map and
//! loads a new registry from it, as an open would.

use std::collections::BTreeMap;
use std::time::Duration;

use super::*;
use crate::namespace::catalog::CatalogWrite;

/// An in-memory catalog: the writes applied in order.
#[derive(Default)]
struct Catalog(BTreeMap<Vec<u8>, Vec<u8>>);

impl Catalog {
    fn save(&mut self, r: &mut NamespaceRegistry) {
        for w in r.take_writes() {
            match w {
                CatalogWrite::Put { key, value } => {
                    self.0.insert(key, value);
                }
                CatalogWrite::Delete { key } => {
                    self.0.remove(&key);
                }
            }
        }
    }

    fn load(&self) -> NamespaceRegistry {
        NamespaceRegistry::from_catalog(self.0.clone()).unwrap()
    }
}

/// A new database's registry, its first records saved.
fn opened() -> (Catalog, NamespaceRegistry) {
    let mut catalog = Catalog::default();
    let mut r = NamespaceRegistry::from_catalog(Vec::new()).unwrap();
    catalog.save(&mut r);
    (catalog, r)
}

#[test]
fn a_new_registry_has_the_fixed_namespaces_and_saves_them() {
    let (catalog, r) = opened();
    for (name, id) in [
        (DEFAULT_NAMESPACE, DEFAULT_NAMESPACE_ID),
        (SYSTEM_NAMESPACE, SYSTEM_NAMESPACE_ID),
        (CATALOG_NAMESPACE, NamespaceId::CATALOG),
    ] {
        assert_eq!(r.get_id(name), Some(id));
        assert_eq!(r.get_name(id), Some(name));
    }
    let mut reloaded = catalog.load();
    assert!(reloaded.take_writes().is_empty(), "nothing left to repair");
    assert_eq!(reloaded.get_id(DEFAULT_NAMESPACE), Some(DEFAULT_NAMESPACE_ID));
    assert_eq!(reloaded.get_id(CATALOG_NAMESPACE), Some(NamespaceId::CATALOG));
}

#[test]
fn ids_come_from_one_counter_for_namespaces_and_fields() {
    let (mut catalog, mut r) = opened();
    let users = r.create("users").unwrap();
    let status = r.register_schema_field(users, "status", IndexValueType::Str).unwrap();
    let orders = r.create("orders").unwrap();
    assert_eq!(users.as_u64(), NamespaceId::FIRST_ALLOCATED);
    assert_eq!(status, NamespaceId::FIRST_ALLOCATED + 1);
    assert_eq!(orders.as_u64(), NamespaceId::FIRST_ALLOCATED + 2);
    catalog.save(&mut r);
    // The counter is saved: a reopened registry carries on above every id.
    let mut reloaded = catalog.load();
    assert_eq!(reloaded.create("next").unwrap().as_u64(), NamespaceId::FIRST_ALLOCATED + 3);
}

#[test]
fn the_counter_is_saved_before_the_record_that_uses_its_id() {
    let (_, mut r) = opened();
    r.create("users").unwrap();
    let writes = r.take_writes();
    assert_eq!(writes.len(), 2);
    assert!(
        matches!(&writes[0], CatalogWrite::Put { key, .. } if key == b"next_id"),
        "a crash after the first write leaves a gap, never a reused id: {writes:?}"
    );
}

#[test]
fn a_recreated_name_gets_a_new_id() {
    let (mut catalog, mut r) = opened();
    let first = r.create("ns1").unwrap();
    r.remove("ns1").unwrap();
    let second = r.create("ns1").unwrap();
    assert_ne!(first, second);
    catalog.save(&mut r);
    assert_eq!(catalog.load().get_id("ns1"), Some(second));
}

#[test]
fn a_removed_namespace_takes_its_ttl_and_field_records_with_it() {
    let (mut catalog, mut r) = opened();
    let cache = r.create("cache").unwrap();
    r.set_ttl_config(cache, Duration::from_secs(60), 10).unwrap();
    r.register_schema_field(cache, "status", IndexValueType::Str).unwrap();
    catalog.save(&mut r);
    r.remove("cache").unwrap();
    assert_eq!(r.ttl_config(cache), None);
    catalog.save(&mut r);
    let reloaded = catalog.load();
    assert!(!reloaded.exists("cache"));
    assert!(
        !catalog.0.keys().any(|k| k.starts_with(b"field/")),
        "no field record is left: {:?}",
        catalog.0.keys().map(|k| String::from_utf8_lossy(k).into_owned()).collect::<Vec<_>>()
    );
}

#[test]
fn ttl_settings_are_saved_and_removed() {
    let (mut catalog, mut r) = opened();
    let cache = r.create("cache").unwrap();
    r.set_ttl_config(cache, Duration::from_secs(3600), 500).unwrap();
    catalog.save(&mut r);
    assert_eq!(catalog.load().ttl_config(cache), Some((Duration::from_secs(3600), 500)));
    r.remove_ttl_config(cache).unwrap();
    catalog.save(&mut r);
    assert_eq!(catalog.load().ttl_config(cache), None);
    assert!(
        r.set_ttl_config(NamespaceId::from_u64(99_999), Duration::from_secs(1), 1).is_err(),
        "an unknown namespace"
    );
}

#[test]
fn fields_and_their_dropped_flag_are_saved() {
    let (mut catalog, mut r) = opened();
    let users = r.create("users").unwrap();
    let status = r.register_schema_field(users, "status", IndexValueType::Str).unwrap();
    let age = r.register_schema_field(users, "age", IndexValueType::Int).unwrap();
    assert!(r.mark_schema_field_dropped(users, age).unwrap());
    catalog.save(&mut r);

    let mut reloaded = catalog.load();
    let schema = reloaded.schema(users).unwrap();
    assert_eq!(schema.get_field_by_name("status").unwrap().field_id, status);
    assert!(!schema.get_field_by_name("status").unwrap().dropped);
    assert!(schema.get_field_by_name("age").unwrap().dropped);
    assert_eq!(reloaded.all_indexed_fields(), vec![(users, status)]);

    // A dropped field registered again keeps its id, and is live again.
    assert_eq!(reloaded.register_schema_field(users, "age", IndexValueType::Int).unwrap(), age);
    catalog.save(&mut reloaded);
    assert!(!catalog.load().schema(users).unwrap().get_field_by_name("age").unwrap().dropped);
}

#[test]
fn registering_a_live_field_again_writes_nothing() {
    let (_, mut r) = opened();
    let users = r.create("users").unwrap();
    let id = r.register_schema_field(users, "status", IndexValueType::Str).unwrap();
    r.take_writes();
    assert_eq!(r.register_schema_field(users, "status", IndexValueType::Str).unwrap(), id);
    assert!(r.take_writes().is_empty());
    assert!(r.register_schema_field(users, "status", IndexValueType::Int).is_err(), "another type");
    assert!(!r.mark_schema_field_dropped(users, 12345).unwrap(), "no such field");
}

#[test]
fn names_that_cannot_be_a_directory_or_are_taken_are_refused() {
    let (_, mut r) = opened();
    r.create(&"x".repeat(crate::layout::MAX_NAMESPACE_NAME_BYTES)).unwrap();
    for bad in [
        "x".repeat(crate::layout::MAX_NAMESPACE_NAME_BYTES + 1),
        String::new(),
        "a/b".to_owned(),
        "../up".to_owned(),
        CATALOG_NAMESPACE.to_owned(),
        DEFAULT_NAMESPACE.to_owned(),
    ] {
        assert!(r.create(&bad).is_err(), "{bad:?}");
    }
}

#[test]
fn the_fixed_namespaces_cannot_be_removed() {
    let (_, mut r) = opened();
    for name in [DEFAULT_NAMESPACE, SYSTEM_NAMESPACE, CATALOG_NAMESPACE] {
        assert!(r.remove(name).is_err(), "{name}");
    }
}

/// A failed create once left such an entry, and opening it failed every later
/// start of the database.
#[test]
fn a_record_that_cannot_be_a_directory_is_removed_at_open() {
    let (mut catalog, mut r) = opened();
    let kept = r.create("kept").unwrap();
    catalog.save(&mut r);
    let too_long = "x".repeat(300);
    let CatalogWrite::Put { key, value } = crate::namespace::catalog::namespace_write(&too_long, NamespaceId::from_u64(5000), None) else {
        unreachable!()
    };
    catalog.0.insert(key, value);

    let mut reloaded = catalog.load();
    assert!(!reloaded.exists(&too_long));
    assert_eq!(reloaded.get_id("kept"), Some(kept));
    catalog.save(&mut reloaded);
    let mut again = catalog.load();
    assert!(again.take_writes().is_empty(), "the record was deleted");
}

#[test]
fn a_counter_below_an_id_in_use_is_moved_above_it() {
    let (mut catalog, mut r) = opened();
    let users = r.create("users").unwrap();
    catalog.save(&mut r);
    catalog.0.insert(b"next_id".to_vec(), 7u64.to_le_bytes().to_vec());
    let mut reloaded = catalog.load();
    assert!(reloaded.create("orders").unwrap() > users);
}
