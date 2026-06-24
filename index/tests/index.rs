use index::field::{FieldIndex, Predicate};

// ── FieldIndex — point predicates ────────────────────────────────────────────

#[test]
fn eq_single_match() {
    let mut idx = FieldIndex::<i64>::new();
    idx.insert(10, 1);
    idx.insert(20, 2);
    idx.insert(10, 3);

    let rows: Vec<u128> = idx.evaluate(&Predicate::Eq(10)).iter().collect();
    assert_eq!(rows, vec![1, 3]);
}

#[test]
fn eq_no_match() {
    let mut idx = FieldIndex::<i64>::new();
    idx.insert(10, 1);

    assert!(idx.evaluate(&Predicate::Eq(999)).is_empty());
}

#[test]
fn ne_excludes_one_value() {
    let mut idx = FieldIndex::<i64>::new();
    for v in [10i64, 20, 30] {
        idx.insert(v, v as u128);
    }

    let rows: Vec<u128> = idx.evaluate(&Predicate::Ne(20)).iter().collect();
    assert_eq!(rows, vec![10, 30]);
}

#[test]
fn ne_no_indexed_value_returns_all() {
    let mut idx = FieldIndex::<i64>::new();
    for v in [10i64, 20, 30] {
        idx.insert(v, v as u128);
    }

    // Excluding a value not in the index returns all rows.
    let rows: Vec<u128> = idx.evaluate(&Predicate::Ne(999)).iter().collect();
    assert_eq!(rows, vec![10, 20, 30]);
}

// ── FieldIndex — range predicates ────────────────────────────────────────────

#[test]
fn lt_strict() {
    let mut idx = FieldIndex::<i64>::new();
    for v in [10i64, 20, 30, 40, 50] {
        idx.insert(v, v as u128);
    }

    let rows: Vec<u128> = idx.evaluate(&Predicate::Lt(30)).iter().collect();
    assert_eq!(rows, vec![10, 20]);
}

#[test]
fn le_inclusive() {
    let mut idx = FieldIndex::<i64>::new();
    for v in [10i64, 20, 30, 40] {
        idx.insert(v, v as u128);
    }

    let rows: Vec<u128> = idx.evaluate(&Predicate::Le(30)).iter().collect();
    assert_eq!(rows, vec![10, 20, 30]);
}

#[test]
fn gt_strict() {
    let mut idx = FieldIndex::<i64>::new();
    for v in [10i64, 20, 30, 40, 50] {
        idx.insert(v, v as u128);
    }

    let rows: Vec<u128> = idx.evaluate(&Predicate::Gt(30)).iter().collect();
    assert_eq!(rows, vec![40, 50]);
}

#[test]
fn ge_inclusive() {
    let mut idx = FieldIndex::<i64>::new();
    for v in [10i64, 20, 30, 40] {
        idx.insert(v, v as u128);
    }

    let rows: Vec<u128> = idx.evaluate(&Predicate::Ge(30)).iter().collect();
    assert_eq!(rows, vec![30, 40]);
}

#[test]
fn between_inclusive_both_bounds() {
    let mut idx = FieldIndex::<i64>::new();
    for v in [10i64, 20, 30, 40, 50] {
        idx.insert(v, v as u128);
    }

    let rows: Vec<u128> = idx.evaluate(&Predicate::Between { lo: 20, hi: 40 }).iter().collect();
    assert_eq!(rows, vec![20, 30, 40]);
}

#[test]
fn between_empty_range() {
    let mut idx = FieldIndex::<i64>::new();
    for v in [10i64, 50] {
        idx.insert(v, v as u128);
    }

    // No values in [20, 40]
    assert!(idx.evaluate(&Predicate::Between { lo: 20, hi: 40 }).is_empty());
}

// ── FieldIndex — In predicate ─────────────────────────────────────────────────

#[test]
fn in_returns_union_of_matches() {
    let mut idx = FieldIndex::<i64>::new();
    idx.insert(10, 1);
    idx.insert(20, 2);
    idx.insert(30, 3);
    idx.insert(10, 4);

    let rows: Vec<u128> = idx.evaluate(&Predicate::In(vec![10, 30, 99])).iter().collect();
    assert_eq!(rows, vec![1, 3, 4]);
}

// ── FieldIndex — String keys ──────────────────────────────────────────────────

#[test]
fn string_keys_eq_and_lt() {
    let mut idx = FieldIndex::<String>::new();
    idx.insert("banana".into(), 1);
    idx.insert("apple".into(), 2);
    idx.insert("cherry".into(), 3);
    idx.insert("apple".into(), 4);

    let eq_rows: Vec<u128> = idx.evaluate(&Predicate::Eq("apple".into())).iter().collect();
    assert_eq!(eq_rows, vec![2, 4]);

    // "apple" < "banana" lexicographically
    let lt_rows: Vec<u128> = idx.evaluate(&Predicate::Lt("banana".into())).iter().collect();
    assert_eq!(lt_rows, vec![2, 4]);
}

// ── FieldIndex — remove ───────────────────────────────────────────────────────

#[test]
fn remove_updates_evaluate() {
    let mut idx = FieldIndex::<i64>::new();
    idx.insert(10, 1);
    idx.insert(10, 2);
    idx.insert(20, 3);

    idx.remove(10, 1);

    let rows: Vec<u128> = idx.evaluate(&Predicate::Eq(10)).iter().collect();
    assert_eq!(rows, vec![2]);
    assert_eq!(idx.distinct_count(), 2);
}

#[test]
fn remove_last_row_cleans_up_entry() {
    let mut idx = FieldIndex::<i64>::new();
    idx.insert(42, 7);

    idx.remove(42, 7);

    assert_eq!(idx.distinct_count(), 0);
    assert!(idx.evaluate(&Predicate::Eq(42)).is_empty());
}
