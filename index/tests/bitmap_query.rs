//! Integration tests for the query-support API added to `RoaringBitmap`:
//! `rank`, `select`, `flip`, `range_and`, `range_or`.

use index::RoaringBitmap;

// ── rank ─────────────────────────────────────────────────────────────────────

#[test]
fn rank_empty_bitmap() {
    let bm = RoaringBitmap::new();
    assert_eq!(bm.rank(0), 0);
    assert_eq!(bm.rank(u128::MAX), 0);
}

#[test]
fn rank_counts_le_value() {
    let bm: RoaringBitmap = [10u128, 20, 30].into_iter().collect();
    assert_eq!(bm.rank(9), 0);
    assert_eq!(bm.rank(10), 1);
    assert_eq!(bm.rank(19), 1);
    assert_eq!(bm.rank(20), 2);
    assert_eq!(bm.rank(30), 3);
    assert_eq!(bm.rank(31), 3);
}

#[test]
fn rank_large_u128_keys() {
    let base: u128 = 0xDEAD_0000_0000_0000_0000_0000_0000_0000;
    let bm: RoaringBitmap = [base, base + 1, base + 100].into_iter().collect();
    assert_eq!(bm.rank(base - 1), 0);
    assert_eq!(bm.rank(base), 1);
    assert_eq!(bm.rank(base + 1), 2);
    assert_eq!(bm.rank(base + 50), 2);
    assert_eq!(bm.rank(base + 100), 3);
}

// ── select ───────────────────────────────────────────────────────────────────

#[test]
fn select_empty_bitmap() {
    let bm = RoaringBitmap::new();
    assert_eq!(bm.select(0), None);
}

#[test]
fn select_returns_nth_element() {
    let bm: RoaringBitmap = [5u128, 15, 25, 35].into_iter().collect();
    assert_eq!(bm.select(0), Some(5));
    assert_eq!(bm.select(1), Some(15));
    assert_eq!(bm.select(2), Some(25));
    assert_eq!(bm.select(3), Some(35));
    assert_eq!(bm.select(4), None);
}

#[test]
fn rank_select_inverse() {
    // For every element e in the bitmap: select(rank(e) - 1) == e
    let vals: Vec<u128> = (0u128..300).map(|i| i * 13).collect();
    let bm = RoaringBitmap::from_sorted_iter(vals.iter().copied());
    for &v in &vals {
        let r = bm.rank(v);
        assert!(r > 0, "rank must be ≥ 1 for present element {v}");
        assert_eq!(bm.select(r - 1), Some(v), "select(rank({v})-1) ≠ {v}");
    }
}

#[test]
fn select_spans_container_boundary() {
    // Put values in two container keys (high 112-bit buckets)
    let bm: RoaringBitmap = [0u128, 1, 0x1_0000, 0x1_0001].into_iter().collect();
    assert_eq!(bm.select(0), Some(0));
    assert_eq!(bm.select(1), Some(1));
    assert_eq!(bm.select(2), Some(0x1_0000));
    assert_eq!(bm.select(3), Some(0x1_0001));
}

// ── flip ─────────────────────────────────────────────────────────────────────

#[test]
fn flip_clears_and_adds_bits() {
    // {1,3,5} flip [0,6) → {0,2,4} (present bits cleared, absent bits set within range)
    let mut bm: RoaringBitmap = [1u128, 3, 5].into_iter().collect();
    bm.flip(0, 6);
    let vals: Vec<u128> = bm.iter().collect();
    assert_eq!(vals, vec![0, 2, 4]);
}

#[test]
fn flip_twice_is_identity() {
    let original: RoaringBitmap = (0u128..50).collect();
    let mut bm = original.clone();
    bm.flip(10, 40);
    bm.flip(10, 40);
    assert_eq!(bm, original);
}

#[test]
fn flip_removes_empty_containers() {
    // A single-element bitmap whose element is flipped out should remove the container.
    let mut bm: RoaringBitmap = [42u128].into_iter().collect();
    bm.flip(42, 43); // flip [42, 43) — removes 42
    assert!(bm.is_empty());
    assert_eq!(bm.num_containers(), 0);
}

#[test]
fn flip_noop_on_empty_or_reversed_range() {
    let mut bm: RoaringBitmap = [1u128, 2, 3].into_iter().collect();
    bm.flip(5, 5); // lo == hi
    bm.flip(5, 4); // lo > hi
    assert_eq!(bm.cardinality(), 3);
}

#[test]
fn flip_only_touches_existing_containers() {
    // Values 1,2 are in container 0; value 0x2_0000 is in container 2.
    // Flipping [0, 0x1_0000) only covers container 0 — container 2 unchanged.
    let mut bm: RoaringBitmap = [1u128, 2, 0x2_0000].into_iter().collect();
    bm.flip(0, 0x1_0000);
    // 1 and 2 are now absent; 0x2_0000 untouched
    assert!(!bm.contains(1));
    assert!(!bm.contains(2));
    assert!(bm.contains(0x2_0000));
}

// ── range_and ────────────────────────────────────────────────────────────────

#[test]
fn range_and_empty_range_returns_empty() {
    let a: RoaringBitmap = (0u128..100).collect();
    let b: RoaringBitmap = (0u128..100).collect();
    assert!(a.range_and(&b, 50, 50).is_empty());
    assert!(a.range_and(&b, 60, 50).is_empty());
}

#[test]
fn range_and_is_subset_of_full_and() {
    let a: RoaringBitmap = (0u128..200).collect();
    let b: RoaringBitmap = (100u128..300).collect();
    let full_and = a.and(&b);
    let ranged = a.range_and(&b, 100, 200);
    assert_eq!(full_and, ranged);
}

#[test]
fn range_and_clips_within_range() {
    let a: RoaringBitmap = (0u128..100).collect();
    let b: RoaringBitmap = (0u128..100).collect();
    let result = a.range_and(&b, 30, 60);
    let expected: RoaringBitmap = (30u128..60).collect();
    assert_eq!(result, expected);
}

#[test]
fn range_and_disjoint_bitmaps_returns_empty() {
    let a: RoaringBitmap = (0u128..50).collect();
    let b: RoaringBitmap = (50u128..100).collect();
    assert!(a.range_and(&b, 0, 100).is_empty());
}

#[test]
fn range_and_cross_container_boundary() {
    // a and b both have values in two container keys
    let a: RoaringBitmap = [5u128, 0x1_0005, 0x1_0010].into_iter().collect();
    let b: RoaringBitmap = [5u128, 0x1_0005, 0x1_0020].into_iter().collect();
    // Restrict to the second container only
    let result = a.range_and(&b, 0x1_0000, 0x2_0000);
    let vals: Vec<u128> = result.iter().collect();
    assert_eq!(vals, vec![0x1_0005]); // 0x1_0010 and 0x1_0020 not in both
}

// ── range_or ─────────────────────────────────────────────────────────────────

#[test]
fn range_or_empty_range_returns_empty() {
    let a: RoaringBitmap = (0u128..100).collect();
    let b: RoaringBitmap = (0u128..100).collect();
    assert!(a.range_or(&b, 50, 50).is_empty());
}

#[test]
fn range_or_clips_union_to_range() {
    let a: RoaringBitmap = (0u128..50).collect();
    let b: RoaringBitmap = (50u128..100).collect();
    let result = a.range_or(&b, 25, 75);
    let expected: RoaringBitmap = (25u128..75).collect();
    assert_eq!(result, expected);
}

#[test]
fn range_or_same_as_full_or_when_range_covers_all() {
    let a: RoaringBitmap = (10u128..50).collect();
    let b: RoaringBitmap = (40u128..80).collect();
    let full_or = a.or(&b);
    let ranged = a.range_or(&b, 10, 80);
    assert_eq!(full_or, ranged);
}

#[test]
fn range_or_cross_container_boundary() {
    let a: RoaringBitmap = [5u128, 0x1_0005].into_iter().collect();
    let b: RoaringBitmap = [0x1_0010u128].into_iter().collect();
    // Only cover the second container
    let result = a.range_or(&b, 0x1_0000, 0x2_0000);
    let vals: Vec<u128> = result.iter().collect();
    assert_eq!(vals, vec![0x1_0005, 0x1_0010]);
}

// ── combined: rank + range_and for pagination ────────────────────────────────

#[test]
fn range_and_then_select_for_pagination() {
    // Simulate: find rows 20..40 that appear in both bitmaps, then page through them.
    let a: RoaringBitmap = (0u128..100).collect();
    let b: RoaringBitmap = (0u128..100).step_by(2).collect();
    let page = a.range_and(&b, 20, 40);
    assert_eq!(page.cardinality(), 10); // even numbers 20,22,..,38
    assert_eq!(page.select(0), Some(20));
    assert_eq!(page.select(9), Some(38));
}
