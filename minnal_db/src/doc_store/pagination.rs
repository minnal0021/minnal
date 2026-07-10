/// Pagination parameters for query endpoints.
///
/// `page_no` is 1-based; `page_size` controls how many results are returned.
#[derive(Debug, Clone, Copy)]
pub struct Pagination {
    pub page_no: usize,
    pub page_size: usize,
}

impl Default for Pagination {
    fn default() -> Self {
        Self { page_no: 1, page_size: 20 }
    }
}

impl Pagination {
    pub fn new(page_no: usize, page_size: usize) -> Self {
        Self {
            page_no: page_no.max(1),
            page_size: page_size.max(1),
        }
    }

    pub fn offset(self) -> usize {
        (self.page_no - 1) * self.page_size
    }
}

/// A single page of query results with the total match count before pagination.
#[derive(Debug)]
pub struct Page<T> {
    pub results: Vec<T>,
    pub page_no: usize,
    pub page_size: usize,
    /// Total number of matching records across all pages.
    pub total: usize,
}

impl<T> Page<T> {
    /// Slice `all` according to `pagination` and record the pre-slice total.
    pub fn from_vec(all: Vec<T>, pagination: Pagination) -> Self {
        let total = all.len();
        let results = all.into_iter().skip(pagination.offset()).take(pagination.page_size).collect();
        Self {
            results,
            page_no: pagination.page_no,
            page_size: pagination.page_size,
            total,
        }
    }

    /// Wrap a pre-sliced vec when the total is known separately.
    pub fn from_slice(results: Vec<T>, pagination: Pagination, total: usize) -> Self {
        Self {
            results,
            page_no: pagination.page_no,
            page_size: pagination.page_size,
            total,
        }
    }
}

/// A single page of a cursor-paginated scan.
///
/// Unlike [`Page`], this carries no total (an exact count would require a full
/// scan) and no page number. `next_cursor` is the raw key at which the *next*
/// page begins, or `None` when the scan is exhausted. It is opaque key bytes —
/// the transport layer is responsible for encoding it (e.g. hex) for round-trip
/// through a request parameter.
#[derive(Debug)]
pub struct CursorPage<T> {
    pub results: Vec<T>,
    pub next_cursor: Option<Vec<u8>>,
}

impl<T> CursorPage<T> {
    /// Build a page from already-decoded results and the engine's next cursor.
    pub fn new(results: Vec<T>, next_cursor: Option<Vec<u8>>) -> Self {
        Self { results, next_cursor }
    }
}

/// Smallest key strictly greater than every key beginning with `prefix` — the
/// exclusive upper bound for a prefix scan expressed as a range `[prefix, …)`.
///
/// Returns `None` when `prefix` is empty or all `0xFF`: there is no finite
/// exclusive bound, so the scan runs open-ended to the last key.
pub fn prefix_upper_bound(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut end = prefix.to_vec();
    while let Some(&last) = end.last() {
        if last != 0xFF {
            *end.last_mut().unwrap() = last + 1;
            return Some(end);
        }
        end.pop();
    }
    None
}

#[cfg(test)]
mod cursor_tests {
    use super::prefix_upper_bound;

    #[test]
    fn upper_bound_increments_last_byte() {
        assert_eq!(prefix_upper_bound(b"abc"), Some(b"abd".to_vec()));
    }

    #[test]
    fn upper_bound_carries_over_trailing_ff() {
        assert_eq!(prefix_upper_bound(&[b'a', 0xFF, 0xFF]), Some(vec![b'b']));
    }

    #[test]
    fn upper_bound_all_ff_is_open_ended() {
        assert_eq!(prefix_upper_bound(&[0xFF, 0xFF]), None);
    }

    #[test]
    fn upper_bound_empty_is_open_ended() {
        assert_eq!(prefix_upper_bound(b""), None);
    }
}
