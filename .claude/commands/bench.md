Run the benchmark specified in $ARGUMENTS and summarise the results.

Benchmarks live in two crates:
- minnal_db:       bench_write, bench_read, bench_scan, bench_mixed, bench_wal, bench_typed
- semantic_search: bench_distance_estimation

Usage examples:
  /bench bench_write
  /bench bench_read
  /bench bench_mixed
  /bench bench_distance_estimation

Steps:
1. Determine the target crate from the bench name:
   - bench_distance_estimation → semantic_search
   - anything else             → minnal_db
2. Run: `cargo bench -p <crate> --bench $ARGUMENTS 2>&1`
3. Parse the criterion output — extract the test names, mean throughput or latency, and any regression/improvement notices (lines with "Performance has regressed" or "Performance has improved").
4. Report a concise table: benchmark name | mean | throughput (if available) | change vs baseline (if available).
5. If the bench name is not found, list all available benches from both crates above.
