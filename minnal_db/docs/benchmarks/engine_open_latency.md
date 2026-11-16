# Engine latency at a fixed arrival rate

The figures behind the chart in `benchmark.md` → *Latency at a fixed arrival rate*, drawn by `docs/benchmarks/engine/plot.py`. Operations from the mix arrive at the rate whatever the engine is doing, and each latency runs from the operation's scheduled arrival. Milliseconds. In flight: operations under way at once, the mean (rate × mean latency) and the most. A rate whose operations under way reached the limit (1,024) dropped arrivals; its latencies are only a lower bound and are not given. Every rate from the first such one up is past capacity: near capacity, random bursts of arrivals build queues that drain slowly, so its latencies are given but are not a steady state. Each cell ran once.

| Arrivals per second | 500 | 1,000 | 1,500 | 1,650 | 1,800 | 1,950 |
|---|---:|---:|---:|---:|---:|---:|
| Completed per second | 499 | 1002 | 1502 | 1652 | 1799 | 1946 |
| Mean in flight | 1.61 | 4.43 | 8.02 | 10.2 | 102 | 68.0 |
| Most in flight | 12 | 30 | 42 | 117 | 1,024 | 697 |
| Dispatch lag p99 (ms) | 0.07 | 0.32 | 0.58 | 0.69 | 3.31 | 1.62 |
| Arrivals dropped | 0 | 0 | 0 | 0 | 35 | 0 |
| Past capacity | no | no | no | no | **yes** | **yes** |

**`get`** (ms)

| Arrivals per second | p50 | p75 | p90 | p95 | p99 | p99.9 |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 0.08 | 0.09 | 0.10 | 0.10 | 0.13 | 0.68 |
| 1,000 | 0.09 | 0.10 | 0.11 | 0.13 | 0.48 | 1.16 |
| 1,500 | 0.09 | 0.11 | 0.15 | 0.31 | 0.74 | 1.48 |
| 1,650 | 0.10 | 0.11 | 0.23 | 0.41 | 0.93 | 2.14 |
| 1,800 (dropped arrivals) | — | — | — | — | — | — |
| 1,950 (past capacity) | 0.10 | 0.19 | 0.69 | 2.19 | 20.3 | 31.8 |

**`scan`** (ms)

| Arrivals per second | p50 | p75 | p90 | p95 | p99 | p99.9 |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 0.36 | 0.42 | 0.51 | 0.59 | 0.93 | 2.03 |
| 1,000 | 0.46 | 0.61 | 0.88 | 1.15 | 2.08 | 3.60 |
| 1,500 | 0.62 | 0.97 | 1.75 | 2.62 | 5.26 | 9.91 |
| 1,650 | 0.59 | 1.06 | 2.36 | 3.90 | 9.01 | 21.5 |
| 1,800 (dropped arrivals) | — | — | — | — | — | — |
| 1,950 (past capacity) | 0.87 | 3.47 | 49.0 | 119 | 203 | 285 |

**`field_query`** (ms)

| Arrivals per second | p50 | p75 | p90 | p95 | p99 | p99.9 |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 2.66 | 6.04 | 8.92 | 10.2 | 12.6 | 19.9 |
| 1,000 | 3.34 | 7.25 | 10.4 | 12.0 | 15.8 | 28.3 |
| 1,500 | 4.38 | 9.22 | 13.1 | 15.5 | 21.4 | 36.2 |
| 1,650 | 4.58 | 9.19 | 13.3 | 16.2 | 26.2 | 42.9 |
| 1,800 (dropped arrivals) | — | — | — | — | — | — |
| 1,950 (past capacity) | 7.28 | 14.9 | 101 | 213 | 334 | 438 |

**`search`** (ms)

| Arrivals per second | p50 | p75 | p90 | p95 | p99 | p99.9 |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 4.98 | 5.59 | 6.33 | 6.84 | 8.11 | 9.84 |
| 1,000 | 7.57 | 8.79 | 10.4 | 11.7 | 15.3 | 23.0 |
| 1,500 | 8.28 | 10.8 | 15.1 | 19.0 | 28.3 | 45.0 |
| 1,650 | 9.47 | 13.4 | 20.5 | 27.7 | 51.7 | 98.4 |
| 1,800 (dropped arrivals) | — | — | — | — | — | — |
| 1,950 (past capacity) | 11.2 | 25.7 | 356 | 649 | 873 | 1053 |

**`filtered_search`** (ms)

| Arrivals per second | p50 | p75 | p90 | p95 | p99 | p99.9 |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 6.29 | 7.63 | 8.74 | 9.43 | 10.9 | 14.7 |
| 1,000 | 9.24 | 11.1 | 13.2 | 14.8 | 19.4 | 32.2 |
| 1,500 | 10.3 | 13.5 | 18.3 | 22.2 | 31.9 | 49.5 |
| 1,650 | 11.5 | 16.1 | 23.8 | 31.1 | 55.6 | 106 |
| 1,800 (dropped arrivals) | — | — | — | — | — | — |
| 1,950 (past capacity) | 13.9 | 29.5 | 362 | 664 | 880 | 1055 |

**`insert`** (ms)

| Arrivals per second | p50 | p75 | p90 | p95 | p99 | p99.9 |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 2.23 | 2.67 | 3.10 | 3.73 | 8.70 | 26.9 |
| 1,000 | 2.57 | 2.95 | 3.36 | 4.90 | 9.72 | 26.6 |
| 1,500 | 2.81 | 3.16 | 3.90 | 5.49 | 9.88 | 27.2 |
| 1,650 | 2.90 | 3.29 | 4.30 | 5.94 | 9.99 | 27.3 |
| 1,800 (dropped arrivals) | — | — | — | — | — | — |
| 1,950 (past capacity) | 3.10 | 3.90 | 6.52 | 10.4 | 61.7 | 95.7 |

**`update`** (ms)

| Arrivals per second | p50 | p75 | p90 | p95 | p99 | p99.9 |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 2.24 | 2.68 | 3.11 | 3.65 | 8.57 | 24.7 |
| 1,000 | 2.58 | 2.96 | 3.37 | 4.98 | 9.78 | 18.1 |
| 1,500 | 2.82 | 3.18 | 3.94 | 5.69 | 9.74 | 26.6 |
| 1,650 | 2.92 | 3.31 | 4.33 | 6.03 | 9.78 | 28.1 |
| 1,800 (dropped arrivals) | — | — | — | — | — | — |
| 1,950 (past capacity) | 3.12 | 3.89 | 6.46 | 10.5 | 61.7 | 95.7 |

**`delete`** (ms)

| Arrivals per second | p50 | p75 | p90 | p95 | p99 | p99.9 |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 6.25 | 7.81 | 10.5 | 13.1 | 28.2 | 42.8 |
| 1,000 | 7.39 | 9.17 | 12.4 | 16.0 | 30.1 | 46.8 |
| 1,500 | 8.18 | 10.3 | 14.1 | 18.3 | 30.9 | 59.1 |
| 1,650 | 8.53 | 10.8 | 14.8 | 19.0 | 30.5 | 53.7 |
| 1,800 (dropped arrivals) | — | — | — | — | — | — |
| 1,950 (past capacity) | 9.48 | 13.0 | 19.9 | 32.3 | 159 | 282 |
