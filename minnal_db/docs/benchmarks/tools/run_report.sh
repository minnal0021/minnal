#!/usr/bin/env bash
#
# Regenerate everything behind ../../../benchmark.md: run every benchmark suite,
# extract Criterion's means, and re-render the charts in ../.
#
#   minnal_db/docs/benchmarks/tools/run_report.sh
#
# Takes roughly 95 minutes on an idle machine. Run it on an otherwise quiet
# host: several suites measure sub-microsecond operations where a busy machine
# shows up as a 20-40% swing (see benchmark.md's closing section).
#
# Any previous target/criterion is moved aside rather than deleted, so Criterion
# does not report cross-run "regressions" against a baseline measured on other
# hardware, and the old numbers stay available for comparison.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
tools_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
charts_dir=$(dirname "$tools_dir")
cd "$repo_root"

stamp=$(date +%Y%m%d_%H%M%S)
out_dir="target/bench_report_$stamp"
mkdir -p "$out_dir"

if [ -d target/criterion ]; then
  echo "==> archiving previous run to target/criterion_$stamp"
  mv target/criterion "target/criterion_$stamp"
fi

echo "==> recording environment"
{
  echo "date:    $(date -Iseconds)"
  echo "commit:  $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD)"
  echo "dirty:   $(git status --porcelain | wc -l) uncommitted file(s)"
  echo "kernel:  $(uname -sr)"
  echo "cpu:     $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | xargs || sysctl -n machdep.cpu.brand_string)"
  echo "rustc:   $(rustc -V)"
  echo "ulimit:  $(ulimit -n) open files"
} | tee "$out_dir/environment.txt"

echo "==> running all benchmark suites (this is the slow part)"
cargo bench -p minnal_db --all-features 2>&1 | tee "$out_dir/bench.log"

echo "==> extracting means"
python3 "$tools_dir/extract.py" target/criterion > "$out_dir/means.tsv"
echo "    $(wc -l < "$out_dir/means.tsv") cases -> $out_dir/means.tsv"

echo "==> rendering charts into $charts_dir"
python3 "$tools_dir/plot.py" "$out_dir/means.tsv" "$charts_dir" "$tools_dir/charts.json"

echo
echo "Done. Update benchmark.md from $out_dir/means.tsv."
