#!/usr/bin/env bash
# ==============================================================================
# MinnalDB Benchmark Runner
#
# Detects AVX-512 support, builds with the right RUSTFLAGS, runs the full
# Criterion suite, and writes a timestamped report under benchmark/reports/.
#
# Usage:
#   ./benchmark/run_benchmarks.sh [OPTIONS]
#
# Options:
#   --filter <REGEX>   Only run benchmarks whose names match REGEX
#   --baseline <NAME>  Save results as a named Criterion baseline
#   --compare <NAME>   Compare against a previously saved baseline
#   --no-avx512        Force scalar build even if AVX-512 is detected
#   --quick            Shorter measurement time (useful for smoke tests)
#   --help             Show this message
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPORTS_DIR="${SCRIPT_DIR}/reports"
CRITERION_DIR="${PROJECT_ROOT}/target/criterion"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_FILE="${REPORTS_DIR}/bench_${TIMESTAMP}.txt"

# ── Defaults ──────────────────────────────────────────────────────────────────
FILTER=""
BASELINE=""
COMPARE=""
FORCE_SCALAR=false
QUICK=false

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --filter)    FILTER="$2";    shift 2 ;;
        --baseline)  BASELINE="$2";  shift 2 ;;
        --compare)   COMPARE="$2";   shift 2 ;;
        --no-avx512) FORCE_SCALAR=true; shift ;;
        --quick)     QUICK=true;     shift ;;
        --help)
            sed -n '/^# Usage/,/^# ===/p' "$0" | grep -v '^# ===' | sed 's/^# //'
            exit 0
            ;;
        *) echo "Unknown option: $1  (run with --help for usage)"; exit 1 ;;
    esac
done

mkdir -p "${REPORTS_DIR}"

# ── Helpers ───────────────────────────────────────────────────────────────────
hr()   { printf '%0.s─' {1..72}; echo; }
hdr()  { hr; echo "  $*"; hr; }
log()  { echo "  $*"; }
ok()   { echo "  ✓ $*"; }
warn() { echo "  ⚠  $*"; }
die()  { echo "  ✗ $*" >&2; exit 1; }

tee_report() {
    # Write both to stdout and to the report file
    tee -a "${REPORT_FILE}"
}

# ── System information ────────────────────────────────────────────────────────
{
hdr "MinnalDB Benchmark Run — ${TIMESTAMP}"

log "Project  : ${PROJECT_ROOT}"
log "Report   : ${REPORT_FILE}"
echo

log "── System ────────────────────────────────────────────────────────────────"
log "OS       : $(uname -srm)"
if [[ -f /proc/cpuinfo ]]; then
    CPU_MODEL=$(grep -m1 "model name" /proc/cpuinfo | cut -d: -f2 | xargs)
    CPU_CORES=$(grep -c "^processor" /proc/cpuinfo)
    log "CPU      : ${CPU_MODEL}"
    log "Cores    : ${CPU_CORES}"
fi
if command -v free &>/dev/null; then
    MEM_GB=$(free -g | awk '/^Mem:/{print $2}')
    log "Memory   : ${MEM_GB} GiB"
fi
RUSTC_VER=$(rustc --version 2>/dev/null || echo "unknown")
CARGO_VER=$(cargo --version 2>/dev/null || echo "unknown")
log "rustc    : ${RUSTC_VER}"
log "cargo    : ${CARGO_VER}"
echo
} | tee_report

# ── AVX-512 detection ─────────────────────────────────────────────────────────
# Detection runs in the parent shell so the variable survives past the pipe.
AVX512_SUPPORTED=false
AVX512_MSG=""

if [[ "${FORCE_SCALAR}" == "true" ]]; then
    AVX512_MSG="$(warn "AVX-512 disabled by --no-avx512 flag")"
elif [[ "$(uname)" == "Linux" ]]; then
    if grep -q avx512f /proc/cpuinfo 2>/dev/null; then
        AVX512_SUPPORTED=true
        AVX512_MSG="$(ok "AVX-512F detected (avx512f in /proc/cpuinfo)")"
    else
        AVX512_MSG="$(warn "AVX-512F not detected — building with native CPU features only")"
    fi
elif [[ "$(uname)" == "Darwin" ]]; then
    if sysctl hw.optional.avx512f 2>/dev/null | grep -q ": 1"; then
        AVX512_SUPPORTED=true
        AVX512_MSG="$(ok "AVX-512F detected (hw.optional.avx512f)")"
    else
        AVX512_MSG="$(warn "AVX-512F not detected — building with native CPU features only")"
    fi
else
    AVX512_MSG="$(warn "Unknown OS — cannot auto-detect AVX-512; use --no-avx512 to silence")"
fi

{
log "── CPU feature detection ─────────────────────────────────────────────────"
echo "${AVX512_MSG}"
echo
} | tee_report

# ── RUSTFLAGS construction ────────────────────────────────────────────────────
BASE_FLAGS="-C target-cpu=native"
if [[ "${AVX512_SUPPORTED}" == "true" ]]; then
    BENCH_RUSTFLAGS="${BASE_FLAGS} -C target-feature=+avx512f"
else
    BENCH_RUSTFLAGS="${BASE_FLAGS}"
fi

{
log "── Build flags ───────────────────────────────────────────────────────────"
log "RUSTFLAGS: ${BENCH_RUSTFLAGS}"
echo
} | tee_report

# ── Build step ────────────────────────────────────────────────────────────────
{
log "── Building benchmarks (release) ─────────────────────────────────────────"
} | tee_report

cd "${PROJECT_ROOT}"
if ! RUSTFLAGS="${BENCH_RUSTFLAGS}" cargo build --release --benches 2>&1 | tee -a "${REPORT_FILE}"; then
    die "Build failed — see output above"
fi
{ echo; ok "Build successful"; echo; } | tee_report

# ── Criterion arguments ───────────────────────────────────────────────────────
BENCH_ARGS=()

if [[ -n "${FILTER}" ]]; then
    BENCH_ARGS+=("--bench" "minnal_bench" "${FILTER}")
fi

if [[ -n "${BASELINE}" ]]; then
    BENCH_ARGS+=("--" "--save-baseline" "${BASELINE}")
elif [[ -n "${COMPARE}" ]]; then
    BENCH_ARGS+=("--" "--baseline" "${COMPARE}")
fi

# Quick mode: override measurement time via env var (Criterion respects this)
if [[ "${QUICK}" == "true" ]]; then
    BENCH_ARGS+=("--" "--measurement-time" "3")
    { warn "Quick mode: measurement time reduced to 3 s per benchmark"; echo; } | tee_report
fi

# ── Run benchmarks ────────────────────────────────────────────────────────────
{
hdr "Running benchmarks"
} | tee_report

BENCH_START=$(date +%s)

RUSTFLAGS="${BENCH_RUSTFLAGS}" \
    cargo bench \
    --bench minnal_bench \
    "${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"}" \
    2>&1 | grep -v "^Gnuplot not found" | tee -a "${REPORT_FILE}"

BENCH_END=$(date +%s)
ELAPSED=$(( BENCH_END - BENCH_START ))

# ── Summary ───────────────────────────────────────────────────────────────────
{
echo
hdr "Results summary"

log "Total wall-clock time : $((ELAPSED / 60))m $((ELAPSED % 60))s"
echo

# Parse the Criterion plain-text output for timing lines and reformat them.
# Criterion lines look like:
#   write/put/value_size/128B   time:   [1.2345 µs 1.2500 µs 1.2678 µs]
# or (with throughput):
#   write/put/value_size/128B   thrpt:  [98.123 MiB/s 99.456 MiB/s 100.79 MiB/s]
#
# We extract the middle estimate (the mean) for each benchmark.

log "── Per-benchmark mean estimates ──────────────────────────────────────────"
printf "  %-52s %s\n" "Benchmark" "Mean (median estimate)"
printf "  %-52s %s\n" "$(printf '%0.s─' {1..52})" "$(printf '%0.s─' {1..24})"

grep -E "^\s+(write|read|scan|mixed|wal)" "${REPORT_FILE}" \
    | grep "time:" \
    | while IFS= read -r line; do
        bench=$(echo "${line}" | awk '{print $1}')
        mean=$(echo "${line}"  | grep -oP '\[.*?\]' | head -1 \
               | grep -oP '[\d.]+ [nµm]s' | sed -n '2p')
        printf "  %-52s %s\n" "${bench}" "${mean:-see report}"
    done

grep -E "^\s+(write|read|scan|mixed|wal)" "${REPORT_FILE}" \
    | grep "thrpt:" \
    | while IFS= read -r line; do
        bench=$(echo "${line}" | awk '{print $1}')
        tput=$(echo "${line}"  | grep -oP '\[.*?\]' | head -1 \
               | grep -oP '[\d.]+ [KMG]?iB/s|[\d.]+ elem/s' | sed -n '2p')
        printf "  %-52s %s\n" "${bench} [throughput]" "${tput:-see report}"
    done

echo
} | tee_report

# ── Report locations ──────────────────────────────────────────────────────────
{
hdr "Report locations"
log "Text report : ${REPORT_FILE}"
if [[ -d "${CRITERION_DIR}" ]]; then
    log "HTML report : ${CRITERION_DIR}/report/index.html"
    log ""
    log "Open in browser:"
    log "  xdg-open '${CRITERION_DIR}/report/index.html'   # Linux"
    log "  open     '${CRITERION_DIR}/report/index.html'   # macOS"
fi
if [[ -n "${BASELINE}" ]]; then
    log ""
    log "Baseline '${BASELINE}' saved.  To compare a future run:"
    log "  $0 --compare ${BASELINE}"
fi
echo
} | tee_report
