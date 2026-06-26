#!/usr/bin/env bash
# Build release binaries and stage everything under ./work/bin.
#
# What this script does:
#   1. Builds minnal_doc_store_api and minnal_tools in release mode.
#   2. Gracefully stops any running minnal_doc_store_api (via stop.sh if present,
#      otherwise inline — waits up to 30 s before aborting).
#   3. Creates ./work/bin/ (if needed) and copies both binaries into it.
#   4. Copies tools/sample_data/ → ./work/sample_data/  (only with -s flag).
#   5. Copies service/embedding_support/qwen/clusters.json → ./work/bin/clusters.bin.
#   6. Generates ./work/bin/minnal.toml from config/sample.toml, rewriting
#      all data paths to use ./work/doc_store as the base directory.
#   7. Generates ./work/bin/stop.sh   — gracefully stops the server.
#   8. Generates ./work/bin/start.sh + run_tool.sh — helper scripts.
#   9. Copies test_embedding.sh → ./work/bin/.
#
# Run from the workspace root:
#   ./service/scripts/release.sh          # skip sample data
#   ./service/scripts/release.sh -s       # also copy sample data

set -euo pipefail

COPY_SAMPLE_DATA=false
while getopts ":s" opt; do
    case $opt in
        s) COPY_SAMPLE_DATA=true ;;
        *) echo "usage: $0 [-s]" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BIN_DIR="${WORKSPACE_ROOT}/work/bin"
CONFIG_SRC="${WORKSPACE_ROOT}/config/sample.toml"
STOP_TIMEOUT=30

cd "${WORKSPACE_ROOT}"

# ── Helper: graceful stop ──────────────────────────────────────────────────────
stop_server() {
    local pid
    # pgrep -x won't match: Linux truncates comm to 15 chars and the binary
    # name is 20 chars long. Use -f to search the full command line instead.
    pid=$(pgrep -f minnal_doc_store_api 2>/dev/null || true)

    if [[ -z "$pid" ]]; then
        echo "  No running minnal_doc_store_api found — nothing to stop."
        return 0
    fi

    echo "  Found minnal_doc_store_api (PID $pid) — sending SIGTERM..."
    kill -TERM "$pid"

    local elapsed=0
    while kill -0 "$pid" 2>/dev/null; do
        if [[ $elapsed -ge $STOP_TIMEOUT ]]; then
            echo ""
            echo "  ERROR: Process $pid did not exit within ${STOP_TIMEOUT}s. Aborting." >&2
            echo "  Run 'kill -9 $pid' manually if you want to force-quit it." >&2
            exit 1
        fi
        printf "\r  Waiting for process to exit... %ds elapsed" "$elapsed"
        sleep 1
        elapsed=$((elapsed + 1))
    done

    echo ""
    echo "  Server stopped after ${elapsed}s."
}

# ── 1. Build ──────────────────────────────────────────────────────────────────
echo "==> [1/9] Building release binaries..."
cargo build --release -p minnal_doc_store_api -p tools
echo "  Build complete."

# ── 2. Gracefully stop any running server ─────────────────────────────────────
echo ""
echo "==> [2/9] Stopping any running server..."
# Prefer the generated stop.sh from a prior release so it matches the server's
# own shutdown logic; fall back to the inline helper on first run.
if [[ -x "${BIN_DIR}/stop.sh" ]]; then
    "${BIN_DIR}/stop.sh"
else
    stop_server
fi

# ── 3. Stage binaries ─────────────────────────────────────────────────────────
echo ""
echo "==> [3/9] Staging binaries → ${BIN_DIR}"
mkdir -p "${BIN_DIR}"
cp target/release/minnal_doc_store_api "${BIN_DIR}/minnal_doc_store_api"
echo "  Copied minnal_doc_store_api"
cp target/release/minnal_tools         "${BIN_DIR}/minnal_tools"
echo "  Copied minnal_tools"

# ── 4. Copy sample data (optional: -s) ───────────────────────────────────────
echo ""
if [[ "${COPY_SAMPLE_DATA}" == true ]]; then
    echo "==> [4/9] Copying sample data → ${WORKSPACE_ROOT}/work/sample_data"
    mkdir -p "${WORKSPACE_ROOT}/work/sample_data"
    cp "${WORKSPACE_ROOT}/tools/sample_data/"* "${WORKSPACE_ROOT}/work/sample_data/"
    echo "  Sample data staged in ./work/sample_data/"
else
    echo "==> [4/9] Skipping sample data (pass -s to copy)"
fi

# ── 5. Copy cluster centroids ─────────────────────────────────────────────────
echo ""
echo "==> [5/9] Copying cluster centroids → ${BIN_DIR}/clusters.bin"
cp "${WORKSPACE_ROOT}/service/embedding_support/qwen/clusters.json" "${BIN_DIR}/clusters.bin"
echo "  Copied clusters.bin"

# ── 6. Generate config ────────────────────────────────────────────────────────
echo ""
echo "==> [6/9] Generating config → ${BIN_DIR}/minnal.toml"
sed \
    -e 's|db_path = "./data/db"|db_path = "./work/doc_store/db"|' \
    -e 's|schema_dir = "./data/schemas"|schema_dir = "./work/doc_store/schemas"|' \
    -e 's|log_dir = "./data/log"|log_dir = "./work/doc_store/log"|' \
    -e 's|cluster_path = "./clusters.json"|cluster_path = "./work/bin/clusters.bin"|' \
    "${CONFIG_SRC}" > "${BIN_DIR}/minnal.toml"
echo "  Written minnal.toml"

# ── 7. Generate stop.sh ───────────────────────────────────────────────────────
echo ""
echo "==> [7/9] Generating stop script → ${BIN_DIR}/stop.sh"
cat > "${BIN_DIR}/stop.sh" << 'STOP_EOF'
#!/usr/bin/env bash
# Gracefully stop the minnal doc store server.
# Sends SIGTERM and waits up to 30 s for the process to exit.
set -euo pipefail

STOP_TIMEOUT=30

pid=$(pgrep -f minnal_doc_store_api 2>/dev/null || true)

if [[ -z "$pid" ]]; then
    echo "minnal_doc_store_api is not running."
    exit 0
fi

echo "Found minnal_doc_store_api (PID $pid) — sending SIGTERM..."
kill -TERM "$pid"

elapsed=0
while kill -0 "$pid" 2>/dev/null; do
    if [[ $elapsed -ge $STOP_TIMEOUT ]]; then
        echo ""
        echo "ERROR: Process $pid did not exit within ${STOP_TIMEOUT}s." >&2
        echo "Run 'kill -9 $pid' to force-quit it." >&2
        exit 1
    fi
    printf "\rWaiting for server to shut down... %ds elapsed" "$elapsed"
    sleep 1
    elapsed=$((elapsed + 1))
done

echo ""
echo "Server stopped after ${elapsed}s."
STOP_EOF
chmod +x "${BIN_DIR}/stop.sh"
echo "  Written stop.sh"

# ── 8. Generate start.sh ──────────────────────────────────────────────────────
echo ""
echo "==> [8/9] Generating helper scripts → ${BIN_DIR}"
cat > "${BIN_DIR}/start.sh" << 'EOF'
#!/usr/bin/env bash
# Start the minnal doc store using the binaries and config in this directory.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${WORKSPACE_ROOT}"
exec "${SCRIPT_DIR}/minnal_doc_store_api" "${SCRIPT_DIR}/minnal.toml"
EOF
chmod +x "${BIN_DIR}/start.sh"
echo "  Written start.sh"

cat > "${BIN_DIR}/run_tool.sh" << 'EOF'
#!/usr/bin/env bash
# Run a minnal tool using the binary in this directory.
#
# Usage:
#   ./run_tool.sh <tool> [tool-args...]
#
# Examples:
#   ./run_tool.sh bulk_load --schema jobs-mini-schema.json http://localhost:8080 jobs jobId jobs-mini.jsonl
#   ./run_tool.sh bulk_load --kv --schema job-content-kv-schema.json http://localhost:8080 job-content key value job-content-kv.jsonl
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <tool> [args...]" >&2
    echo "" >&2
    echo "tools:" >&2
    echo "  bulk_load    load a JSONL file into a doc store (default) or KV store (--kv)," >&2
    echo "               optionally importing its schema first" >&2
    echo "" >&2
    echo "examples:" >&2
    echo "  $0 bulk_load --schema jobs-mini-schema.json http://localhost:8080 jobs jobId jobs-mini.jsonl" >&2
    echo "  $0 bulk_load --kv --schema job-content-kv-schema.json http://localhost:8080 job-content key value job-content-kv.jsonl" >&2
    exit 1
fi

cd "${WORKSPACE_ROOT}"
exec "${SCRIPT_DIR}/minnal_tools" "$@"
EOF
chmod +x "${BIN_DIR}/run_tool.sh"
echo "  Written run_tool.sh"

# ── 9. Copy test script ───────────────────────────────────────────────────────
echo ""
echo "==> [9/10] Copying test script → ${BIN_DIR}"
cp "${SCRIPT_DIR}/test_embedding.sh" "${BIN_DIR}/test_embedding.sh"
chmod +x "${BIN_DIR}/test_embedding.sh"
echo "  Written test_embedding.sh"

# ── 10. Done ──────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Release ready in ${BIN_DIR}:"
echo "  minnal_doc_store_api      — server binary"
echo "  minnal_tools              — tools binary"
echo "  clusters.bin              — ANN cluster centroids"
echo "  minnal.toml               — server config"
echo "  stop.sh                   — gracefully stop the server"
echo "  start.sh                  — start the server"
echo "  run_tool.sh               — run a tool"
echo "  test_embedding.sh         — smoke-test the embedding service (batch interface)"
if [[ "${COPY_SAMPLE_DATA}" == true ]]; then
    echo ""
    echo "Sample data staged in ./work/sample_data/"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Stop the server:        ./work/bin/stop.sh"
echo "Start the server:       ./work/bin/start.sh"
echo "Import + load docs:     ./work/bin/run_tool.sh bulk_load --schema <schema.json> <url> <namespace> <id_field> <data.jsonl>"
echo "Load into a doc store:  ./work/bin/run_tool.sh bulk_load <url> <namespace> <id_field> <data.jsonl>"
echo "Import + load KV data:  ./work/bin/run_tool.sh bulk_load --kv --schema <schema.json> <url> <namespace> <key_field> <value_field> <data.jsonl>"
echo "Load into a KV store:   ./work/bin/run_tool.sh bulk_load --kv <url> <namespace> <key_field> <value_field> <data.jsonl>"
echo "Test embeddings:        ./work/bin/test_embedding.sh"
