#!/usr/bin/env bash
# test_embedding.sh -- smoke-test the embedding service (batch interface)
#
# Replaces the old test_embedding_single.sh + test_embedding_multiple.sh pair.
#
# The embedding service no longer chunks text — chunking/tokenisation now lives
# in minnal.  The service contract is a plain batch embed:
#
#   Request body : {"payloads": ["text1", "text2", ...], "dimensions": 768}
#   Response     : {"embeddings": [[<float>, ...], ...]}   # one vector per input
#
# The URL paths (the {model} segment from the old API is gone — the model is
# fixed server-side):
#   POST {base}/embedding/document
#   POST {base}/embedding/query
#
# This script:
#   1. Reads URL / model / dimension from the TOML config.
#   2. GETs /healthcheck to confirm the service is reachable.
#   3. Sends a multi-payload document batch and a multi-payload query batch
#      using the new request/response shape, then prints per-vector stats,
#      an L2-normalisation check, and a query×doc cosine-similarity matrix.
#
# Until the in-tree chunker lands, the payload arrays here are produced with a
# crude split (document → sentences, query → words) purely so the batch path is
# exercised with more than one string. This is a smoke test, not the real chunker.
#
# Config is read from config/sample.toml (relative to the workspace root) unless
# overridden by the first argument or the MINNAL_CONFIG env var. The extracted
# values (URL, model, dimension) can each be further overridden by environment
# variables (EMBED_URL, MODEL, DIM).
#
# Usage:
#   ./test_embedding.sh [CONFIG_FILE] [DOC_TEXT] [QUERY_TEXT]
#
# Examples:
#   # Use all defaults (config/sample.toml + built-in sample texts)
#   ./test_embedding.sh
#
#   # Custom config file
#   ./test_embedding.sh /path/to/minnal.toml
#
#   # Custom texts
#   ./test_embedding.sh config/sample.toml \
#       "First sentence. Second sentence. Third sentence." \
#       "query one two three"
#
#   # Override individual settings via env
#   EMBED_URL=http://localhost:8001 MODEL=qwen DIM=768 ./test_embedding.sh
#
# Requirements:
#   curl    -- HTTP calls
#   jq      -- JSON request building + response parsing  (apt install jq / brew install jq)
#   python3 -- similarity maths

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

# Prefer minnal.toml alongside the script (work/bin layout); fall back to
# config/sample.toml relative to the workspace root (service/scripts layout).
if [[ -f "$SCRIPT_DIR/minnal.toml" ]]; then
    _default_config="$SCRIPT_DIR/minnal.toml"
else
    _default_config="$WORKSPACE_ROOT/config/sample.toml"
fi
CONFIG_FILE="${MINNAL_CONFIG:-${1:-$_default_config}}"

# Apostrophes in default values inside ${:-} cause bash to open an unclosed
# single-quote context, so set the defaults in plain variables first.
_default_doc="The complaint, which was filed Tuesday morning (US time) in the US District Court for the Southern District of New York, accuses Meta and Zuckerberg of illegally using millions of copyrighted works to train their artificial intelligence program Llama. The lawsuit asserts that Meta's engineers relied on pirated books and journal articles to train the program. The suit also claims that Zuckerberg himself personally authorised and actively encouraged the infringement."
_default_query="Who filed the complaint and when? What is Llama and why is it mentioned?"

DOC_TEXT="${2:-$_default_doc}"
QUERY_TEXT="${3:-$_default_query}"

# ---------------------------------------------------------------------------
# TOML config parsing
#
# Extracts a bare (unquoted) value for KEY from FILE, ignoring comment lines.
# Falls back to DEFAULT when the key is absent or commented out.
# ---------------------------------------------------------------------------

toml_get() {
    local file="$1" key="$2" default="$3"
    local val
    val=$(grep -v '^\s*#' "$file" 2>/dev/null \
          | grep -E "^\s*${key}\s*=" \
          | head -1 \
          | sed -E 's/^\s*[^=]+=\s*//' \
          | sed -E 's/#.*//' \
          | tr -d "\"'" \
          | tr -d '[:space:]')
    printf '%s' "${val:-$default}"
}

# ---------------------------------------------------------------------------
# Read config
# ---------------------------------------------------------------------------

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: config file not found: $CONFIG_FILE" >&2
    echo "       Pass the path as the first argument or set MINNAL_CONFIG." >&2
    exit 1
fi

EMBED_URL="${EMBED_URL:-$(toml_get "$CONFIG_FILE" embedding_service_url "http://localhost:8001")}"
MODEL="${MODEL:-$(toml_get "$CONFIG_FILE" model "qwen")}"
DIM="${DIM:-$(toml_get "$CONFIG_FILE" dimension "768")}"

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

if ! command -v curl &>/dev/null; then
    echo "ERROR: curl is required but not found in PATH." >&2
    exit 1
fi

if ! command -v jq &>/dev/null; then
    echo "ERROR: jq is required but not found in PATH." >&2
    echo "       Install with: apt install jq  or  brew install jq" >&2
    exit 1
fi

if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 is required but not found in PATH." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

header() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
field()  { printf '  %-24s %s\n' "$1" "$2"; }

# Build a {payloads:[...], dimensions:N} body.
# Usage: build_body MODE TEXT       (MODE = document | query)
#   document → split TEXT on sentence boundaries (. ! ?)
#   query    → split TEXT on whitespace (words)
# These crude splits stand in for the real chunker until it lands.
build_body() {
    local mode="$1" text="$2"
    if [[ "$mode" == "document" ]]; then
        jq -cn --arg t "$text" --argjson d "$DIM" \
            '{payloads: ($t | [splits("(?<=[.!?])\\s+")] | map(gsub("^\\s+|\\s+$";"")) | map(select(length > 0))), dimensions: $d}'
    else
        jq -cn --arg t "$text" --argjson d "$DIM" \
            '{payloads: ($t | [splits("\\s+")] | map(select(length > 0))), dimensions: $d}'
    fi
}

# POST a batch body and write the response to a file. On a non-200 status this
# prints the response body and exits. The HTTP status is stored in the global
# LAST_HTTP (returning it via stdout would swallow error diagnostics, since the
# error branch's output would be captured by the caller's command substitution).
post_batch() {
    local endpoint="$1" body="$2" outfile="$3" label="$4"
    LAST_HTTP=$(curl -sS \
        -o "$outfile" \
        -w '%{http_code}' \
        --max-time 30 \
        -X POST \
        -H "Content-Type: application/json" \
        -d "$body" \
        "$endpoint")
    if [[ "$LAST_HTTP" != "200" ]]; then
        printf '  \033[1;31mERROR\033[0m %s returned HTTP %s:\n' "$label" "$LAST_HTTP"
        cat "$outfile"
        printf '\n'
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

printf '\033[1m=== Embedding Service Smoke Test (batch interface) ===\033[0m\n'
field "config:"      "$CONFIG_FILE"
field "service URL:" "$EMBED_URL"
field "model:"       "$MODEL"
field "dimension:"   "$DIM"
field "doc text:"    "${DOC_TEXT:0:80}..."
field "query text:"  "${QUERY_TEXT:0:80}..."

# ---------------------------------------------------------------------------
# Healthcheck
# ---------------------------------------------------------------------------

header "Healthcheck  GET /healthcheck"
HEALTH_STATUS=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 \
    "$EMBED_URL/healthcheck" || true)
if [[ "$HEALTH_STATUS" == "200" ]]; then
    field "status:" "OK (HTTP 200)"
else
    field "status:" "HTTP $HEALTH_STATUS  <-- service may be unavailable"
    printf '  \033[1;33mWARN\033[0m continuing anyway; embedding calls may fail.\n'
fi

# ---------------------------------------------------------------------------
# Temp files
# ---------------------------------------------------------------------------

DOC_RESP=$(mktemp)
QUERY_RESP=$(mktemp)
trap 'rm -f "$DOC_RESP" "$QUERY_RESP"' EXIT

# ---------------------------------------------------------------------------
# Document embedding  (batch: one vector per sentence)
# ---------------------------------------------------------------------------

DOC_ENDPOINT="${EMBED_URL}/embedding/document"
DOC_BODY=$(build_body document "$DOC_TEXT")

header "Document Embedding  POST .../embedding/document"
printf '  payloads: %s\n' "$(echo "$DOC_BODY" | jq '.payloads | length')"
post_batch "$DOC_ENDPOINT" "$DOC_BODY" "$DOC_RESP" "document embedding"
printf '  HTTP status: %s\n' "$LAST_HTTP"

# ---------------------------------------------------------------------------
# Query embedding  (batch: one vector per word)
# ---------------------------------------------------------------------------

QUERY_ENDPOINT="${EMBED_URL}/embedding/query"
QUERY_BODY=$(build_body query "$QUERY_TEXT")

header "Query Embedding  POST .../embedding/query"
printf '  payloads: %s\n' "$(echo "$QUERY_BODY" | jq '.payloads | length')"
post_batch "$QUERY_ENDPOINT" "$QUERY_BODY" "$QUERY_RESP" "query embedding"
printf '  HTTP status: %s\n' "$LAST_HTTP"

# ---------------------------------------------------------------------------
# Analysis: per-vector summary + L2 check + n x m similarity matrix
# ---------------------------------------------------------------------------

header "Analysis"

python3 - "$DOC_RESP" "$QUERY_RESP" "$DIM" <<'PYEOF'
import sys, json, math

with open(sys.argv[1]) as fh:
    doc_data = json.load(fh)
with open(sys.argv[2]) as fh:
    qry_data = json.load(fh)

expected_dim = int(sys.argv[3])
tol = 1e-3

def l2_norm(vec):
    return math.sqrt(sum(x * x for x in vec))

def cosine(a, b):
    if len(a) != len(b):
        return None
    d = sum(x * y for x, y in zip(a, b))
    na, nb = l2_norm(a), l2_norm(b)
    return d / (na * nb) if na and nb else 0.0

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def label(c):
    if c is None:   return "dim-mismatch"
    if c >= 0.85:   return "very similar"
    if c >= 0.65:   return "moderately similar"
    if c >= 0.40:   return "loosely related"
    return "dissimilar"

def summarise(data, title):
    embeddings = data.get("embeddings", [])
    dim        = len(embeddings[0]) if embeddings else 0

    print(f"\n  -- {title} --")
    print(f"  Embeddings    : {len(embeddings)}  (each {dim}-dimensional, expected {expected_dim})")

    if not embeddings:
        print(f"  FAIL: response contained no embeddings (expected key 'embeddings').")
        return False

    if dim != expected_dim:
        print(f"  WARN: dimension mismatch -- got {dim}, expected {expected_dim}")

    print()
    print("  -- Per-vector preview (first 6 values) " + "-" * 26)
    for i, vec in enumerate(embeddings):
        preview = ", ".join(f"{v:.6f}" for v in vec[:6])
        print(f"    [{i+1:>3}]  [{preview}, ...]")

    print()
    print("  -- L2 normalisation check " + "-" * 40)
    all_pass = True
    for i, vec in enumerate(embeddings):
        norm   = l2_norm(vec)
        status = "PASS" if abs(norm - 1.0) <= tol else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"    [{i+1:>3}]  norm={norm:.8f}  {status}")

    if not all_pass:
        print(f"\n  WARN: one or more {title.lower()} are NOT L2-normalised (tolerance +/-{tol}).")
    else:
        print(f"\n  PASS: all {len(embeddings)} {title.lower()} are L2-normalised (tolerance +/-{tol}).")
    return True

doc_ok = summarise(doc_data, "Document Embeddings")
qry_ok = summarise(qry_data, "Query Embeddings")

doc_embs = doc_data.get("embeddings", [])
qry_embs = qry_data.get("embeddings", [])
m = len(doc_embs)
n = len(qry_embs)

if m and n:
    print(f"\n  -- Similarity matrix  ({n} query chunks x {m} doc chunks) --")
    print(f"  n (query chunks) = {n},  m (doc chunks) = {m},  total pairs = {n * m}")
    print()

    col_w = 26
    hdr = f"  {'query / doc':<18}"
    for j in range(m):
        hdr += f"  {'doc['+str(j)+']':<{col_w}}"
    print(hdr)
    print("  " + "-" * (18 + (col_w + 2) * m))

    for i in range(n):
        row = f"  {'query['+str(i)+']':<18}"
        for j in range(m):
            c    = cosine(qry_embs[i], doc_embs[j])
            cell = f"{c:.4f}  {label(c)}" if c is not None else label(c)
            row += f"  {cell:<{col_w}}"
        print(row)

if not (doc_ok and qry_ok):
    sys.exit(1)
PYEOF

# ---------------------------------------------------------------------------

printf '\n\033[1;32msmoke test complete\033[0m\n\n'
