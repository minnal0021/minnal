#!/usr/bin/env bash
# Index management examples
# Covers: add index (background build → 202), drop index
#
# Assumes the 'users' store exists (run stores.sh first, or let this script
# create a fresh one via the setup block below).
#
# Usage:
#   ./indices.sh [BASE_URL]   (default: http://localhost:8080)

set -euo pipefail

BASE_URL="${1:-http://localhost:8080}"
NS="users"

pp() { command -v jq &>/dev/null && jq . || cat; }

echo "=== Setup: ensure '$NS' store exists ==="
curl -s -X POST "$BASE_URL/stores" \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "users",
    "key_type":  "uuid",
    "attributes": [],
    "indices": [
      {"field": "status", "index_type": "str"}
    ]
  }' -o /dev/null || true   # ignore 409 if already present

echo
echo "=== Add a new index on 'verified' (bool) ==="
# Returns 202 Accepted — the index activates immediately for new writes,
# while a background task rebuilds it over existing documents.
curl -sf -X POST "$BASE_URL/stores/$NS/indices" \
  -H "Content-Type: application/json" \
  -d '{"field": "verified", "index_type": "bool"}' | pp
echo "  -> HTTP 202 Accepted (background build in progress)"

echo
echo "=== Add a new index on 'score' (int) ==="
curl -sf -X POST "$BASE_URL/stores/$NS/indices" \
  -H "Content-Type: application/json" \
  -d '{"field": "score", "index_type": "int"}' | pp
echo "  -> HTTP 202 Accepted"

echo
echo "=== List stores to see updated index definitions ==="
curl -sf "$BASE_URL/stores" | pp

echo
echo "=== Drop the 'score' index ==="
# The field is demoted to a non-indexed attribute in the schema;
# stored document data is unchanged.
curl -sf -X DELETE "$BASE_URL/stores/$NS/indices/score" | pp
echo "  -> HTTP 204 No Content"

echo
echo "=== List stores to confirm 'score' is now a plain attribute ==="
curl -sf "$BASE_URL/stores" | pp

echo
echo "--- Error cases ---"

echo
echo "=== Add an index that already exists (expect 409 Conflict) ==="
curl -s -o /dev/null -w "HTTP %{http_code}\n" \
  -X POST "$BASE_URL/stores/$NS/indices" \
  -H "Content-Type: application/json" \
  -d '{"field": "status", "index_type": "str"}'

echo
echo "=== Drop an index that does not exist (expect 404 Not Found) ==="
curl -s -o /dev/null -w "HTTP %{http_code}\n" \
  -X DELETE "$BASE_URL/stores/$NS/indices/no_such_field"
