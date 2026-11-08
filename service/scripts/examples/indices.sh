#!/usr/bin/env bash
# Index management examples
# Covers: add index (a field_build job → 202), its job and progress,
#         drop index (a field_drop job → 202), the audit log
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

# A store's create, drop and index jobs run in the background (the routes
# answer 202). Wait until store $1 is in state $2 ("created", or "gone").
wait_for_state() {
  local ns="$1" want="$2" st=""
  for _ in $(seq 1 300); do
    st=$(curl -s "$BASE_URL/stores" | python3 -c "import json,sys; print(next((s.get('state','') for s in json.load(sys.stdin) if s['namespace']=='$ns'),'gone'))")
    [[ "$st" == "$want" ]] && return 0
    sleep 0.1
  done
  echo "  timed out waiting for '$ns' to be $want (it is $st)"
  return 1
}

echo "=== Setup: ensure '$NS' store exists ==="
curl -s -X POST "$BASE_URL/stores" \
  -H "Content-Type: application/json" \
  -d '{
    "store_type": "doc",
    "namespace": "users",
    "key_type":  "uuid",
    "attributes": [],
    "indices": [
      {"field": "status", "index_type": "str"}
    ]
  }' -o /dev/null || true   # ignore 409 if already present
wait_for_state "$NS" created

echo
echo "=== Add a new index on 'verified' (bool) ==="
# Returns 202 Accepted with the job's name — the index activates immediately
# for new writes, while a field_build job indexes the existing documents.
curl -sf -X POST "$BASE_URL/stores/$NS/indices" \
  -H "Content-Type: application/json" \
  -d '{"field": "verified", "index_type": "bool"}' | pp
echo "  -> HTTP 202 Accepted, job \"field_build:verified\", state \"index_maintenance\""

echo
echo "=== The store's jobs (status and progress) ==="
curl -sf "$BASE_URL/admin/stores/$NS/jobs" | pp
wait_for_state "$NS" created

echo
echo "=== Add a new index on 'score' (int) ==="
curl -sf -X POST "$BASE_URL/stores/$NS/indices" \
  -H "Content-Type: application/json" \
  -d '{"field": "score", "index_type": "int"}' | pp
echo "  -> HTTP 202 Accepted"
wait_for_state "$NS" created

echo
echo "=== List stores to see updated index definitions ==="
curl -sf "$BASE_URL/stores" | pp

echo
echo "=== Drop the 'score' index ==="
# The field is demoted to a non-indexed attribute in the schema;
# stored document data is unchanged.
curl -sf -X DELETE "$BASE_URL/stores/$NS/indices/score" | pp
echo "  -> HTTP 202 Accepted, job \"field_drop:score\""
wait_for_state "$NS" created

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

echo
echo "=== Cancel a job that is not running (expect 404 Not Found) ==="
curl -s -o /dev/null -w "HTTP %{http_code}\n" \
  -X DELETE "$BASE_URL/admin/stores/$NS/jobs/field_build:verified"

echo
echo "=== Lifecycle audit log for '$NS' (newest first, 10 records) ==="
curl -sf "$BASE_URL/admin/lifecycle/audit?namespace=$NS&limit=10" | pp
