#!/usr/bin/env bash
# Document CRUD and query examples
# Covers: put, find by id, delete, range scan, index predicate query,
#         semantic search (plain and predicate-filtered)
#
# Demonstrates all three key types:
#   uuid  → 'users'    store
#   u64   → 'products' store
#   u128  → 'events'   store
#
# Semantic search examples use a dedicated 'profiles' store
# (uuid-keyed, semantic_search_enabled).  These require:
#   - a cluster index loaded at server startup
#   - the embedding service to be running
#
# Usage:
#   ./docs.sh [BASE_URL]   (default: http://localhost:8080)

set -euo pipefail

BASE_URL="${1:-http://localhost:8080}"

pp() { command -v jq &>/dev/null && jq . || cat; }

# ── Setup ─────────────────────────────────────────────────────────────────────

echo "=== Setup: create test stores ==="

curl -s -X POST "$BASE_URL/stores" \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "users",
    "key_type":  "uuid",
    "attributes": [{"name": "email", "attr_type": "str"}],
    "indices":   [
      {"field": "status", "index_type": "str"},
      {"field": "age",    "index_type": "int"}
    ]
  }' -o /dev/null || true

curl -s -X POST "$BASE_URL/stores" \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "products",
    "key_type":  "u64",
    "attributes": [],
    "indices":   [
      {"field": "in_stock", "index_type": "bool"},
      {"field": "price",    "index_type": "int"}
    ]
  }' -o /dev/null || true

curl -s -X POST "$BASE_URL/stores" \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "events",
    "key_type":  "u128",
    "attributes": [],
    "indices":   [
      {"field": "kind", "index_type": "str"}
    ]
  }' -o /dev/null || true

echo "  -> stores ready"

# ── UUID-keyed store (users) ──────────────────────────────────────────────────

echo
echo "=========================================="
echo " UUID-keyed store: users"
echo "=========================================="

USER_1="550e8400-e29b-41d4-a716-446655440000"
USER_2="6ba7b810-9dad-11d1-80b4-00c04fd430c8"
USER_3="6ba7b811-9dad-11d1-80b4-00c04fd430c8"

echo
echo "=== PUT three user documents ==="
curl -sf -X PUT "$BASE_URL/stores/users/docs/$USER_1" \
  -H "Content-Type: application/json" \
  -d '{"name": "Alice", "status": "active",   "age": 30, "email": "alice@example.com"}' \
  | pp
echo "  -> HTTP 204 No Content"

curl -sf -X PUT "$BASE_URL/stores/users/docs/$USER_2" \
  -H "Content-Type: application/json" \
  -d '{"name": "Bob",   "status": "inactive", "age": 25, "email": "bob@example.com"}' \
  | pp
echo "  -> HTTP 204 No Content"

curl -sf -X PUT "$BASE_URL/stores/users/docs/$USER_3" \
  -H "Content-Type: application/json" \
  -d '{"name": "Carol", "status": "active",   "age": 19, "email": "carol@example.com"}' \
  | pp
echo "  -> HTTP 204 No Content"

echo
echo "=== GET user by UUID ==="
curl -sf "$BASE_URL/stores/users/docs/$USER_1" | pp

echo
echo "=== GET non-existent user (expect 404 Not Found) ==="
curl -s -o /dev/null -w "HTTP %{http_code}\n" \
  "$BASE_URL/stores/users/docs/00000000-0000-0000-0000-000000000000"

echo
echo "=== DELETE a user ==="
curl -sf -X DELETE "$BASE_URL/stores/users/docs/$USER_2" | pp
echo "  -> HTTP 204 No Content"

echo
echo "=== Index query: status = \"active\" ==="
curl -sf -X POST "$BASE_URL/stores/users/query" \
  -H "Content-Type: application/json" \
  -d '{"predicate": "status = \"active\""}' | pp

echo
echo "=== Index query: age >= 20 AND status = \"active\" ==="
curl -sf -X POST "$BASE_URL/stores/users/query" \
  -H "Content-Type: application/json" \
  -d '{"predicate": "age >= 20 AND status = \"active\""}' | pp

echo
echo "=== Range scan: all users from USER_1 to USER_3 (exclusive) ==="
curl -sf "$BASE_URL/stores/users/docs?start=$USER_1&end=$USER_3" | pp

echo
echo "=== Range scan: open-ended from USER_1 ==="
curl -sf "$BASE_URL/stores/users/docs?start=$USER_1" | pp

# ── U64-keyed store (products) ────────────────────────────────────────────────

echo
echo "=========================================="
echo " u64-keyed store: products"
echo "=========================================="

echo
echo "=== PUT three product documents ==="
curl -sf -X PUT "$BASE_URL/stores/products/docs/1001" \
  -H "Content-Type: application/json" \
  -d '{"name": "Widget A", "price": 999,  "in_stock": true}' | pp
echo "  -> HTTP 204 No Content"

curl -sf -X PUT "$BASE_URL/stores/products/docs/1002" \
  -H "Content-Type: application/json" \
  -d '{"name": "Widget B", "price": 1499, "in_stock": false}' | pp
echo "  -> HTTP 204 No Content"

curl -sf -X PUT "$BASE_URL/stores/products/docs/1003" \
  -H "Content-Type: application/json" \
  -d '{"name": "Widget C", "price": 299,  "in_stock": true}' | pp
echo "  -> HTTP 204 No Content"

echo
echo "=== GET product 1002 ==="
curl -sf "$BASE_URL/stores/products/docs/1002" | pp

echo
echo "=== Index query: in_stock = true ==="
curl -sf -X POST "$BASE_URL/stores/products/query" \
  -H "Content-Type: application/json" \
  -d '{"predicate": "in_stock = true"}' | pp

echo
echo "=== Index query: price <= 999 AND in_stock = true ==="
curl -sf -X POST "$BASE_URL/stores/products/query" \
  -H "Content-Type: application/json" \
  -d '{"predicate": "price <= 999 AND in_stock = true"}' | pp

echo
echo "=== Range scan: products 1001 to 1003 (exclusive) ==="
curl -sf "$BASE_URL/stores/products/docs?start=1001&end=1003" | pp

# ── U128-keyed store (events) ─────────────────────────────────────────────────

echo
echo "=========================================="
echo " u128-keyed store: events"
echo "=========================================="

echo
echo "=== PUT two event documents ==="
curl -sf -X PUT "$BASE_URL/stores/events/docs/100000000000000001" \
  -H "Content-Type: application/json" \
  -d '{"kind": "login",  "user": "alice", "ts": 1700000000}' | pp
echo "  -> HTTP 204 No Content"

curl -sf -X PUT "$BASE_URL/stores/events/docs/100000000000000002" \
  -H "Content-Type: application/json" \
  -d '{"kind": "logout", "user": "alice", "ts": 1700003600}' | pp
echo "  -> HTTP 204 No Content"

echo
echo "=== GET event by u128 id ==="
curl -sf "$BASE_URL/stores/events/docs/100000000000000001" | pp

echo
echo "=== Index query: kind = \"login\" ==="
curl -sf -X POST "$BASE_URL/stores/events/query" \
  -H "Content-Type: application/json" \
  -d '{"predicate": "kind = \"login\""}' | pp

echo
echo "=== Range scan: events 100000000000000001 onwards ==="
curl -sf "$BASE_URL/stores/events/docs?start=100000000000000001" | pp

# ── Error cases ───────────────────────────────────────────────────────────────

echo
echo "=========================================="
echo " Error cases"
echo "=========================================="

echo
echo "=== GET from non-existent store (expect 404 Not Found) ==="
curl -sf "$BASE_URL/stores/ghost/docs/1" \
  -w "\nHTTP %{http_code}\n" || true

echo
echo "=== PUT with malformed UUID id (expect 400 Bad Request) ==="
curl -s -o /dev/null -w "HTTP %{http_code}\n" \
  -X PUT "$BASE_URL/stores/users/docs/not-a-valid-uuid" \
  -H "Content-Type: application/json" \
  -d '{"name": "Bad"}'

echo
echo "=== Query with unsatisfiable predicate referencing non-indexed field ==="
curl -sf -X POST "$BASE_URL/stores/users/query" \
  -H "Content-Type: application/json" \
  -d '{"predicate": "email = \"x@y.com\""}' \
  -w "\nHTTP %{http_code}\n" || true

# ── Semantic search ───────────────────────────────────────────────────────────
#
# Prerequisites:
#   1. Server started with the sample config so the cluster index is loaded:
#        ./run.sh config/sample.toml
#   2. Embedding service running at http://localhost:8000/embeddings
#
# If either is missing the store creation returns HTTP 500 and the section is
# skipped automatically.

echo
echo "=========================================="
echo " Semantic search"
echo "=========================================="

echo
echo "=== Setup: create semantic-search-enabled store ==="
# semantic_search_enabled: true tells the server to embed every document on
# write and maintain the companion vector KV store.
# embedding_field names the JSON field whose text is embedded.
# status and seniority are indexed so the filtered examples can use them.
SS_CREATE_STATUS=$(curl -s -o /tmp/ss_create_resp.json -w "%{http_code}" \
  -X POST "$BASE_URL/stores" \
  -H "Content-Type: application/json" \
  -d '{
    "namespace":               "profiles",
    "key_type":                "uuid",
    "semantic_search_enabled": true,
    "embedding_field":         "bio",
    "attributes":              [{"name": "name", "attr_type": "str"}],
    "indices": [
      {"field": "status",    "index_type": "str"},
      {"field": "seniority", "index_type": "str"}
    ]
  }')

if [[ "$SS_CREATE_STATUS" != "201" ]]; then
  echo "  SKIPPED — store creation returned HTTP $SS_CREATE_STATUS:"
  cat /tmp/ss_create_resp.json | pp
  echo
  echo "  Make sure the server was started with the sample config and the"
  echo "  embedding service is running:  ./run.sh config/sample.toml"
  echo
else
  echo "  -> HTTP 201 Created"

  PROFILE_1="aaaaaaaa-0000-0000-0000-000000000001"
  PROFILE_2="aaaaaaaa-0000-0000-0000-000000000002"
  PROFILE_3="aaaaaaaa-0000-0000-0000-000000000003"
  PROFILE_4="aaaaaaaa-0000-0000-0000-000000000004"

  echo
  echo "=== PUT four candidate profiles ==="
  curl -sf -X PUT "$BASE_URL/stores/profiles/docs/$PROFILE_1" \
    -H "Content-Type: application/json" \
    -d '{
      "name":      "Alice",
      "status":    "active",
      "seniority": "senior",
      "bio":       "Senior Rust engineer with 8 years of experience building distributed systems and high-throughput data pipelines."
    }' | pp
  echo "  -> HTTP 204 No Content"

  curl -sf -X PUT "$BASE_URL/stores/profiles/docs/$PROFILE_2" \
    -H "Content-Type: application/json" \
    -d '{
      "name":      "Bob",
      "status":    "active",
      "seniority": "junior",
      "bio":       "Junior backend developer familiar with Python and REST APIs, looking to grow into distributed systems."
    }' | pp
  echo "  -> HTTP 204 No Content"

  curl -sf -X PUT "$BASE_URL/stores/profiles/docs/$PROFILE_3" \
    -H "Content-Type: application/json" \
    -d '{
      "name":      "Carol",
      "status":    "inactive",
      "seniority": "senior",
      "bio":       "Principal engineer specialising in database internals, storage engines, and query optimisation."
    }' | pp
  echo "  -> HTTP 204 No Content"

  curl -sf -X PUT "$BASE_URL/stores/profiles/docs/$PROFILE_4" \
    -H "Content-Type: application/json" \
    -d '{
      "name":      "Dave",
      "status":    "active",
      "seniority": "senior",
      "bio":       "Staff software engineer focused on frontend performance and React architecture."
    }' | pp
  echo "  -> HTTP 204 No Content"

  echo
  echo "=== Semantic search (no filter): find profiles similar to a query ==="
  # Returns all candidates ranked by dot-product similarity.
  # Expect Alice and Carol near the top; Dave near the bottom.
  curl -sf -X POST "$BASE_URL/stores/profiles/semantic-search" \
    -H "Content-Type: application/json" \
    -d '{"query": "experienced systems engineer with distributed databases background"}' | pp

  echo
  echo "=== Semantic search (filtered): same query, active profiles only ==="
  # Carol (inactive, senior) is excluded even though she would rank highly on
  # semantic similarity alone.
  curl -sf -X POST "$BASE_URL/stores/profiles/semantic-search/filtered" \
    -H "Content-Type: application/json" \
    -d '{
      "query":     "experienced systems engineer with distributed databases background",
      "predicate": "status = \"active\""
    }' | pp

  echo
  echo "=== Semantic search (filtered): senior active profiles only ==="
  curl -sf -X POST "$BASE_URL/stores/profiles/semantic-search/filtered" \
    -H "Content-Type: application/json" \
    -d '{
      "query":     "experienced systems engineer with distributed databases background",
      "predicate": "status = \"active\" AND seniority = \"senior\""
    }' | pp

  echo
  echo "=== Error: semantic search on store without semantic_search_enabled ==="
  # The 'users' store was created without semantic_search_enabled, so this
  # returns 500 with an explanatory error message.
  curl -s -X POST "$BASE_URL/stores/users/semantic-search" \
    -H "Content-Type: application/json" \
    -d '{"query": "test"}' \
    -w "\nHTTP %{http_code}\n" || true
fi
