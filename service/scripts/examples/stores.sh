#!/usr/bin/env bash
# Store lifecycle examples
# Covers: list, create (202, then wait for "created"), amend schema
#         (add/update/remove attribute), drop (202, then wait until gone)
#
# Usage:
#   ./stores.sh [BASE_URL]   (default: http://localhost:8080)

set -euo pipefail

BASE_URL="${1:-http://localhost:8080}"

# Pretty-print JSON if jq is available, else pass through raw.
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

echo "=== List all stores (initially empty) ==="
curl -sf "$BASE_URL/stores" | pp

echo
echo "=== Create a 'users' store (UUID keys, indexed on 'status' and 'age') ==="
curl -sf -X POST "$BASE_URL/stores" \
  -H "Content-Type: application/json" \
  -d '{
    "store_type": "doc",
    "namespace": "users",
    "key_type": "uuid",
    "attributes": [
      {"name": "email",      "attr_type": "str"},
      {"name": "created_at", "attr_type": "int", "description": "unix timestamp"}
    ],
    "indices": [
      {"field": "status", "index_type": "str"},
      {"field": "age",    "index_type": "int"}
    ]
  }' | pp
echo "  -> HTTP 202 Accepted, state \"creating\"; the store is set up in the background"
wait_for_state users created && echo "  -> 'users' is created"

echo
echo "=== Create a 'products' store (u64 keys, indexed on 'in_stock') ==="
curl -sf -X POST "$BASE_URL/stores" \
  -H "Content-Type: application/json" \
  -d '{
    "store_type": "doc",
    "namespace": "products",
    "key_type": "u64",
    "attributes": [],
    "indices": [
      {"field": "in_stock", "index_type": "bool"},
      {"field": "price",    "index_type": "int"}
    ]
  }' | pp
echo "  -> HTTP 202 Accepted"
wait_for_state products created && echo "  -> 'products' is created"

echo
echo "=== List all stores (should show users + products) ==="
curl -sf "$BASE_URL/stores" | pp

echo
echo "=== Amend 'users' schema — add a non-indexed attribute ==="
curl -sf -X PATCH "$BASE_URL/stores/users/schema" \
  -H "Content-Type: application/json" \
  -d '{
    "op":        "add_attribute",
    "name":      "phone",
    "attr_type": "str",
    "description": "contact phone number"
  }' | pp
echo "  -> HTTP 204 No Content"

echo
echo "=== Amend 'users' schema — update an attribute type ==="
curl -sf -X PATCH "$BASE_URL/stores/users/schema" \
  -H "Content-Type: application/json" \
  -d '{
    "op":        "update_attribute",
    "name":      "phone",
    "attr_type": "int",
    "description": "phone as international dialling code (int)"
  }' | pp
echo "  -> HTTP 204 No Content"

echo
echo "=== Amend 'users' schema — remove an attribute ==="
curl -sf -X PATCH "$BASE_URL/stores/users/schema" \
  -H "Content-Type: application/json" \
  -d '{
    "op":  "remove_attribute",
    "name": "phone"
  }' | pp
echo "  -> HTTP 204 No Content"

echo
echo "=== Drop the 'products' store (irreversible) ==="
curl -sf -X DELETE "$BASE_URL/stores/products" | pp
echo "  -> HTTP 202 Accepted, state \"deleting\"; its data is removed in the background"
wait_for_state products gone && echo "  -> 'products' is gone"

echo
echo "=== List stores (only 'users' should remain) ==="
curl -sf "$BASE_URL/stores" | pp

echo
echo "--- Error cases ---"

echo
echo "=== Create duplicate store (expect 409 Conflict) ==="
curl -s -o /dev/null -w "HTTP %{http_code}\n" \
  -X POST "$BASE_URL/stores" \
  -H "Content-Type: application/json" \
  -d '{"store_type":"doc","namespace":"users","key_type":"uuid","attributes":[],"indices":[]}'

echo
echo "=== Amend attribute that is an active index (expect 409 Conflict) ==="
curl -s -o /dev/null -w "HTTP %{http_code}\n" \
  -X PATCH "$BASE_URL/stores/users/schema" \
  -H "Content-Type: application/json" \
  -d '{"op":"remove_attribute","name":"status"}'

echo
echo "=== Drop non-existent store (expect 404 Not Found) ==="
curl -s -o /dev/null -w "HTTP %{http_code}\n" \
  -X DELETE "$BASE_URL/stores/no_such_store"

echo
echo "=== Lifecycle audit log: what happened to 'products' (newest first) ==="
curl -sf "$BASE_URL/admin/lifecycle/audit?namespace=products" | pp

echo
echo "Done. 'users' store is still live for subsequent examples."
