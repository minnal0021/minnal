#!/usr/bin/env bash
# Store lifecycle examples
# Covers: list, create, amend schema (add/update/remove attribute), drop
#
# Usage:
#   ./stores.sh [BASE_URL]   (default: http://localhost:8080)

set -euo pipefail

BASE_URL="${1:-http://localhost:8080}"

# Pretty-print JSON if jq is available, else pass through raw.
pp() { command -v jq &>/dev/null && jq . || cat; }

echo "=== List all stores (initially empty) ==="
curl -sf "$BASE_URL/stores" | pp

echo
echo "=== Create a 'users' store (UUID keys, indexed on 'status' and 'age') ==="
curl -sf -X POST "$BASE_URL/stores" \
  -H "Content-Type: application/json" \
  -d '{
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
echo "  -> HTTP 201 Created"

echo
echo "=== Create a 'products' store (u64 keys, indexed on 'in_stock') ==="
curl -sf -X POST "$BASE_URL/stores" \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "products",
    "key_type": "u64",
    "attributes": [],
    "indices": [
      {"field": "in_stock", "index_type": "bool"},
      {"field": "price",    "index_type": "int"}
    ]
  }' | pp
echo "  -> HTTP 201 Created"

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
echo "  -> HTTP 204 No Content"

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
  -d '{"namespace":"users","key_type":"uuid","attributes":[],"indices":[]}'

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
echo "Done. 'users' store is still live for subsequent examples."
