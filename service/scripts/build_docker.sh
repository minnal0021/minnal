#!/usr/bin/env bash
# Build the minnal Docker image locally.
#
# Usage:
#   ./build_docker.sh                   # default tag
#   ./build_docker.sh myrepo/minnal:v1.2  # custom tag
#
# The image is built from the workspace root so that all COPY
# paths in the Dockerfile resolve correctly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DOCKERFILE="$WORKSPACE_ROOT/service/docker/Dockerfile"

DOC_STORE_TAG="${1:-minnal:latest}"

echo "Building image: $DOC_STORE_TAG"
echo "Context:    $WORKSPACE_ROOT"
echo "Dockerfile: $DOCKERFILE"
echo ""

docker build \
    --file  "$DOCKERFILE" \
    --tag   "$DOC_STORE_TAG" \
    "$WORKSPACE_ROOT"

echo ""
echo "Build complete: $DOC_STORE_TAG"
