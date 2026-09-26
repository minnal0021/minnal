# ── Stage 1: build ────────────────────────────────────────────────────────────
FROM rust:1.94-slim AS builder

# System deps needed for linking
RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy the full workspace so Cargo can resolve all members
COPY . .

# Build only the API server in release mode
RUN cargo build --release -p minnal_db_api

# Pick the cluster centroids for the model the config names — never a hard-coded
# model: both bundled sets are 768-dim, so a mismatch passes every load-time
# check and silently skews the IVF partition.
RUN model="$(sed -n 's/^[[:space:]]*model[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' config/sample.toml | head -n 1 | tr '[:upper:]' '[:lower:]')" \
    && test -n "$model" && test -f "service/embedding_support/$model/clusters.json" \
    && cp "service/embedding_support/$model/clusters.json" /build/clusters.json \
    && echo "cluster centroids: $model"

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the compiled binary
COPY --from=builder \
     /build/target/release/minnal_db_api \
     /usr/local/bin/minnal_db_api

# Copy the IVF cluster centroids chosen in the build stage — referenced by config as /clusters.json
COPY --from=builder /build/clusters.json /clusters.json

# Copy the sample config and rewrite the paths for the container layout:
#   db_path    ./data/db      → /data/minnal/db
#   schema_dir ./data/schemas → /data/minnal/schemas
#   log_dir    ./data/log     → /data/minnal/log
#   cluster_path ./clusters.json → /clusters.json (WORKDIR is /data, so the
#                relative path would miss the file)
COPY config/sample.toml /minnal.toml
RUN sed -i \
    -e 's|db_path = "./data/db"|db_path = "/data/minnal/db"|' \
    -e 's|schema_dir = "./data/schemas"|schema_dir = "/data/minnal/schemas"|' \
    -e 's|log_dir = "./data/log"|log_dir = "/data/minnal/log"|' \
    -e 's|cluster_path = "./clusters.json"|cluster_path = "/clusters.json"|' \
    /minnal.toml

# The /data directory is where all persistent database files are written.
# Mount a host directory here to persist data across container restarts:
#   docker run -p 8080:8080 -v "$(pwd)":/data minnal:latest
VOLUME ["/data"]

WORKDIR /data

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/minnal_db_api", "/minnal.toml"]
