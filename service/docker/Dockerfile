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

# Every model's centroids ship: each namespace uses the set for the model its
# schema names. Fail the build on a clone without its Git LFS files (pointer
# stubs instead of centroids).
RUN for f in service/embedding_support/*/clusters.json; do \
        test "$(head -c 7 "$f")" != "version" || { echo "LFS pointer, not centroids: $f" >&2; exit 1; }; \
    done

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

# Copy the IVF cluster centroids, one {model}/clusters.json per model —
# referenced by config as centroid_dir = "/embedding_support"
COPY --from=builder /build/service/embedding_support /embedding_support

# Copy the sample config and rewrite the paths for the container layout:
#   db_path    ./data/db      → /data/minnal/db
#   schema_dir ./data/schemas → /data/minnal/schemas
#   log_dir    ./data/log     → /data/minnal/log
#   centroid_dir service/embedding_support → /embedding_support (WORKDIR is
#                /data, so the relative path would miss the files)
COPY config/sample.toml /minnal.toml
RUN sed -i \
    -e 's|db_path = "./data/db"|db_path = "/data/minnal/db"|' \
    -e 's|schema_dir = "./data/schemas"|schema_dir = "/data/minnal/schemas"|' \
    -e 's|log_dir = "./data/log"|log_dir = "/data/minnal/log"|' \
    -e 's|centroid_dir = "service/embedding_support"|centroid_dir = "/embedding_support"|' \
    /minnal.toml

# The /data directory is where all persistent database files are written.
# Mount a host directory here to persist data across container restarts:
#   docker run -p 8080:8080 -v "$(pwd)":/data minnal:latest
VOLUME ["/data"]

WORKDIR /data

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/minnal_db_api", "/minnal.toml"]
