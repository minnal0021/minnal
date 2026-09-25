#!/usr/bin/env bash
# fetch_beir.sh -- download a BEIR dataset for the rank-fusion relevance eval
#
# Usage: service/scripts/fetch_beir.sh [dataset ...]      (default: scifact)
#
# Downloads https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/<name>.zip
# and unpacks it to work/beir/<name>/ (corpus.jsonl, queries.jsonl, qrels/*.tsv).
# work/ is gitignored. Small datasets that suit the eval: scifact (5k docs),
# nfcorpus (3.6k docs, graded qrels), fiqa (57k docs — slow to embed).
#
# Then run the eval (see minnal_db/src/semantic_search/beir_eval.rs):
#   MINNAL_EMBED_URL=http://localhost:8001 \
#     cargo test -p minnal_db --all-features --release --lib beir_rank_fusion_eval -- --ignored --nocapture

set -euo pipefail

BASE_URL="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${REPO_ROOT}/work/beir"

for tool in curl unzip; do
    command -v "$tool" >/dev/null || { echo "error: '$tool' is required" >&2; exit 1; }
done

mkdir -p "$DEST"
for name in "${@:-scifact}"; do
    if [[ -f "$DEST/$name/corpus.jsonl" ]]; then
        echo "$name: already present in $DEST/$name"
        continue
    fi
    zip="$DEST/$name.zip"
    echo "$name: downloading $BASE_URL/$name.zip"
    curl -fL --retry 3 -o "$zip" "$BASE_URL/$name.zip"
    unzip -q -o "$zip" -d "$DEST"
    rm -f "$zip"
    [[ -f "$DEST/$name/corpus.jsonl" ]] || { echo "error: $name unpacked without corpus.jsonl" >&2; exit 1; }
    echo "$name: $(wc -l < "$DEST/$name/corpus.jsonl") docs, $(wc -l < "$DEST/$name/queries.jsonl") queries -> $DEST/$name"
done
