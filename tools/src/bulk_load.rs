//! `bulk_load` — load a JSONL file into a minnal **document store** or **KV
//! store** via the REST API, optionally importing the store's schema first.
//!
//! A single command handles both store kinds.  By default it loads a document
//! store (each JSON line becomes one document); pass `--kv` to load a KV store
//! (each line supplies a key and a value via separate fields).
//!
//! # Usage
//!
//! ```text
//! # Document store (default)
//! minnal_tools bulk_load [--no-wal] [--schema <schema.json>] <url> <namespace> <id_field> <data.jsonl>
//!
//! # KV store
//! minnal_tools bulk_load --kv [--no-wal] [--schema <schema.json>] <url> <namespace> <key_field> <value_field> <data.jsonl>
//! ```
//!
//! # Positional arguments
//!
//! Document store (default):
//!
//! | Argument     | Description                                               |
//! |--------------|-----------------------------------------------------------|
//! | `url`        | Base URL of the running REST API                         |
//! | `namespace`  | Name of the doc store namespace to load into             |
//! | `id_field`   | JSON field name whose value is the document ID           |
//! | `data.jsonl` | Full path to the JSONL file (one JSON object per line)   |
//!
//! KV store (`--kv`):
//!
//! | Argument      | Description                                              |
//! |---------------|----------------------------------------------------------|
//! | `url`         | Base URL of the running REST API                         |
//! | `namespace`   | Name of the KV store namespace to load into              |
//! | `key_field`   | JSON field name whose value is the entry key             |
//! | `value_field` | JSON field name whose value is stored as the entry value |
//! | `data.jsonl`  | Full path to the JSONL file (one JSON object per line)   |
//!
//! # Flags
//!
//! | Flag                | Description                                                 |
//! |---------------------|-------------------------------------------------------------|
//! | `--kv`              | Load a KV store instead of a document store.                |
//! | `--schema <file>`   | Import the schema before loading (doc → `POST                |
//! |                     | /admin/stores/import`, KV → `POST /admin/kv-stores/import`).  |
//! |                     | The schema file's `namespace` must match the `namespace`     |
//! |                     | argument and its `key_type` must match the selected store    |
//! |                     | kind.  An existing store is reused, so re-runs are safe.     |
//! |                     | Without this flag the namespace must already exist.         |
//! | `--no-wal`          | Bypass WAL writes for maximum throughput.  Data written     |
//! |                     | this way is unrecoverable on a crash — only use when         |
//! |                     | re-running the load is acceptable.                          |
//!
//! # ID / key / value rules
//!
//! For document stores the `id_field` value is parsed according to the
//! namespace's `key_type`:
//!
//! | `key_type` | Expected JSON value                     |
//! |------------|-----------------------------------------|
//! | `u64`      | JSON number or numeric string           |
//! | `u128`     | JSON number or numeric string           |
//! | `uuid`     | UUID string (`xxxxxxxx-xxxx-…`)         |
//!
//! For KV stores the `key_field` value is parsed according to the namespace's
//! `key_type` (`str` → JSON string, `int` → JSON integer or numeric string) and
//! the `value_field` value is sent verbatim, validated server-side against the
//! namespace's `value_type` (`str`, `int`, `f32`, `vec_f32`).
//!
//! Lines with a missing or unparseable ID/key are skipped with a warning and
//! logged to a sibling `.errors` file.
//!
//! # Examples
//!
//! ```text
//! # Import a doc-store schema, then load (one step from a fresh server)
//! minnal_tools bulk_load --schema ./jobs-mini-schema.json http://localhost:8080 jobs jobId ./jobs-mini.jsonl
//!
//! # Import a KV-store schema, then load key/value pairs
//! minnal_tools bulk_load --kv --schema ./job-content-kv-schema.json \
//!   http://localhost:8080 job-content key value ./job-content-kv.jsonl
//!
//! # Load into a store that already exists
//! minnal_tools bulk_load http://localhost:8080 users id ./users.jsonl
//!
//! # Fast load (no WAL — only use when re-running is acceptable)
//! minnal_tools bulk_load --no-wal http://localhost:8080 users id ./users.jsonl
//! ```
//!
//! # Semantic search caveat
//!
//! If the schema has `semantic_search_enabled = true` but no embedding service
//! is running, loads still succeed and attribute queries still work — only
//! semantic-search queries return nothing until embeddings are produced.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use reqwest::{Client, StatusCode};
use serde::Deserialize;
use serde_json::Value;

// ── Store kind ──────────────────────────────────────────────────────────────

/// Which store type a `bulk_load` invocation targets.
#[derive(Debug, Clone, Copy, PartialEq)]
enum StoreKind {
    Doc,
    Kv,
}

impl StoreKind {
    /// Human-readable label for messages.
    fn label(self) -> &'static str {
        match self {
            StoreKind::Doc => "doc store",
            StoreKind::Kv => "KV store",
        }
    }

    /// Classify a schema `key_type` string into the store kind it belongs to.
    /// Returns `None` for an unrecognised value.
    fn from_key_type(key_type: &str) -> Option<StoreKind> {
        match key_type {
            "u64" | "u128" | "uuid" => Some(StoreKind::Doc),
            "str" | "int" => Some(StoreKind::Kv),
            _ => None,
        }
    }
}

// ── Doc-store key type ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
pub(crate) enum DocKeyType {
    #[serde(rename = "u64")]
    U64,
    #[serde(rename = "u128")]
    U128,
    #[serde(rename = "uuid")]
    Uuid,
}

/// Minimal projection of the doc-store schema returned by `GET /stores`.
#[derive(Debug, Deserialize)]
struct DocSchema {
    namespace: String,
    key_type: DocKeyType,
}

// ── KV-store key type ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
pub(crate) enum KvKeyType {
    #[serde(rename = "str")]
    Str,
    #[serde(rename = "int")]
    Int,
}

/// Minimal projection of the KV-store schema returned by `GET /kv-stores`.
#[derive(Debug, Deserialize)]
struct KvSchema {
    namespace: String,
    key_type: KvKeyType,
}

// ── Bulk load ───────────────────────────────────────────────────────────────

fn usage() -> ! {
    eprintln!(concat!(
        "usage:\n",
        "  doc store:  minnal_tools bulk_load [--no-wal] [--schema <schema.json>] <url> <namespace> <id_field> <data.jsonl>\n",
        "  KV store:   minnal_tools bulk_load --kv [--no-wal] [--schema <schema.json>] <url> <namespace> <key_field> <value_field> <data.jsonl>\n",
        "\n",
        "arguments:\n",
        "  url          base URL of the running REST API\n",
        "  namespace    name of the namespace to load into\n",
        "  id_field     (doc) JSON field name whose value is the document ID\n",
        "  key_field    (--kv) JSON field name whose value is the entry key\n",
        "  value_field  (--kv) JSON field name whose value is stored as the entry value\n",
        "  data.jsonl   full path to the JSONL file (one JSON object per line)\n",
        "\n",
        "flags:\n",
        "  --kv                    load a KV store instead of a document store\n",
        "  --schema <schema.json>  import the schema before loading (an existing store\n",
        "                          is reused, so re-runs are safe); the schema's\n",
        "                          'namespace' must match the namespace argument and its\n",
        "                          'key_type' must match the selected store kind. Without\n",
        "                          this flag the namespace must already exist\n",
        "  --no-wal                bypass WAL writes for maximum throughput; data written\n",
        "                          this way is unrecoverable on a crash — only use when\n",
        "                          re-running the load is acceptable\n",
        "\n",
        "examples:\n",
        "  minnal_tools bulk_load --schema ./jobs-mini-schema.json http://localhost:8080 jobs jobId ./jobs-mini.jsonl\n",
        "  minnal_tools bulk_load --kv --schema ./job-content-kv-schema.json http://localhost:8080 job-content key value ./job-content-kv.jsonl\n",
        "  minnal_tools bulk_load --no-wal http://localhost:8080 users id ./users.jsonl",
    ));
    std::process::exit(1);
}

/// Entry point called from `main.rs` with the arguments after the tool name.
pub async fn run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    // ── Parse flags and positional arguments ──────────────────────────────
    let mut skip_wal = false;
    let mut kind = StoreKind::Doc;
    let mut schema_path: Option<PathBuf> = None;
    let mut positional: Vec<&String> = Vec::new();

    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--no-wal" => skip_wal = true,
            "--kv" => kind = StoreKind::Kv,
            "--schema" => {
                let path = iter.next().ok_or("--schema requires a <schema.json> path argument")?;
                schema_path = Some(PathBuf::from(path));
            }
            flag if flag.starts_with("--") => return Err(format!("unknown flag: {flag}").into()),
            _ => positional.push(arg),
        }
    }

    if skip_wal {
        println!("WAL disabled — maximum throughput mode (data unrecoverable on crash)");
    }

    match kind {
        StoreKind::Doc => run_doc(positional, schema_path, skip_wal).await,
        StoreKind::Kv => run_kv(positional, schema_path, skip_wal).await,
    }
}

/// Document-store load path.
async fn run_doc(positional: Vec<&String>, schema_path: Option<PathBuf>, skip_wal: bool) -> Result<(), Box<dyn std::error::Error>> {
    if positional.len() != 4 {
        usage();
    }
    let base_url = positional[0].trim_end_matches('/');
    let namespace = positional[1];
    let id_field = positional[2];
    let jsonl_path = PathBuf::from(positional[3]);

    let client = Client::new();

    if let Some(schema_path) = &schema_path {
        import_schema(&client, base_url, schema_path, namespace, StoreKind::Doc, "/admin/stores/import").await?;
    }

    let key_type = resolve_doc_key_type(&client, base_url, namespace).await?;
    println!("namespace '{namespace}' found  key_type={key_type:?}");

    load_doc_jsonl(&client, base_url, namespace, id_field, key_type, &jsonl_path, skip_wal).await
}

/// KV-store load path.
async fn run_kv(positional: Vec<&String>, schema_path: Option<PathBuf>, skip_wal: bool) -> Result<(), Box<dyn std::error::Error>> {
    if positional.len() != 5 {
        usage();
    }
    let base_url = positional[0].trim_end_matches('/');
    let namespace = positional[1];
    let key_field = positional[2];
    let value_field = positional[3];
    let jsonl_path = PathBuf::from(positional[4]);

    let client = Client::new();

    if let Some(schema_path) = &schema_path {
        import_schema(&client, base_url, schema_path, namespace, StoreKind::Kv, "/admin/kv-stores/import").await?;
    }

    let key_type = resolve_kv_key_type(&client, base_url, namespace).await?;
    println!("KV namespace '{namespace}' found  key_type={key_type:?}");

    load_kv_jsonl(&client, base_url, namespace, key_field, value_field, key_type, &jsonl_path, skip_wal).await
}

/// Import a schema via the store-kind-appropriate import endpoint.  An existing
/// store (HTTP 409 Conflict) is reused, so re-runs are safe.  The schema file's
/// `namespace` must match `namespace`, and its `key_type` must belong to the
/// selected store kind — guarding against, e.g., a doc schema loaded with `--kv`.
async fn import_schema(
    client: &Client,
    base_url: &str,
    schema_path: &Path,
    namespace: &str,
    kind: StoreKind,
    endpoint: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let schema_bytes = std::fs::read(schema_path).map_err(|e| format!("cannot open '{}': {e}", schema_path.display()))?;
    let schema: Value = serde_json::from_slice(&schema_bytes).map_err(|e| format!("'{}' is not valid JSON: {e}", schema_path.display()))?;

    let schema_ns = schema
        .get("namespace")
        .and_then(Value::as_str)
        .ok_or("schema file has no string 'namespace' field")?;
    if schema_ns != namespace {
        return Err(format!("namespace mismatch: argument is '{namespace}' but schema declares '{schema_ns}'").into());
    }

    // Validate the schema's key_type matches the selected store kind so a
    // mismatched --kv flag fails fast with a clear message.
    let schema_key_type = schema
        .get("key_type")
        .and_then(Value::as_str)
        .ok_or("schema file has no string 'key_type' field")?;
    match StoreKind::from_key_type(schema_key_type) {
        Some(schema_kind) if schema_kind == kind => {}
        Some(schema_kind) => {
            let hint = match schema_kind {
                StoreKind::Kv => "pass --kv to load a KV store",
                StoreKind::Doc => "drop --kv to load a document store",
            };
            return Err(format!(
                "store-kind mismatch: requested a {} but schema declares key_type '{schema_key_type}' (a {}) — {hint}",
                kind.label(),
                schema_kind.label(),
            )
            .into());
        }
        None => return Err(format!("schema declares an unrecognised key_type '{schema_key_type}'").into()),
    }

    let resp = client
        .post(format!("{base_url}{endpoint}"))
        .json(&schema)
        .send()
        .await
        .map_err(|e| format!("cannot reach '{base_url}{endpoint}': {e}"))?;

    match resp.status() {
        s if s.is_success() => println!("schema imported — {} '{namespace}' created", kind.label()),
        StatusCode::CONFLICT => println!("{} '{namespace}' already exists — reusing it", kind.label()),
        s => {
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("schema import failed ({s}): {body}").into());
        }
    }
    Ok(())
}

// ── Document store ──────────────────────────────────────────────────────────

/// Resolve a doc namespace's `key_type` via `GET /stores`, erroring if the
/// namespace does not exist.
async fn resolve_doc_key_type(client: &Client, base_url: &str, namespace: &str) -> Result<DocKeyType, Box<dyn std::error::Error>> {
    let stores: Vec<DocSchema> = client
        .get(format!("{base_url}/stores"))
        .send()
        .await
        .map_err(|e| format!("cannot reach '{base_url}/stores': {e}"))?
        .error_for_status()
        .map_err(|e| format!("GET /stores failed: {e}"))?
        .json()
        .await
        .map_err(|e| format!("cannot parse /stores response: {e}"))?;

    stores.iter().find(|s| s.namespace == namespace).map(|s| s.key_type).ok_or_else(|| {
        let available: Vec<&str> = stores.iter().map(|s| s.namespace.as_str()).collect();
        format!(
            "namespace '{namespace}' not found — available: {}",
            if available.is_empty() {
                "(none)".to_owned()
            } else {
                available.join(", ")
            }
        )
        .into()
    })
}

/// Stream a JSONL file into an existing doc `namespace`, PUTting one document per
/// line.  Lines that fail validation are counted as skipped and written to a
/// sibling `.errors` file.
async fn load_doc_jsonl(
    client: &Client,
    base_url: &str,
    namespace: &str,
    id_field: &str,
    key_type: DocKeyType,
    jsonl_path: &Path,
    skip_wal: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("id_field='{id_field}'");

    let file = File::open(jsonl_path).map_err(|e| format!("cannot open '{}': {e}", jsonl_path.display()))?;

    let reader = BufReader::new(file);
    let mut loaded: u64 = 0;
    let mut skipped: u64 = 0;
    let mut error_lines: Vec<String> = Vec::new();

    let started = Instant::now();

    for (line_no, line) in reader.lines().enumerate() {
        let line_no = line_no + 1;
        let line = line.map_err(|e| format!("read error at line {line_no}: {e}"))?;
        let trimmed = line.trim();

        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }

        let doc: Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                let msg = format!("line {line_no}: invalid JSON ({e})");
                eprintln!("  {msg} — skipped");
                error_lines.push(format!("{msg}\n  {trimmed}"));
                skipped += 1;
                continue;
            }
        };

        let id_str = match extract_doc_id(&doc, id_field, key_type, line_no, trimmed, &mut error_lines) {
            Some(s) => s,
            None => {
                skipped += 1;
                continue;
            }
        };

        let put_url = if skip_wal {
            format!("{base_url}/stores/{namespace}/docs/{id_str}?skip_wal=true")
        } else {
            format!("{base_url}/stores/{namespace}/docs/{id_str}")
        };

        match client.put(put_url).json(&doc).send().await {
            Ok(resp) if resp.status().is_success() => {
                loaded += 1;
                if loaded.is_multiple_of(1_000) {
                    println!("  {loaded} documents loaded…");
                }
            }
            Ok(resp) => {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                let msg = format!("line {line_no}: PUT failed ({status}: {body})");
                eprintln!("  {msg} — skipped");
                error_lines.push(format!("{msg}\n  {trimmed}"));
                skipped += 1;
            }
            Err(e) => {
                let msg = format!("line {line_no}: request error ({e})");
                eprintln!("  {msg} — skipped");
                error_lines.push(format!("{msg}\n  {trimmed}"));
                skipped += 1;
            }
        }
    }

    finish(jsonl_path, loaded, skipped, &error_lines, started, "documents")
}

/// Extract the document ID from `doc[id_field]` and format it as a URL path
/// segment, according to the namespace's `key_type`.  Appends the error message
/// to `errors` and returns `None` when the field is absent or unparseable.
fn extract_doc_id(doc: &Value, id_field: &str, key_type: DocKeyType, line_no: usize, raw_line: &str, errors: &mut Vec<String>) -> Option<String> {
    let raw = match doc.get(id_field) {
        Some(v) => v,
        None => {
            let msg = format!("line {line_no}: missing id field '{id_field}'");
            eprintln!("  {msg} — skipped");
            errors.push(format!("{msg}\n  {raw_line}"));
            return None;
        }
    };

    let result = match key_type {
        DocKeyType::U64 => match raw {
            Value::Number(n) => n.as_u64().map(|v| v.to_string()),
            Value::String(s) => s.parse::<u64>().ok().map(|v| v.to_string()),
            _ => None,
        },
        DocKeyType::U128 => match raw {
            Value::Number(n) => n.as_u128().map(|v| v.to_string()),
            Value::String(s) => s.parse::<u128>().ok().map(|v| v.to_string()),
            _ => None,
        },
        DocKeyType::Uuid => match raw {
            Value::String(s) if is_valid_uuid(s) => Some(s.clone()),
            _ => None,
        },
    };

    if result.is_none() {
        let msg = format!("line {line_no}: cannot parse '{raw}' as {key_type:?}");
        eprintln!("  {msg} — skipped");
        errors.push(format!("{msg}\n  {raw_line}"));
    }
    result
}

/// Returns `true` if `s` contains exactly 32 ASCII hex digits (with or without
/// hyphens in the standard UUID positions).
fn is_valid_uuid(s: &str) -> bool {
    let hex_count = s.chars().filter(|c| c.is_ascii_hexdigit()).count();
    hex_count == 32
}

// ── KV store ────────────────────────────────────────────────────────────────

/// Resolve a KV namespace's `key_type` via `GET /kv-stores`, erroring if the
/// namespace does not exist.
async fn resolve_kv_key_type(client: &Client, base_url: &str, namespace: &str) -> Result<KvKeyType, Box<dyn std::error::Error>> {
    let stores: Vec<KvSchema> = client
        .get(format!("{base_url}/kv-stores"))
        .send()
        .await
        .map_err(|e| format!("cannot reach '{base_url}/kv-stores': {e}"))?
        .error_for_status()
        .map_err(|e| format!("GET /kv-stores failed: {e}"))?
        .json()
        .await
        .map_err(|e| format!("cannot parse /kv-stores response: {e}"))?;

    stores.iter().find(|s| s.namespace == namespace).map(|s| s.key_type).ok_or_else(|| {
        let available: Vec<&str> = stores.iter().map(|s| s.namespace.as_str()).collect();
        format!(
            "KV namespace '{namespace}' not found — available: {}",
            if available.is_empty() {
                "(none)".to_owned()
            } else {
                available.join(", ")
            }
        )
        .into()
    })
}

/// Stream a JSONL file into an existing KV `namespace`, PUTting one entry per
/// line.  Lines that fail validation are counted as skipped and written to a
/// sibling `.errors` file.
#[allow(clippy::too_many_arguments)]
async fn load_kv_jsonl(
    client: &Client,
    base_url: &str,
    namespace: &str,
    key_field: &str,
    value_field: &str,
    key_type: KvKeyType,
    jsonl_path: &Path,
    skip_wal: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("key_field='{key_field}'  value_field='{value_field}'");

    let file = File::open(jsonl_path).map_err(|e| format!("cannot open '{}': {e}", jsonl_path.display()))?;

    let reader = BufReader::new(file);
    let mut loaded: u64 = 0;
    let mut skipped: u64 = 0;
    let mut error_lines: Vec<String> = Vec::new();

    let started = Instant::now();

    for (line_no, line) in reader.lines().enumerate() {
        let line_no = line_no + 1;
        let line = line.map_err(|e| format!("read error at line {line_no}: {e}"))?;
        let trimmed = line.trim();

        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }

        let row: Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                let msg = format!("line {line_no}: invalid JSON ({e})");
                eprintln!("  {msg} — skipped");
                error_lines.push(format!("{msg}\n  {trimmed}"));
                skipped += 1;
                continue;
            }
        };

        let key_str = match extract_kv_key(&row, key_field, key_type, line_no, trimmed, &mut error_lines) {
            Some(s) => s,
            None => {
                skipped += 1;
                continue;
            }
        };

        let value = match row.get(value_field) {
            Some(v) => v,
            None => {
                let msg = format!("line {line_no}: missing value field '{value_field}'");
                eprintln!("  {msg} — skipped");
                error_lines.push(format!("{msg}\n  {trimmed}"));
                skipped += 1;
                continue;
            }
        };

        let put_url = if skip_wal {
            format!("{base_url}/kv-stores/{namespace}/kv/{key_str}?skip_wal=true")
        } else {
            format!("{base_url}/kv-stores/{namespace}/kv/{key_str}")
        };

        match client.put(put_url).json(value).send().await {
            Ok(resp) if resp.status().is_success() => {
                loaded += 1;
                if loaded.is_multiple_of(1_000) {
                    println!("  {loaded} entries loaded…");
                }
            }
            Ok(resp) => {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                let msg = format!("line {line_no}: PUT failed ({status}: {body})");
                eprintln!("  {msg} — skipped");
                error_lines.push(format!("{msg}\n  {trimmed}"));
                skipped += 1;
            }
            Err(e) => {
                let msg = format!("line {line_no}: request error ({e})");
                eprintln!("  {msg} — skipped");
                error_lines.push(format!("{msg}\n  {trimmed}"));
                skipped += 1;
            }
        }
    }

    finish(jsonl_path, loaded, skipped, &error_lines, started, "entries")
}

/// Extract the entry key from `row[key_field]` and format it as a URL path
/// segment, according to the namespace's `key_type`.  Appends the error message
/// to `errors` and returns `None` when the field is absent or unparseable.
fn extract_kv_key(row: &Value, key_field: &str, key_type: KvKeyType, line_no: usize, raw_line: &str, errors: &mut Vec<String>) -> Option<String> {
    let raw = match row.get(key_field) {
        Some(v) => v,
        None => {
            let msg = format!("line {line_no}: missing key field '{key_field}'");
            eprintln!("  {msg} — skipped");
            errors.push(format!("{msg}\n  {raw_line}"));
            return None;
        }
    };

    let result = match key_type {
        KvKeyType::Str => match raw {
            Value::String(s) => Some(s.clone()),
            _ => None,
        },
        KvKeyType::Int => match raw {
            Value::Number(n) => n.as_i64().map(|v| v.to_string()),
            Value::String(s) => s.parse::<i64>().ok().map(|v| v.to_string()),
            _ => None,
        },
    };

    if result.is_none() {
        let msg = format!("line {line_no}: cannot parse '{raw}' as {key_type:?}");
        eprintln!("  {msg} — skipped");
        errors.push(format!("{msg}\n  {raw_line}"));
    }
    result
}

// ── Shared completion ───────────────────────────────────────────────────────

/// Write the `.errors` file (if any) and print the summary line.  `noun` is the
/// unit reported in the summary ("documents" / "entries").
fn finish(
    jsonl_path: &Path,
    loaded: u64,
    skipped: u64,
    error_lines: &[String],
    started: Instant,
    noun: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let elapsed_secs = started.elapsed().as_secs_f64();

    let error_path = jsonl_path.with_extension("errors");
    if !error_lines.is_empty() {
        let ef = File::create(&error_path).map_err(|e| format!("cannot create error file '{}': {e}", error_path.display()))?;
        let mut w = BufWriter::new(ef);
        for entry in error_lines {
            writeln!(w, "{entry}").map_err(|e| format!("write error: {e}"))?;
        }
        eprintln!("{} error(s) written to '{}'", error_lines.len(), error_path.display());
    }

    println!(
        "done  loaded={loaded} {noun}  skipped={skipped}  total={}  elapsed={elapsed_secs:.2}s",
        loaded + skipped
    );
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── is_valid_uuid ─────────────────────────────────────────────────────

    #[test]
    fn valid_uuid_with_hyphens() {
        assert!(is_valid_uuid("550e8400-e29b-41d4-a716-446655440000"));
    }

    #[test]
    fn valid_uuid_without_hyphens() {
        assert!(is_valid_uuid("550e8400e29b41d4a716446655440000"));
    }

    #[test]
    fn uuid_too_short() {
        assert!(!is_valid_uuid("550e8400-e29b-41d4"));
    }

    #[test]
    fn uuid_non_hex() {
        assert!(!is_valid_uuid("zzzzzzzz-zzzz-zzzz-zzzz-zzzzzzzzzzzz"));
    }

    // ── StoreKind::from_key_type ──────────────────────────────────────────

    #[test]
    fn classifies_doc_key_types() {
        assert_eq!(StoreKind::from_key_type("u64"), Some(StoreKind::Doc));
        assert_eq!(StoreKind::from_key_type("u128"), Some(StoreKind::Doc));
        assert_eq!(StoreKind::from_key_type("uuid"), Some(StoreKind::Doc));
    }

    #[test]
    fn classifies_kv_key_types() {
        assert_eq!(StoreKind::from_key_type("str"), Some(StoreKind::Kv));
        assert_eq!(StoreKind::from_key_type("int"), Some(StoreKind::Kv));
    }

    #[test]
    fn rejects_unknown_key_type() {
        assert_eq!(StoreKind::from_key_type("f64"), None);
    }

    // ── extract_doc_id ────────────────────────────────────────────────────

    #[test]
    fn doc_u64_from_number() {
        let doc = json!({ "id": 42, "name": "Alice" });
        let mut errs = Vec::new();
        assert_eq!(extract_doc_id(&doc, "id", DocKeyType::U64, 1, "", &mut errs).unwrap(), "42");
    }

    #[test]
    fn doc_u64_from_string() {
        let doc = json!({ "id": "99" });
        let mut errs = Vec::new();
        assert_eq!(extract_doc_id(&doc, "id", DocKeyType::U64, 1, "", &mut errs).unwrap(), "99");
    }

    #[test]
    fn doc_u128_from_number() {
        let doc = json!({ "id": 12345678901234567890u64 });
        let mut errs = Vec::new();
        assert_eq!(
            extract_doc_id(&doc, "id", DocKeyType::U128, 1, "", &mut errs).unwrap(),
            "12345678901234567890"
        );
    }

    #[test]
    fn doc_uuid_from_string() {
        let doc = json!({ "id": "550e8400-e29b-41d4-a716-446655440000" });
        let mut errs = Vec::new();
        assert_eq!(
            extract_doc_id(&doc, "id", DocKeyType::Uuid, 1, "", &mut errs).unwrap(),
            "550e8400-e29b-41d4-a716-446655440000"
        );
    }

    #[test]
    fn doc_missing_field_returns_none() {
        let doc = json!({ "name": "Dave" });
        let mut errs = Vec::new();
        assert!(extract_doc_id(&doc, "id", DocKeyType::U64, 1, "", &mut errs).is_none());
    }

    #[test]
    fn doc_bad_u64_value_returns_none() {
        let doc = json!({ "id": "not-a-number" });
        let mut errs = Vec::new();
        assert!(extract_doc_id(&doc, "id", DocKeyType::U64, 1, "", &mut errs).is_none());
    }

    #[test]
    fn doc_invalid_uuid_returns_none() {
        let doc = json!({ "id": "not-a-uuid" });
        let mut errs = Vec::new();
        assert!(extract_doc_id(&doc, "id", DocKeyType::Uuid, 1, "", &mut errs).is_none());
    }

    // ── extract_kv_key ────────────────────────────────────────────────────

    #[test]
    fn kv_str_key_from_string() {
        let row = json!({ "key": "job-42", "value": "…" });
        let mut errs = Vec::new();
        assert_eq!(extract_kv_key(&row, "key", KvKeyType::Str, 1, "", &mut errs).unwrap(), "job-42");
        assert!(errs.is_empty());
    }

    #[test]
    fn kv_str_key_rejects_number() {
        let row = json!({ "key": 42 });
        let mut errs = Vec::new();
        assert!(extract_kv_key(&row, "key", KvKeyType::Str, 1, "", &mut errs).is_none());
        assert_eq!(errs.len(), 1);
    }

    #[test]
    fn kv_int_key_from_number() {
        let row = json!({ "key": 99 });
        let mut errs = Vec::new();
        assert_eq!(extract_kv_key(&row, "key", KvKeyType::Int, 1, "", &mut errs).unwrap(), "99");
    }

    #[test]
    fn kv_int_key_from_numeric_string() {
        let row = json!({ "key": "100" });
        let mut errs = Vec::new();
        assert_eq!(extract_kv_key(&row, "key", KvKeyType::Int, 1, "", &mut errs).unwrap(), "100");
    }

    #[test]
    fn kv_missing_key_field_returns_none() {
        let row = json!({ "value": "…" });
        let mut errs = Vec::new();
        assert!(extract_kv_key(&row, "key", KvKeyType::Str, 1, "", &mut errs).is_none());
        assert_eq!(errs.len(), 1);
    }
}
