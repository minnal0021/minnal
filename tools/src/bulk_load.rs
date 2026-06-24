//! `bulk_load` — load a JSONL file into a minnal doc store via the REST API,
//! optionally importing the store's schema first.
//!
//! # Usage
//!
//! ```text
//! minnal_tools bulk_load [--no-wal] [--schema <schema.json>] <url> <namespace> <id_field> <data.jsonl>
//! ```
//!
//! # Positional arguments
//!
//! | Argument     | Description                                               |
//! |--------------|-----------------------------------------------------------|
//! | `url`        | Base URL of the running doc store REST API               |
//! | `namespace`  | Name of the doc store namespace to load into             |
//! | `id_field`   | JSON field name whose value is the document ID           |
//! | `data.jsonl` | Full path to the JSONL file (one JSON object per line)   |
//!
//! # Flags
//!
//! | Flag                | Description                                                 |
//! |---------------------|-------------------------------------------------------------|
//! | `--schema <file>`   | Import the schema (`POST /admin/stores/import`) before      |
//! |                     | loading.  The schema file's `namespace` must match the      |
//! |                     | `namespace` argument.  An existing store is reused, so       |
//! |                     | re-runs are safe.  Without this flag the namespace must      |
//! |                     | already exist.                                              |
//! | `--no-wal`          | Bypass WAL writes for maximum throughput.  Data written     |
//! |                     | this way is unrecoverable on a crash — only use when         |
//! |                     | re-running the load is acceptable (e.g. initial bulk         |
//! |                     | imports from a source of truth).                            |
//!
//! Attribute validation is performed by the REST service — lines whose PUT
//! request returns an error are counted as skipped with the reason logged to a
//! sibling `.errors` file.
//!
//! # ID field rules
//!
//! The `id_field` value is parsed according to the namespace's `key_type`:
//!
//! | `key_type` | Expected JSON value                     |
//! |------------|-----------------------------------------|
//! | `u64`      | JSON number or numeric string           |
//! | `u128`     | JSON number or numeric string           |
//! | `uuid`     | UUID string (`xxxxxxxx-xxxx-…`)         |
//!
//! Lines with a missing or unparseable ID field are skipped with a warning.
//! The ID field is **not** removed from the stored document.
//!
//! # Examples
//!
//! ```text
//! # Import a schema, then load into the new store (one step from a fresh server)
//! minnal_tools bulk_load --schema ./jobs-schema.json http://localhost:8080 jobs jobId ./jobs.jsonl
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
//! is running, document loads still succeed and attribute queries still work —
//! only semantic-search queries return nothing until embeddings are produced.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use reqwest::{Client, StatusCode};
use serde::Deserialize;
use serde_json::Value;

// ── Key type ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
pub(crate) enum KeyType {
    #[serde(rename = "u64")]
    U64,
    #[serde(rename = "u128")]
    U128,
    #[serde(rename = "uuid")]
    Uuid,
}

// ── Schema response ───────────────────────────────────────────────────────────

/// Minimal projection of the store schema returned by `GET /stores`.
#[derive(Debug, Deserialize)]
struct StoreSchema {
    namespace: String,
    key_type: KeyType,
}

// ── ID helpers ────────────────────────────────────────────────────────────────

/// Extract the document ID from `doc[id_field]` and format it as a URL path
/// segment, according to the namespace's `key_type`.
///
/// Returns `None` (with a printed warning) if the field is absent or the value
/// cannot be parsed.
#[cfg(test)]
fn extract_id_str(doc: &Value, id_field: &str, key_type: KeyType, line_no: usize) -> Option<String> {
    let raw = match doc.get(id_field) {
        Some(v) => v,
        None => {
            eprintln!("  line {line_no}: missing id field '{id_field}' — skipped");
            return None;
        }
    };

    let result = match key_type {
        KeyType::U64 => match raw {
            Value::Number(n) => n.as_u64().map(|v| v.to_string()),
            Value::String(s) => s.parse::<u64>().ok().map(|v| v.to_string()),
            _ => None,
        },
        KeyType::U128 => match raw {
            Value::Number(n) => n.as_u128().map(|v| v.to_string()),
            Value::String(s) => s.parse::<u128>().ok().map(|v| v.to_string()),
            _ => None,
        },
        KeyType::Uuid => match raw {
            Value::String(s) if is_valid_uuid(s) => Some(s.clone()),
            _ => None,
        },
    };

    if result.is_none() {
        eprintln!("  line {line_no}: cannot parse '{raw}' as {key_type:?} — skipped");
    }
    result
}

/// Returns `true` if `s` contains exactly 32 ASCII hex digits (with or without
/// hyphens in the standard UUID positions).
fn is_valid_uuid(s: &str) -> bool {
    let hex_count = s.chars().filter(|c| c.is_ascii_hexdigit()).count();
    hex_count == 32
}

// ── Bulk load ───────────────────────────────────────────────────────────────

fn usage() -> ! {
    eprintln!(concat!(
        "usage: minnal_tools bulk_load [--no-wal] [--schema <schema.json>] <url> <namespace> <id_field> <data.jsonl>\n",
        "\n",
        "arguments:\n",
        "  url         base URL of the running doc store REST API\n",
        "  namespace   name of the namespace to load into\n",
        "  id_field    JSON field name whose value is the document ID\n",
        "  data.jsonl  full path to the JSONL file (one JSON object per line)\n",
        "\n",
        "flags:\n",
        "  --schema <schema.json>  import the schema before loading (an existing store\n",
        "                          is reused, so re-runs are safe); the schema's\n",
        "                          'namespace' must match the namespace argument. Without\n",
        "                          this flag the namespace must already exist\n",
        "  --no-wal                bypass WAL writes for maximum throughput; data written\n",
        "                          this way is unrecoverable on a crash — only use when\n",
        "                          re-running the load is acceptable\n",
        "\n",
        "examples:\n",
        "  minnal_tools bulk_load --schema ./jobs-schema.json http://localhost:8080 jobs jobId ./jobs.jsonl\n",
        "  minnal_tools bulk_load http://localhost:8080 users id ./users.jsonl\n",
        "  minnal_tools bulk_load --no-wal http://localhost:8080 users id ./users.jsonl",
    ));
    std::process::exit(1);
}

/// Entry point called from `main.rs` with the arguments after the tool name.
pub async fn run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    // ── Parse flags and positional arguments ──────────────────────────────
    let mut skip_wal = false;
    let mut schema_path: Option<PathBuf> = None;
    let mut positional: Vec<&String> = Vec::new();

    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--no-wal" => skip_wal = true,
            "--schema" => {
                let path = iter.next().ok_or("--schema requires a <schema.json> path argument")?;
                schema_path = Some(PathBuf::from(path));
            }
            flag if flag.starts_with("--") => return Err(format!("unknown flag: {flag}").into()),
            _ => positional.push(arg),
        }
    }

    if positional.len() != 4 {
        usage();
    }

    let base_url = positional[0].trim_end_matches('/');
    let namespace = positional[1];
    let id_field = positional[2];
    let jsonl_path = PathBuf::from(positional[3]);

    let client = Client::new();

    if skip_wal {
        println!("WAL disabled — maximum throughput mode (data unrecoverable on crash)");
    }

    // ── Optionally import the schema before loading ───────────────────────
    if let Some(schema_path) = &schema_path {
        import_schema(&client, base_url, schema_path, namespace).await?;
    }

    let key_type = resolve_key_type(&client, base_url, namespace).await?;
    println!("namespace '{namespace}' found  key_type={key_type:?}");

    load_jsonl(&client, base_url, namespace, id_field, key_type, &jsonl_path, skip_wal).await
}

/// Import a doc store schema via `POST /admin/stores/import`.  An existing store
/// (HTTP 409 Conflict) is reused, so re-runs are safe.  The schema file's
/// `namespace` field must match `namespace`.
async fn import_schema(client: &Client, base_url: &str, schema_path: &Path, namespace: &str) -> Result<(), Box<dyn std::error::Error>> {
    let schema_bytes = std::fs::read(schema_path).map_err(|e| format!("cannot open '{}': {e}", schema_path.display()))?;
    let schema: Value = serde_json::from_slice(&schema_bytes).map_err(|e| format!("'{}' is not valid JSON: {e}", schema_path.display()))?;

    let schema_ns = schema
        .get("namespace")
        .and_then(Value::as_str)
        .ok_or("schema file has no string 'namespace' field")?;
    if schema_ns != namespace {
        return Err(format!("namespace mismatch: argument is '{namespace}' but schema declares '{schema_ns}'").into());
    }

    let resp = client
        .post(format!("{base_url}/admin/stores/import"))
        .json(&schema)
        .send()
        .await
        .map_err(|e| format!("cannot reach '{base_url}/admin/stores/import': {e}"))?;

    match resp.status() {
        s if s.is_success() => println!("schema imported — store '{namespace}' created"),
        StatusCode::CONFLICT => println!("store '{namespace}' already exists — reusing it"),
        s => {
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("schema import failed ({s}): {body}").into());
        }
    }
    Ok(())
}

/// Resolve a namespace's `key_type` via `GET /stores`, erroring if the
/// namespace does not exist.
async fn resolve_key_type(client: &Client, base_url: &str, namespace: &str) -> Result<KeyType, Box<dyn std::error::Error>> {
    let stores: Vec<StoreSchema> = client
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

/// Stream a JSONL file into an existing `namespace`, PUTting one document per
/// line.  Lines that fail validation are counted as skipped and written to a
/// sibling `.errors` file.
async fn load_jsonl(
    client: &Client,
    base_url: &str,
    namespace: &str,
    id_field: &str,
    key_type: KeyType,
    jsonl_path: &std::path::Path,
    skip_wal: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("id_field='{id_field}'");

    // ── Stream JSONL file ─────────────────────────────────────────────────
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

        let id_str = match extract_id_str_collecting(&doc, id_field, key_type, line_no, trimmed, &mut error_lines) {
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

    let elapsed = started.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();

    // ── Write error file if there were any failures ───────────────────────
    let error_path = jsonl_path.with_extension("errors");
    if !error_lines.is_empty() {
        let ef = File::create(&error_path).map_err(|e| format!("cannot create error file '{}': {e}", error_path.display()))?;
        let mut w = BufWriter::new(ef);
        for entry in &error_lines {
            writeln!(w, "{entry}").map_err(|e| format!("write error: {e}"))?;
        }
        eprintln!("{} error(s) written to '{}'", error_lines.len(), error_path.display());
    }

    println!(
        "done  loaded={loaded}  skipped={skipped}  total={}  elapsed={elapsed_secs:.2}s",
        loaded + skipped
    );
    Ok(())
}

/// Like [`extract_id_str`] but appends the error message to `errors` when it
/// returns `None`, so callers don't have to duplicate the collection logic.
fn extract_id_str_collecting(
    doc: &Value,
    id_field: &str,
    key_type: KeyType,
    line_no: usize,
    raw_line: &str,
    errors: &mut Vec<String>,
) -> Option<String> {
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
        KeyType::U64 => match raw {
            Value::Number(n) => n.as_u64().map(|v| v.to_string()),
            Value::String(s) => s.parse::<u64>().ok().map(|v| v.to_string()),
            _ => None,
        },
        KeyType::U128 => match raw {
            Value::Number(n) => n.as_u128().map(|v| v.to_string()),
            Value::String(s) => s.parse::<u128>().ok().map(|v| v.to_string()),
            _ => None,
        },
        KeyType::Uuid => match raw {
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

    // ── extract_id_str ────────────────────────────────────────────────────

    #[test]
    fn u64_from_number() {
        let doc = json!({ "id": 42, "name": "Alice" });
        assert_eq!(extract_id_str(&doc, "id", KeyType::U64, 1).unwrap(), "42");
    }

    #[test]
    fn u64_from_string() {
        let doc = json!({ "id": "99" });
        assert_eq!(extract_id_str(&doc, "id", KeyType::U64, 1).unwrap(), "99");
    }

    #[test]
    fn u128_from_number() {
        let doc = json!({ "id": 12345678901234567890u64 });
        assert_eq!(extract_id_str(&doc, "id", KeyType::U128, 1).unwrap(), "12345678901234567890");
    }

    #[test]
    fn uuid_from_string() {
        let doc = json!({ "id": "550e8400-e29b-41d4-a716-446655440000" });
        assert_eq!(
            extract_id_str(&doc, "id", KeyType::Uuid, 1).unwrap(),
            "550e8400-e29b-41d4-a716-446655440000"
        );
    }

    #[test]
    fn missing_field_returns_none() {
        let doc = json!({ "name": "Dave" });
        assert!(extract_id_str(&doc, "id", KeyType::U64, 1).is_none());
    }

    #[test]
    fn bad_u64_value_returns_none() {
        let doc = json!({ "id": "not-a-number" });
        assert!(extract_id_str(&doc, "id", KeyType::U64, 1).is_none());
    }

    #[test]
    fn invalid_uuid_returns_none() {
        let doc = json!({ "id": "not-a-uuid" });
        assert!(extract_id_str(&doc, "id", KeyType::Uuid, 1).is_none());
    }
}
