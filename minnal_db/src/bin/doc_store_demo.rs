use std::sync::Arc;

use minnal_db::semantic_search::index::vector_index::QueryResult;
use minnal_db::semantic_search::{ClusterIndex, service::SemanticSearchConfig};
use minnal_db::{
    AttributeDef, AttributeType, DocId, DocStore, DocStoreError, DocStoreSchema, IndexSpec, IndexType, KeyType, Pagination, SchemaAmendment,
    SemanticSearchContext, StoreType,
};

type Result<T> = std::result::Result<T, DocStoreError>;

// ── helpers ───────────────────────────────────────────────────────────────────

fn section(title: &str) {
    log::info!("\n── {} ──────────────────────────────────────────", title);
}

fn show_doc(id: DocId, doc: &serde_json::Value) {
    log::info!("  [{:?}] {}", id, serde_json::to_string(doc).unwrap_or_default());
}

fn user_schema() -> DocStoreSchema {
    DocStoreSchema {
        store_type: StoreType::Doc,
        namespace: "users".to_owned(),
        ns_id: None,
        key_type: KeyType::U64,
        attributes: vec![],
        indices: vec![
            IndexSpec {
                field: "status".to_owned(),
                index_type: IndexType::Str,
            },
            IndexSpec {
                field: "verified".to_owned(),
                index_type: IndexType::Bool,
            },
        ],
        semantic_search_enabled: false,
        embedding_fields: vec![],
    }
}

// ── demo functions ────────────────────────────────────────────────────────────

/// Create a doc store, verify it appears in the list, then inspect the schema.
async fn demo_create_and_list(store: &DocStore) -> Result<()> {
    section("Create & List");

    store.create(user_schema()).await?;
    log::info!("created 'users' store (status:Str index, verified:Bool index)");

    // Create a second store — no indices, KV-only
    store
        .create(DocStoreSchema {
            store_type: StoreType::Doc,
            namespace: "events".to_owned(),
            ns_id: None,
            key_type: KeyType::U64,
            attributes: vec![],
            indices: vec![],
            semantic_search_enabled: false,
            embedding_fields: vec![],
        })
        .await?;
    log::info!("created 'events' store (no indices)");

    let list = store.list()?;
    log::info!("doc stores ({}):", list.len());
    for entry in &list {
        log::info!(
            "  namespace={} key_type={} indices={}",
            entry["namespace"].as_str().unwrap_or("?"),
            entry["key_type"].as_str().unwrap_or("?"),
            entry["indices"].as_array().map(|a| a.len()).unwrap_or(0),
        );
    }

    Ok(())
}

/// Insert, update, retrieve, and delete documents.
async fn demo_crud(store: &DocStore) -> Result<()> {
    section("CRUD");

    let docs = [
        (
            1u64,
            serde_json::json!({ "name": "Alice", "status": "active",   "verified": true,  "age": 30 }),
        ),
        (
            2u64,
            serde_json::json!({ "name": "Bob",   "status": "inactive", "verified": false, "age": 25 }),
        ),
        (
            3u64,
            serde_json::json!({ "name": "Carol", "status": "active",   "verified": true,  "age": 40 }),
        ),
        (
            4u64,
            serde_json::json!({ "name": "Dave",  "status": "active",   "verified": false, "age": 17 }),
        ),
    ];

    for (id, doc) in &docs {
        store.put("users", DocId::U64(*id), doc.clone()).await?;
    }
    log::info!("inserted {} documents", docs.len());

    let found = store.get("users", DocId::U64(2)).await?;
    log::info!("get(2) = {}", found.as_ref().map(|d| d["name"].as_str().unwrap_or("?")).unwrap_or("None"));

    // Update Bob's status
    store
        .put(
            "users",
            DocId::U64(2),
            serde_json::json!({ "name": "Bob", "status": "active", "verified": true, "age": 26 }),
        )
        .await?;
    let updated = store.get("users", DocId::U64(2)).await?;
    log::info!(
        "after update(2): status={}",
        updated.as_ref().and_then(|d| d["status"].as_str()).unwrap_or("?")
    );

    // Delete Dave
    store.delete("users", DocId::U64(4)).await?;
    let gone = store.get("users", DocId::U64(4)).await?;
    log::info!("after delete(4): get = {:?}", gone);

    Ok(())
}

/// Range scan by document ID.
async fn demo_range_query(store: &DocStore) -> Result<()> {
    section("Range Query");

    // Insert ordered numeric docs into 'events'
    for i in 1u64..=8 {
        store
            .put(
                "events",
                DocId::U64(i),
                serde_json::json!({ "seq": i, "type": if i % 2 == 0 { "even" } else { "odd" } }),
            )
            .await?;
    }
    log::info!("inserted events 1–8");

    let range = store.scan_range("events", DocId::U64(3), Some(DocId::U64(6)), None, 100).await?;
    log::info!("range [3, 6) — {} results:", range.results.len());
    for (id, doc) in &range.results {
        show_doc(*id, doc);
    }

    let tail = store.scan_range("events", DocId::U64(6), None, None, 100).await?;
    log::info!("range [6, ∞) — {} results:", tail.results.len());
    for (id, doc) in &tail.results {
        show_doc(*id, doc);
    }

    Ok(())
}

/// Predicate queries using field indices.
async fn demo_index_query(store: &DocStore) -> Result<()> {
    section("Index Query");

    let active = store.query("users", "status = \"active\"", Pagination::default()).await?;
    log::info!("status=active ({} results):", active.total);
    for (id, doc) in &active.results {
        show_doc(*id, doc);
    }

    let verified = store.query("users", "verified = true", Pagination::default()).await?;
    log::info!("verified=true ({} results):", verified.total);
    for (id, doc) in &verified.results {
        show_doc(*id, doc);
    }

    let inactive_unverified = store
        .query("users", "status = \"inactive\" OR verified = false", Pagination::default())
        .await?;
    log::info!("status=inactive OR verified=false ({} results):", inactive_unverified.total);
    for (id, doc) in &inactive_unverified.results {
        show_doc(*id, doc);
    }

    Ok(())
}

/// Add a new index on an existing store with a background build + progress polling.
async fn demo_add_index(store: &DocStore) -> Result<()> {
    section("Add Index (background build)");

    // The 'age' field exists in the stored JSON but was not originally indexed
    let spec = IndexSpec {
        field: "age".to_owned(),
        index_type: IndexType::Int,
    };
    log::info!("adding 'age' (Int) index to 'users'…");

    let handle = store.add_index("users", spec).await?;

    // Poll progress until done
    loop {
        let p = handle.progress();
        log::info!("  build progress: {}/{} docs indexed, done={}", p.indexed, p.total, p.done);
        if p.done {
            break;
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }

    handle.wait().await?;
    log::info!("index build complete");

    // Query using the new index
    let adults = store.query("users", "age >= 26", Pagination::default()).await?;
    log::info!("age>=26 after index build ({} results):", adults.total);
    for (id, doc) in &adults.results {
        show_doc(*id, doc);
    }

    Ok(())
}

/// Drop an existing index — field is demoted to a non-indexed attribute in the schema.
async fn demo_drop_index(store: &DocStore) -> Result<()> {
    section("Drop Index");

    let before = minnal_db::DocStoreSchema::load(std::path::Path::new("/tmp/doc_store_demo/schemas"), "users").unwrap();
    log::info!(
        "before drop: indices={:?}, attributes={}",
        before.indices.iter().map(|i| i.field.as_str()).collect::<Vec<_>>(),
        before.attributes.len(),
    );

    store.drop_index("users", "verified")?;
    log::info!("dropped 'verified' index");

    let after = minnal_db::DocStoreSchema::load(std::path::Path::new("/tmp/doc_store_demo/schemas"), "users").unwrap();
    log::info!(
        "after drop: indices={:?}, attributes={:?}",
        after.indices.iter().map(|i| i.field.as_str()).collect::<Vec<_>>(),
        after
            .attributes
            .iter()
            .map(|a| format!("{}:{:?}", a.name, a.attr_type))
            .collect::<Vec<_>>(),
    );
    log::info!("  → 'verified' preserved as non-indexed attribute (data still in documents)");

    Ok(())
}

/// Add, update, and remove non-indexed attributes from the schema.
async fn demo_amend_schema(store: &DocStore) -> Result<()> {
    section("Amend Schema (non-indexed attributes)");

    store.amend(
        "users",
        SchemaAmendment::AddAttribute {
            name: "email".to_owned(),
            attr_type: AttributeType::Str,
            description: Some("user email address".to_owned()),
        },
    )?;
    log::info!("added 'email' attribute");

    store.amend(
        "users",
        SchemaAmendment::AddAttribute {
            name: "score".to_owned(),
            attr_type: AttributeType::Int,
            description: None,
        },
    )?;
    log::info!("added 'score' attribute");

    store.amend(
        "users",
        SchemaAmendment::UpdateAttribute {
            name: "score".to_owned(),
            attr_type: AttributeType::Int,
            description: Some("loyalty score (0–100)".to_owned()),
        },
    )?;
    log::info!("updated 'score' description");

    store.amend("users", SchemaAmendment::RemoveAttribute { name: "email".to_owned() })?;
    log::info!("removed 'email' attribute");

    // Attempting to remove an indexed attribute must fail
    let err = store
        .amend("users", SchemaAmendment::RemoveAttribute { name: "status".to_owned() })
        .unwrap_err();
    log::info!("expected error removing indexed 'status': {}", err);

    let schema = minnal_db::DocStoreSchema::load(std::path::Path::new("/tmp/doc_store_demo/schemas"), "users").unwrap();
    log::info!(
        "final attributes: {:?}",
        schema.attributes.iter().map(|a| a.name.as_str()).collect::<Vec<_>>()
    );

    Ok(())
}

/// Drop a store — all data, index files, and schema are removed.
async fn demo_drop_store(store: &DocStore) -> Result<()> {
    section("Drop Store");

    let before = store.list()?.len();
    log::info!("stores before drop: {}", before);

    store.remove("events").await?;
    log::info!("dropped 'events'");

    let after = store.list()?.len();
    log::info!("stores after drop: {} (schema file deleted, data and index dirs removed)", after);

    Ok(())
}

/// Restart demo — create a store, shut it down, reopen, and verify
/// schema + indices are restored automatically from config.json.
async fn demo_restart(db_path: &str, schema_dir: &str) -> Result<()> {
    section("Schema Persistence across Restart");

    let _ = std::fs::remove_dir_all(db_path);
    let _ = std::fs::remove_dir_all(schema_dir);

    // ── Pass 1: create + write ────────────────────────────────────────────
    log::info!("[pass 1] creating store and writing documents");
    {
        let store = DocStore::open(db_path, schema_dir).await?;
        store
            .create(DocStoreSchema {
                store_type: StoreType::Doc,
                namespace: "products".to_owned(),
                ns_id: None,
                key_type: KeyType::U64,
                attributes: vec![],
                indices: vec![
                    IndexSpec {
                        field: "category".to_owned(),
                        index_type: IndexType::Str,
                    },
                    IndexSpec {
                        field: "in_stock".to_owned(),
                        index_type: IndexType::Bool,
                    },
                ],
                semantic_search_enabled: false,
                embedding_fields: vec![],
            })
            .await?;

        let docs = [
            (
                1u64,
                serde_json::json!({ "name": "Laptop",  "category": "electronics", "in_stock": true  }),
            ),
            (
                2u64,
                serde_json::json!({ "name": "T-shirt", "category": "apparel",     "in_stock": true  }),
            ),
            (
                3u64,
                serde_json::json!({ "name": "Desk",    "category": "furniture",   "in_stock": false }),
            ),
            (
                4u64,
                serde_json::json!({ "name": "Monitor", "category": "electronics", "in_stock": true  }),
            ),
        ];
        for (id, doc) in &docs {
            store.put("products", DocId::U64(*id), doc.clone()).await?;
        }
        log::info!("[pass 1] inserted {} products, shutting down", docs.len());
    }

    // ── Pass 2: reopen — no create() call ────────────────────────────────
    log::info!("[pass 2] reopening — schema + indices loaded from config.json automatically");
    {
        let store = DocStore::open(db_path, schema_dir).await?;

        let list = store.list()?;
        log::info!("[pass 2] found {} store(s) in schema_dir", list.len());

        // Indices are active immediately — no register_index_field call needed
        let electronics = store.query("products", "category = \"electronics\"", Pagination::default()).await?;
        log::info!("[pass 2] category=electronics ({} results):", electronics.total);
        for (id, doc) in &electronics.results {
            show_doc(*id, doc);
        }

        let in_stock = store.query("products", "in_stock = true", Pagination::default()).await?;
        log::info!("[pass 2] in_stock=true ({} results):", in_stock.total);
        for (id, doc) in &in_stock.results {
            show_doc(*id, doc);
        }
    }

    let _ = std::fs::remove_dir_all(db_path);
    let _ = std::fs::remove_dir_all(schema_dir);
    Ok(())
}

// ── Semantic search demo ──────────────────────────────────────────────────────

/// Build a DocStore with a SemanticSearchContext attached.
///
/// Loads the IVF cluster index from `service/embedding_support/qwen/clusters.json`
/// (relative to the workspace root) and configures the embedding service at
/// `http://localhost:8000/embeddings`.
async fn open_store_with_semantic_search(db_path: &str, schema_dir: &str) -> Result<DocStore> {
    // Load the IVF cluster index bundled with the repo, pinning the centroid
    // dimension to the embedding dim the demo config uses.
    let cluster_index = ClusterIndex::load_with_dim(
        "service/embedding_support/qwen/clusters.json",
        SemanticSearchConfig::default().embedding_dim,
    )
    .map_err(|e| DocStoreError::EmbeddingFailed(format!("cluster load failed: {e}")))?;

    let store = DocStore::open(db_path, schema_dir).await?;

    let ctx = SemanticSearchContext {
        config: SemanticSearchConfig::default(), // localhost:8000, 768-dim, 4-bit quant
        cluster_index: Arc::new(cluster_index),
    };

    Ok(store.with_semantic_search(ctx))
}

/// Create a 'jobs' namespace with semantic search on the 'description' field,
/// insert a handful of job postings, run a few queries, then show results.
async fn demo_semantic_search(store: &DocStore) -> Result<()> {
    section("Semantic Search");

    // Schema: semantic search on 'description'; 'title' is a regular Str index.
    store
        .create(DocStoreSchema {
            store_type: StoreType::Doc,
            namespace: "jobs".to_owned(),
            ns_id: None,
            key_type: KeyType::U64,
            attributes: vec![AttributeDef {
                name: "description".to_owned(),
                attr_type: AttributeType::Str,
                description: None,
            }],
            indices: vec![IndexSpec {
                field: "title".to_owned(),
                index_type: IndexType::Str,
            }],
            semantic_search_enabled: true,
            embedding_fields: vec!["description".to_owned()],
        })
        .await?;
    log::info!("created 'jobs' store (semantic search on 'description')");

    // Insert representative job postings.
    let jobs: &[(u64, &str, &str)] = &[
        (
            1,
            "Senior Rust Engineer",
            "Design and build high-performance systems using Rust. Work on memory-safe \
          concurrency primitives, async runtimes, and low-latency storage engines.",
        ),
        (
            2,
            "Machine Learning Engineer",
            "Develop and deploy large-scale ML models. Experience with PyTorch, distributed \
          training, and model serving infrastructure required.",
        ),
        (
            3,
            "Frontend Engineer",
            "Build responsive web applications with React and TypeScript. Collaborate with \
          designers to implement pixel-perfect UIs with great accessibility.",
        ),
        (
            4,
            "Data Engineer",
            "Build reliable data pipelines and ETL workflows. Proficiency in Spark, dbt, \
          and cloud data warehouses such as BigQuery or Snowflake.",
        ),
        (
            5,
            "DevOps Engineer",
            "Maintain Kubernetes clusters, CI/CD pipelines, and cloud infrastructure on AWS. \
          Automate deployments with Terraform and Helm.",
        ),
        (
            6,
            "Embedded Systems Engineer",
            "Write bare-metal firmware in C and Rust for microcontrollers. Optimise for \
          constrained memory and real-time requirements.",
        ),
        (
            7,
            "NLP Research Scientist",
            "Research and implement state-of-the-art NLP models. Deep expertise in \
          transformers, tokenisation, and retrieval-augmented generation.",
        ),
        (
            8,
            "Backend Engineer",
            "Build scalable REST and gRPC services in Go. Design database schemas, \
          write integration tests, and participate in on-call rotations.",
        ),
    ];

    for (id, title, description) in jobs {
        store
            .put(
                "jobs",
                DocId::U64(*id),
                serde_json::json!({
                    "title": title,
                    "description": description,
                }),
            )
            .await?;
        log::info!("  indexed [{id}] {title}");
    }

    // Run semantic queries and display top-3 results with their titles.
    let queries = [
        "systems programming low latency",
        "natural language processing and transformers",
        "cloud infrastructure automation",
    ];

    for query in queries {
        let page = store.search_semantic("jobs", query, None, Pagination::default()).await?;
        let top: Vec<&QueryResult> = page.results.iter().take(3).collect();

        log::info!("\n  query: \"{query}\"");
        for (rank, r) in top.iter().enumerate() {
            // Resolve title from the doc store for display.
            let key_type = minnal_db::KeyType::U64;
            let doc_id = minnal_db::DocId::from_bytes(&r.document_id, key_type).unwrap_or(DocId::U64(0));
            let title = store
                .get("jobs", doc_id)
                .await?
                .and_then(|d| d["title"].as_str().map(str::to_owned))
                .unwrap_or_else(|| "?".to_owned());
            log::info!("    #{rank} [{doc_id:?}] {title}  (dot={:.4}, err={:.4})", r.dot_product, r.error_bound);
        }
    }

    // Demonstrate delete: remove job 2, re-run the NLP query.
    store.delete("jobs", DocId::U64(2)).await?;
    log::info!("\n  deleted job 2 (ML Engineer) — re-running NLP query");
    let after_delete = store
        .search_semantic("jobs", "natural language processing", None, Pagination::default())
        .await?;
    let top_after: Vec<&QueryResult> = after_delete.results.iter().take(3).collect();
    for (rank, r) in top_after.iter().enumerate() {
        let key_type = minnal_db::KeyType::U64;
        let doc_id = DocId::from_bytes(&r.document_id, key_type).unwrap_or(DocId::U64(0));
        let title = store
            .get("jobs", doc_id)
            .await?
            .and_then(|d| d["title"].as_str().map(str::to_owned))
            .unwrap_or_else(|| "?".to_owned());
        log::info!("    #{rank} [{doc_id:?}] {title}  (dot={:.4})", r.dot_product);
    }

    // ── Embedding cache demo ──────────────────────────────────────────────────
    //
    // Re-run the same queries a second time.  The embedding service is only
    // called on the *first* run; the second run is served entirely from the
    // system-wide TTL cache (`system_qemb_cache`), so no HTTP round-trip to
    // the embedding service is made.
    //
    // The results should be byte-for-byte identical to the first run, which
    // confirms that the cached vectors are being used correctly.
    log::info!("\n── Embedding cache demo ────────────────────────────────────");
    log::info!("  Re-running the same queries — embeddings should be served");
    log::info!("  from the system-wide cache (system_qemb_cache), not the");
    log::info!("  embedding service.");

    for query in queries {
        let t0 = std::time::Instant::now();
        let cached_page = store.search_semantic("jobs", query, None, Pagination::default()).await?;
        let elapsed = t0.elapsed();

        let top: Vec<&QueryResult> = cached_page.results.iter().take(3).collect();
        log::info!("\n  query (cached): \"{query}\"  [{elapsed:?}]");
        for (rank, r) in top.iter().enumerate() {
            let key_type = minnal_db::KeyType::U64;
            let doc_id = minnal_db::DocId::from_bytes(&r.document_id, key_type).unwrap_or(DocId::U64(0));
            let title = store
                .get("jobs", doc_id)
                .await?
                .and_then(|d| d["title"].as_str().map(str::to_owned))
                .unwrap_or_else(|| "?".to_owned());
            log::info!("    #{rank} [{doc_id:?}] {title}  (dot={:.4}, err={:.4})", r.dot_product, r.error_bound);
        }
    }

    Ok(())
}

// ── main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    // ── Standard demos (no embedding service required) ────────────────────
    let db_path = "/tmp/doc_store_demo/db";
    let schema_dir = "/tmp/doc_store_demo/schemas";
    let _ = std::fs::remove_dir_all("/tmp/doc_store_demo");

    let log_dir = "/tmp/doc_store_demo/log";
    std::fs::create_dir_all(log_dir).expect("failed to create log directory");
    let file_appender = tracing_appender::rolling::daily(log_dir, "doc_store_demo.log");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

    use tracing_subscriber::prelude::*;
    let filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".parse().unwrap());
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_subscriber::fmt::layer().with_writer(non_blocking).with_ansi(false))
        .with(filter)
        .init();

    let store = Arc::new(DocStore::open(db_path, schema_dir).await?);
    log::info!("[DocStore] opened at {db_path}");

    demo_create_and_list(&store).await?;
    demo_crud(&store).await?;
    demo_range_query(&store).await?;
    demo_index_query(&store).await?;
    demo_add_index(&store).await?;
    demo_drop_index(&store).await?;
    demo_amend_schema(&store).await?;
    demo_drop_store(&store).await?;

    let _ = std::fs::remove_dir_all("/tmp/doc_store_demo");

    demo_restart("/tmp/doc_store_restart/db", "/tmp/doc_store_restart/schemas").await?;

    // ── Semantic search demo (requires embedding service on :8000) ────────
    let sem_db = "/tmp/doc_store_semantic/db";
    let sem_schema = "/tmp/doc_store_semantic/schemas";
    let _ = std::fs::remove_dir_all("/tmp/doc_store_semantic");

    log::info!("\nopening store with semantic search context…");
    match open_store_with_semantic_search(sem_db, sem_schema).await {
        Ok(sem_store) => {
            demo_semantic_search(&sem_store).await?;
            let _ = std::fs::remove_dir_all("/tmp/doc_store_semantic");
        }
        Err(e) => {
            log::warn!("semantic search demo skipped: {e}");
            log::warn!("  → start the embedding service on http://localhost:8000/embeddings to enable it");
        }
    }

    section("Done");
    log::info!("all demos complete");
    Ok(())
}
