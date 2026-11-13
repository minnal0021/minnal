use super::*;

#[test]
fn supported_models_are_checked_and_optional() {
    let base = "[storage]\ndb_path = \"/tmp/x\"\n";
    let cfg: DocStoreApiConfig = toml::from_str(base).unwrap();
    cfg.validate_supported_models().unwrap();
    assert!(cfg.semantic_search.resolve().supported_models().is_empty(), "any model by default");
    let listed = format!(
        "{base}[[semantic_search.supported_models]]\nname = \"Gemma\"\ndimension = 768\n\
         [[semantic_search.supported_models]]\nname = \"e5\"\ndimension = 1024\n"
    );
    let cfg: DocStoreApiConfig = toml::from_str(&listed).unwrap();
    cfg.validate_supported_models().unwrap();
    assert_eq!(
        cfg.semantic_search.resolve().supported_models(),
        vec![("gemma".to_string(), 768), ("e5".to_string(), 1024)]
    );
    for bad in ["name = \" \"\ndimension = 768", "name = \"x\"\ndimension = 0"] {
        let toml = format!("{base}[[semantic_search.supported_models]]\n{bad}\n");
        let cfg: DocStoreApiConfig = toml::from_str(&toml).unwrap();
        assert!(cfg.validate_supported_models().is_err(), "{bad}");
    }
    let twice = format!(
        "{base}[[semantic_search.supported_models]]\nname = \"qwen\"\ndimension = 768\n\
         [[semantic_search.supported_models]]\nname = \"QWEN\"\ndimension = 768\n"
    );
    let cfg: DocStoreApiConfig = toml::from_str(&twice).unwrap();
    assert!(cfg.validate_supported_models().is_err());
}

#[test]
fn resolve_carries_embedding_timeouts() {
    // Defaults: request 30s, connect 10s. resolve must surface both as Durations,
    // with connect shorter than the overall request cap.
    let section = SemanticSearchSection::default();
    assert_eq!(section.embedding_request_timeout_secs, 30);
    assert_eq!(section.embedding_connect_timeout_secs, 10);
    let resolved = section.resolve();
    assert_eq!(resolved.embedding_request_timeout, std::time::Duration::from_secs(30));
    assert_eq!(resolved.embedding_connect_timeout, std::time::Duration::from_secs(10));
    assert!(resolved.embedding_connect_timeout < resolved.embedding_request_timeout);
}

#[test]
fn a_leftover_schema_dir_is_rejected() {
    let err = toml::from_str::<DocStoreApiConfig>("[storage]\ndb_path = \"/tmp/x\"\nschema_dir = \"/tmp/y\"\n")
        .unwrap_err()
        .to_string();
    assert!(err.contains("unknown field") && err.contains("schema_dir"), "{err}");
}

#[test]
fn a_namespace_setting_left_in_the_server_config_is_rejected() {
    for key in [
        "model = \"qwen\"",
        "embedding_dim = 768",
        "n_probes = 64",
        "max_probes = 1024",
        "window_size = 4",
        "cluster_path = \"x\"",
        "centroid_dir = \"x\"",
    ] {
        let toml = format!("[storage]\ndb_path = \"/tmp/x\"\n[semantic_search]\n{key}\n");
        let err = toml::from_str::<DocStoreApiConfig>(&toml).unwrap_err().to_string();
        assert!(err.contains("unknown field"), "{key}: {err}");
    }
    let ok = "[storage]\ndb_path = \"/tmp/x\"\n[semantic_search]\nembedding_service_url = \"http://h:1\"\n";
    let cfg: DocStoreApiConfig = toml::from_str(ok).unwrap();
    assert_eq!(cfg.semantic_search.embedding_service_url, "http://h:1");
}

#[test]
fn maintenance_and_search_defaults_are_read_and_validated() {
    let base = "[storage]\ndb_path = \"/tmp/x\"\n";
    let cfg: DocStoreApiConfig = toml::from_str(base).unwrap();
    let d = cfg.semantic_search.index_defaults().unwrap();
    assert_eq!(d, minnal_db::doc_store::schema::vector_settings::IndexDefaults::default());
    assert_eq!(
        cfg.semantic_search.maintenance.threads().unwrap(),
        minnal_db::defaults::semantic_search::maintenance::THREADS
    );

    let toml = format!(
        "{base}[semantic_search.maintenance]\ntarget_posting_size = 256\nskip_uncertain_moves = true\nthreads = 4\n\
         [semantic_search.search_defaults]\nmax_probes = 200\n"
    );
    let cfg: DocStoreApiConfig = toml::from_str(&toml).unwrap();
    let d = cfg.semantic_search.index_defaults().unwrap();
    assert_eq!((d.maintenance.target_posting_size, d.maintenance.skip_uncertain_moves), (256, true));
    assert_eq!(d.maintenance.reassign_range, 64, "unset keys keep the built-in value");
    assert_eq!(d.search.max_probes, 200);
    assert_eq!(cfg.semantic_search.maintenance.threads().unwrap(), 4);
    assert_eq!(cfg.semantic_search.maintenance.rebuild_threads().unwrap(), 2);

    for (table, key) in [
        ("maintenance", "target_posting_size = 4"),
        ("maintenance", "merge_ratio = 0.7"),
        ("maintenance", "threads = 0"),
        ("maintenance", "rebuild_threads = 3"),
        ("search_defaults", "max_probes = 0"),
    ] {
        let toml = format!("{base}[semantic_search.{table}]\n{key}\n");
        let cfg: DocStoreApiConfig = toml::from_str(&toml).unwrap();
        assert!(cfg.semantic_search.index_defaults().is_err(), "{key}");
    }
    for (table, key) in [
        ("maintenance", "error_bound_gate = true"),
        ("search_defaults", "n_probes = 64"),
        ("search_defaults", "probe_budget_entries = 40000"),
    ] {
        let toml = format!("{base}[semantic_search.{table}]\n{key}\n");
        assert!(toml::from_str::<DocStoreApiConfig>(&toml).is_err(), "{key}");
    }
}

#[test]
fn the_sample_config_parses_with_every_commented_default_switched_on() {
    let sample = include_str!("../../../config/sample.toml");
    let cfg: DocStoreApiConfig = toml::from_str(sample).unwrap();
    cfg.semantic_search.index_defaults().unwrap();
    // Uncomment the documented defaults of the two semantic-search tables:
    // they must parse and equal the built-in values.
    let mut table = "";
    let uncommented: String = sample
        .lines()
        .map(|l| {
            if l.starts_with('[') {
                table = l;
            }
            match l.strip_prefix("# ") {
                Some(kv) if table.starts_with("[semantic_search.") && kv.contains(" = ") && !kv.contains("fraction") && !kv.contains("floor") => {
                    kv.to_string()
                }
                _ => l.to_string(),
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    let cfg: DocStoreApiConfig = toml::from_str(&uncommented).unwrap();
    let d = cfg.semantic_search.index_defaults().unwrap();
    let builtin = minnal_db::doc_store::schema::vector_settings::IndexDefaults::default();
    assert_eq!(d.search, builtin.search);
    assert_eq!(d.maintenance.target_posting_size, builtin.maintenance.target_posting_size);
    assert!((d.maintenance.merge_ratio.0 - builtin.maintenance.merge_ratio.0).abs() < 1e-4);
}

#[test]
fn service_config_carries_only_service_settings() {
    let section = SemanticSearchSection {
        embedding_service_url: "http://h:1".into(),
        embedding_request_timeout_secs: 7,
        ..SemanticSearchSection::default()
    };
    let c = section.resolve().service_config();
    assert_eq!(c.embedding_service_url, "http://h:1");
    assert_eq!(c.embedding_request_timeout, std::time::Duration::from_secs(7));
}
