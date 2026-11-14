//! The server's own tables. Everything else in the file is
//! `minnal_db::ConfigFile`'s, tested there.

use super::*;

#[test]
fn the_server_tables_are_read_and_default() {
    let cfg = DocStoreApiConfig::parse("").unwrap();
    assert_eq!(cfg.listen_addr(), DEFAULT_LISTEN_ADDR);
    assert_eq!(cfg.log_level(), DEFAULT_LOG_LEVEL);
    assert_eq!(cfg.log_dir(), PathBuf::from(minnal_db::defaults::storage::LOG_DIR));
    assert_eq!(cfg.file, ConfigFile::default());

    let cfg = DocStoreApiConfig::parse(
        "[api]\nlisten_addr = \"127.0.0.1:9\"\n[logging]\nlevel = \"debug\"\n[storage]\nlog_dir = \"/l\"\n[sharding]\nnum_buckets = 4\n",
    )
    .unwrap();
    assert_eq!(cfg.listen_addr(), "127.0.0.1:9");
    assert_eq!(cfg.log_level(), "debug");
    assert_eq!(cfg.log_dir(), PathBuf::from("/l"));
    assert_eq!(cfg.file.db_config().num_buckets, 4, "the database's tables reach its config");
}

#[test]
fn unknown_keys_and_tables_are_refused() {
    for bad in [
        "[api]\nlisten = \"x\"\n",
        "[logging]\nlvl = \"info\"\n",
        "[apii]\nlisten_addr = \"x\"\n",
        "[storage]\nschema_dir = \"/tmp/y\"\n",
        "[semantic_search]\nmodel = \"qwen\"\n",
    ] {
        assert!(DocStoreApiConfig::parse(bad).is_err(), "{bad}");
    }
}

#[test]
fn the_sample_config_loads() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../config/sample.toml");
    let cfg = DocStoreApiConfig::from_file(&path).unwrap();
    assert_eq!(cfg.listen_addr(), DEFAULT_LISTEN_ADDR);
    assert_eq!(cfg.file.semantic_search.supported_models().len(), 2);
}
