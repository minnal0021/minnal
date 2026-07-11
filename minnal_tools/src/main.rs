//! `minnal_tools` — stand-alone utility tools for minnal.
//!
//! # Usage
//!
//! ```text
//! minnal_tools <tool> [args...]
//! ```
//!
//! # Tools
//!
//! | Tool        | Description                                                          |
//! |-------------|----------------------------------------------------------------------|
//! | `bulk_load` | Load a JSONL file into a doc store (default) or KV store (`--kv`), optionally importing its schema first |
//!
//! # Example
//!
//! ```text
//! # Import a doc-store schema, then load into the new store
//! minnal_tools bulk_load --schema jobs-mini-schema.json http://localhost:8080 jobs jobId jobs-mini.jsonl
//!
//! # Import a KV-store schema, then load key/value pairs (--kv)
//! minnal_tools bulk_load --kv --schema job-content-kv-schema.json http://localhost:8080 job-content key value job-content-kv.jsonl
//!
//! # Load into a store that already exists
//! minnal_tools bulk_load http://localhost:8080 profiles id profiles.jsonl
//! ```

mod bulk_load;

fn usage() -> ! {
    eprintln!(concat!(
        "usage: minnal_tools <tool> [args...]\n",
        "\n",
        "tools:\n",
        "  bulk_load    load a JSONL file into a doc store (default) or KV store (--kv),\n",
        "               optionally importing its schema first\n",
        "\n",
        "run 'minnal_tools <tool>' with no further arguments for tool-specific help",
    ));
    std::process::exit(1);
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        usage();
    }

    let result = match args[1].as_str() {
        "bulk_load" => bulk_load::run(&args[2..]).await,
        other => {
            eprintln!("unknown tool: {other}");
            usage();
        }
    };

    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
