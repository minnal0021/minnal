//! Evaluations of search quality, run by hand on real datasets: `#[ignore]`d
//! tests that go through the production write and search paths and report
//! relevance, recall, estimator error and partition shape. Speed benchmarks of
//! the public API are in `minnal_db/benches/`; these live in the crate because
//! they need its internals.
//!
//! - [`beir`]: `eval_beir`, a BEIR corpus through the live embedding service,
//!   and the dataset readers and relevance scoring the others share.
//! - [`frozen_eval`]: `eval_frozen` and `eval_compare`, the main evaluation, on
//!   saved embeddings; also `eval_worker_completion` and `eval_residual_spread`.
//! - [`frozen`]: dumps and loads the saved embeddings. [`exact`]: the exact
//!   ranking they are compared against. [`indexing`]: embeds and indexes a
//!   corpus through the live service, for `eval_beir`.
//! - [`code_width`], [`pass1_width`], [`study`]: focused experiments on the
//!   saved embeddings (`eval_code_width`, `eval_pass1_width`, `eval_pass1_study`,
//!   `eval_rotation_spread`).
//!
//! Each module's docs give its run command and settings.

mod beir;
mod code_width;
mod exact;
mod frozen;
mod frozen_eval;
mod indexing;
mod pass1_width;
mod study;
