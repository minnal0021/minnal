//! Tests of the lifecycle layer on its own, against a fake host: the jobs
//! registry, and the governor running creations, drops and jobs, queueing them
//! and supervised.

mod fake_host;
mod governor;
mod jobs;
mod queueing;
mod supervision;
