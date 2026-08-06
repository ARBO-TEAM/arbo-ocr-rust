//! Rust wrapper for [arboOCR](https://github.com/wafik/ArboOCR) — runs the
//! prebuilt `arboocr_demo` binary via `std::process::Command`, no C++ build
//! required.

mod engine;
mod error;
pub mod installer;
mod types;

pub use engine::{Config, Engine};
pub use error::OcrError;
pub use types::{LineResult, PageResult, Point};
