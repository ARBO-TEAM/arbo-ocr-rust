use std::fmt;

/// Error from [`crate::Engine::recognize`] or [`crate::Engine::new`] — only
/// returned when the process itself can't be started, exits non-zero, or
/// produces unparseable output. An empty `PageResult.lines` is a normal,
/// successful result, not an error.
#[derive(Debug)]
pub struct OcrError {
    pub message: String,
    pub exit_code: Option<i32>,
    pub stderr: String,
}

impl fmt::Display for OcrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for OcrError {}
