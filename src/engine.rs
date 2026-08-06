use std::path::PathBuf;
use std::process::Command;

use crate::error::OcrError;
use crate::installer;
use crate::types::PageResult;

/// Configures an [`Engine`]: where to find the `arboocr_demo` binary and
/// which CLI flags to pass it on every [`Engine::recognize`] call.
#[derive(Debug, Default, Clone)]
pub struct Config {
    /// Explicit path to `arboocr_demo`; `None` = lazily download via
    /// [`installer::ensure_installed`] into the default cache dir.
    pub bin_path: Option<PathBuf>,
    pub models_dir: Option<String>,
    pub ocr_version: Option<String>,
    pub model_type: Option<String>,
    pub use_angle_cls: bool,
    pub use_cuda: bool,
    pub use_tensorrt: bool,
    pub use_fp16: bool,
    pub use_clahe: bool,
    pub det_model_path: Option<String>,
    pub cls_model_path: Option<String>,
    pub rec_model_path: Option<String>,
    pub dict_path: Option<String>,
}

/// Runs the prebuilt `arboocr_demo` binary via `std::process::Command` and
/// parses its `--json` output. Requires no C++ build — only the binary
/// [`installer::ensure_installed`] downloaded (or one you point at
/// manually via [`Config::bin_path`]).
pub struct Engine {
    bin_path: PathBuf,
    cfg: Config,
}

impl Engine {
    /// Resolves the binary path — using `cfg.bin_path` as-is if set, or
    /// lazily downloading via [`installer::ensure_installed`] if it's
    /// `None` — and returns a ready `Engine`, or an error if the binary
    /// can't be found/installed.
    pub fn new(cfg: Config) -> Result<Self, OcrError> {
        let bin_path = match &cfg.bin_path {
            Some(p) => {
                if !p.is_file() {
                    return Err(OcrError {
                        message: format!(
                            "arboocr_demo binary not found at {}. Pass a valid bin_path or leave it None to auto-install.",
                            p.display()
                        ),
                        exit_code: None,
                        stderr: String::new(),
                    });
                }
                p.clone()
            }
            None => installer::ensure_installed(None).map_err(|e| OcrError {
                message: e,
                exit_code: None,
                stderr: String::new(),
            })?,
        };

        Ok(Engine { bin_path, cfg })
    }

    /// Runs `arboocr_demo --image <image_path> --json` (plus flags derived
    /// from `Config`) as a subprocess and parses its JSON stdout into a
    /// [`PageResult`]. Returns an error only when the process can't be
    /// started, exits non-zero, or its stdout isn't valid JSON. An empty
    /// `PageResult.lines` is a normal, successful result — not an error.
    pub fn recognize(&self, image_path: &str) -> Result<PageResult, OcrError> {
        let mut args = vec!["--image".to_string(), image_path.to_string(), "--json".to_string()];
        args.extend(self.flags_from_config());

        // Command::output() reads stdout and stderr concurrently on
        // separate threads internally, so there's no deadlock risk even
        // though arboocr_demo can write ~200KB of ONNXRuntime
        // schema-registration warnings to stderr before producing any
        // stdout — unlike a naive sequential read of two pipes.
        let output = Command::new(&self.bin_path)
            .args(&args)
            .output()
            .map_err(|e| OcrError {
                message: format!("could not start process: {e}"),
                exit_code: None,
                stderr: String::new(),
            })?;

        if !output.status.success() {
            return Err(OcrError {
                message: format!(
                    "arboocr_demo exited with code {}",
                    output.status.code().unwrap_or(-1)
                ),
                exit_code: output.status.code(),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            });
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let trimmed = stdout.trim();
        serde_json::from_str(trimmed).map_err(|_| {
            let raw: String = trimmed.chars().take(500).collect();
            OcrError {
                message: format!("arboocr_demo --json produced unparseable output: {raw}"),
                exit_code: None,
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            }
        })
    }

    /// Mirrors Engine.php's `flagsFromOptions()` / arbo-ocr-go's
    /// `flagsFromConfig()`: string fields emit `--flag-name`, `<value>`
    /// only when set; the five bool fields always emit a single
    /// `--flag-name=value` token — cxxopts only binds a bool flag's value
    /// via "=", so a bare "--flag" followed by a separate "true"/"false"
    /// token would leave the flag implicitly true and the value ignored.
    fn flags_from_config(&self) -> Vec<String> {
        let mut flags = Vec::new();

        let string_flags: [(&Option<String>, &str); 7] = [
            (&self.cfg.models_dir, "models-dir"),
            (&self.cfg.ocr_version, "ocr-version"),
            (&self.cfg.model_type, "model-type"),
            (&self.cfg.det_model_path, "det-model"),
            (&self.cfg.cls_model_path, "cls-model"),
            (&self.cfg.rec_model_path, "rec-model"),
            (&self.cfg.dict_path, "dict"),
        ];
        for (value, flag) in string_flags {
            if let Some(v) = value {
                flags.push(format!("--{flag}"));
                flags.push(v.clone());
            }
        }

        let bool_flags: [(bool, &str); 5] = [
            (self.cfg.use_angle_cls, "angle"),
            (self.cfg.use_cuda, "cuda"),
            (self.cfg.use_tensorrt, "tensorrt"),
            (self.cfg.use_fp16, "fp16"),
            (self.cfg.use_clahe, "clahe"),
        ];
        for (value, flag) in bool_flags {
            flags.push(format!("--{flag}={value}"));
        }

        flags
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test: cxxopts only binds a bool flag's value via "=" — a
    /// bare "--angle" followed by a separate "true"/"false" token leaves
    /// the flag implicitly true and the value ignored (confirmed against
    /// the real arboocr_demo binary; the identical bug was found and fixed
    /// in arbo-ocr-php and arbo-ocr-go). flags_from_config must always emit
    /// bool options as a single "--flag=value" token.
    #[test]
    fn bool_flags_use_single_token_form() {
        let engine = Engine {
            bin_path: PathBuf::new(),
            cfg: Config {
                use_angle_cls: false,
                use_cuda: true,
                use_tensorrt: false,
                use_fp16: false,
                use_clahe: true,
                ..Default::default()
            },
        };
        let flags = engine.flags_from_config();

        assert!(flags.contains(&"--angle=false".to_string()));
        assert!(flags.contains(&"--cuda=true".to_string()));
        assert!(flags.contains(&"--tensorrt=false".to_string()));
        assert!(flags.contains(&"--fp16=false".to_string()));
        assert!(flags.contains(&"--clahe=true".to_string()));
        for bare in ["--angle", "--cuda", "--tensorrt", "--fp16", "--clahe"] {
            assert!(
                !flags.contains(&bare.to_string()),
                "{bare} must not appear as a bare token"
            );
        }
    }
}
