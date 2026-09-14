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
    /// Also emit a polygon per word (per character for CJK) — populates
    /// [`crate::LineResult::words`]. Off by default: the spans are nearly
    /// free to compute, but carrying them for every line of every page is
    /// not.
    pub word_boxes: bool,
    pub det_model_path: Option<String>,
    pub cls_model_path: Option<String>,
    pub rec_model_path: Option<String>,
    pub dict_path: Option<String>,
    /// Forbid the binary from fetching missing models — fail instead of
    /// downloading. Emits `--no-download` only when `true`; `false` (the
    /// default) leaves the flag off entirely, so the config stays compatible
    /// with binaries predating model auto-download.
    ///
    /// Live as of the pinned `v0.3.0`. Only relevant if you point
    /// [`Config::bin_path`] at an older binary, which does not know this
    /// flag and exits 1 on it.
    pub no_download: bool,
    /// Directory URL to fetch missing models from — an internal mirror
    /// instead of the default pinned models release. `None` = leave the flag
    /// off and let the binary use its own default,
    /// `https://github.com/ARBO-TEAM/arbo-ocr-models/releases/download/models-v1/`.
    ///
    /// Live as of the pinned `v0.3.0`. Only relevant if you point
    /// [`Config::bin_path`] at an older binary, which does not know this
    /// flag and exits 1 on it.
    pub models_url: Option<String>,
    /// Drop lines below this recognition confidence; `0.0` disables the
    /// filter. `None` leaves arboOCR's own default (0.5) in place.
    pub min_confidence: Option<f32>,
    /// Crops per recognition inference call. `None` = arboOCR's default (6).
    pub rec_batch_num: Option<u32>,
    /// Longest image side for the detection resize. `None` = arboOCR's
    /// default (960).
    pub det_limit_side_len: Option<u32>,
    /// `"debug"` | `"info"` | `"warn"` | `"error"`. `None` = arboOCR's
    /// default, which is silent. Worth setting when a run fails: as of
    /// v0.2.0 the binary writes nothing to stderr unless this is passed, so
    /// [`crate::OcrError::stderr`] is empty without it.
    pub log_level: Option<String>,
    /// Drop detection boxes at or below this area in detector-input pixels;
    /// `0.0` disables the filter. `None` leaves arboOCR's own default (20.0)
    /// in place. `Some(0.0)` is a meaningful value, which is why this is an
    /// `Option` rather than a bare `f32` whose zero would have to double as
    /// "unset" and make disabling the filter inexpressible.
    ///
    /// New in `v0.4.0`. Emitted only when `Some`, so a config that never sets
    /// it still builds an argv an older binary accepts — see
    /// [`Config::space_recovery`] for why that matters.
    pub min_det_box_area: Option<f32>,
    /// Recover inter-word spaces that greedy CTC decode swallows. `true`
    /// emits `--space-recovery=true`; `false` (the default) leaves the flag
    /// off entirely.
    ///
    /// New in `v0.4.0`, and off-by-default means "no token at all" rather
    /// than `--space-recovery=false`, for the same reason as
    /// [`Config::no_download`]: a [`Config::bin_path`] pointing at an older
    /// binary does not know this flag and exits 1 on it.
    pub space_recovery: bool,
    /// Leave ORT's CPU memory arena on: faster, higher RSS. `true` emits
    /// `--enable-cpu-mem-arena=true`; `false` (the default) leaves the flag
    /// off entirely, for the same `v0.4.0` reason as above.
    pub enable_cpu_mem_arena: bool,
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

    /// OCR many images with **one** `arboocr_demo` process
    /// (`--images-from <list> --json`), returning one [`PageResult`] per input
    /// in input order.
    ///
    /// [`Engine::recognize`] starts a fresh process per image, and that
    /// process start plus model load dominates a short page; this pays it once
    /// for the whole list instead.
    ///
    /// Results are matched to inputs **by position** because the binary
    /// reports only a basename. That is sound only while the counts agree, so
    /// a mismatch is an error rather than a shifted list.
    ///
    /// A batch exits `1` when *any* image came back empty. That is an ordinary
    /// outcome, not a failure, and is tolerated as long as the JSON array is
    /// still on stdout — a usage error exits `1` too but leaves stdout empty,
    /// and that one is an error.
    pub fn recognize_batch(&self, image_paths: &[String]) -> Result<Vec<PageResult>, OcrError> {
        if image_paths.is_empty() {
            return Ok(Vec::new());
        }

        // The list file is newline-delimited and the binary skips blank lines
        // and '#' lines as comments, so a path in either shape would be
        // dropped silently and shift every later result onto the wrong input.
        for (i, path) in image_paths.iter().enumerate() {
            if path.is_empty() {
                return Err(OcrError {
                    message: format!("recognize_batch: image_paths[{i}] is empty"),
                    exit_code: None,
                    stderr: String::new(),
                });
            }
            if path.contains('\n') || path.contains('\r') {
                return Err(OcrError {
                    message: format!(
                        "recognize_batch: image_paths[{i}] contains a newline, which the image list format cannot represent: {path:?}"
                    ),
                    exit_code: None,
                    stderr: String::new(),
                });
            }
            if path.trim_start_matches([' ', '\t']).starts_with('#') {
                return Err(OcrError {
                    message: format!(
                        "recognize_batch: image_paths[{i}] starts with '#', which arboocr_demo reads as a comment and would skip: {path:?}"
                    ),
                    exit_code: None,
                    stderr: String::new(),
                });
            }
        }

        // No `tempfile` dependency for one list file: the pid plus a
        // nanosecond stamp is unique enough, and the file is removed on the
        // way out below.
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let list_path = std::env::temp_dir()
            .join(format!("arbo-ocr-rust-{}-{stamp}.txt", std::process::id()));

        std::fs::write(&list_path, format!("{}\n", image_paths.join("\n"))).map_err(|e| {
            OcrError {
                message: format!("could not write image list: {e}"),
                exit_code: None,
                stderr: String::new(),
            }
        })?;

        let mut args = vec![
            "--images-from".to_string(),
            list_path.to_string_lossy().into_owned(),
            "--json".to_string(),
        ];
        args.extend(self.flags_from_config());

        // Same Command::output() concurrency note as recognize().
        let output = Command::new(&self.bin_path).args(&args).output();
        let _ = std::fs::remove_file(&list_path);

        let output = output.map_err(|e| OcrError {
            message: format!("could not start process: {e}"),
            exit_code: None,
            stderr: String::new(),
        })?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let trimmed = stdout.trim();

        // The binary overloads exit 1: "a page had no text" and "you passed a
        // bad flag" share it. Only the first leaves the JSON array on stdout,
        // so requiring that payload keeps a usage error an error.
        if !output.status.success()
            && !(output.status.code() == Some(1) && trimmed.starts_with('['))
        {
            return Err(OcrError {
                message: format!(
                    "arboocr_demo exited with code {}",
                    output.status.code().unwrap_or(-1)
                ),
                exit_code: output.status.code(),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            });
        }

        let pages: Vec<PageResult> = serde_json::from_str(trimmed).map_err(|_| {
            let raw: String = trimmed.chars().take(500).collect();
            OcrError {
                message: format!(
                    "arboocr_demo --images-from produced unparseable output: {raw}"
                ),
                exit_code: None,
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            }
        })?;

        // Every check above was positional, so a short or long array has to
        // fail here rather than shift text onto the wrong file.
        if pages.len() != image_paths.len() {
            return Err(OcrError {
                message: format!(
                    "arboocr_demo returned {} results for {} images; cannot match results to inputs by position",
                    pages.len(),
                    image_paths.len()
                ),
                exit_code: None,
                stderr: String::new(),
            });
        }

        Ok(pages)
    }

    /// Mirrors Engine.php's `flagsFromOptions()` / arbo-ocr-go's
    /// `flagsFromConfig()`: string and numeric fields emit `--flag-name`,
    /// `<value>` only when set; the six bool fields always emit a single
    /// `--flag-name=value` token — cxxopts only binds a bool flag's value
    /// via "=", so a bare "--flag" followed by a separate "true"/"false"
    /// token would leave the flag implicitly true and the value ignored.
    /// That quirk is bool-only: options taking a value parse the following
    /// argv token fine, so the two-token form is correct for them.
    fn flags_from_config(&self) -> Vec<String> {
        let mut flags = Vec::new();

        let string_flags: [(&Option<String>, &str); 9] = [
            (&self.cfg.models_dir, "models-dir"),
            (&self.cfg.ocr_version, "ocr-version"),
            (&self.cfg.model_type, "model-type"),
            (&self.cfg.det_model_path, "det-model"),
            (&self.cfg.cls_model_path, "cls-model"),
            (&self.cfg.rec_model_path, "rec-model"),
            (&self.cfg.dict_path, "dict"),
            (&self.cfg.models_url, "models-url"),
            (&self.cfg.log_level, "log-level"),
        ];
        for (value, flag) in string_flags {
            if let Some(v) = value {
                flags.push(format!("--{flag}"));
                flags.push(v.clone());
            }
        }

        // Left off entirely when None so arboOCR keeps its own defaults
        // (min-confidence 0.5, rec-batch-num 6, det-limit-side-len 960)
        // rather than us restating them and having to track future changes.
        if let Some(v) = self.cfg.min_confidence {
            flags.push("--min-confidence".to_string());
            flags.push(v.to_string());
        }
        // v0.4.0 flag, so `None` must stay off the argv: a pre-v0.4.0 binary
        // treats it as an unknown option. `Some(0.0)` does emit, and is the
        // documented way to disable the area filter.
        if let Some(v) = self.cfg.min_det_box_area {
            flags.push("--min-det-box-area".to_string());
            flags.push(v.to_string());
        }
        let uint_flags: [(&Option<u32>, &str); 2] = [
            (&self.cfg.rec_batch_num, "rec-batch-num"),
            (&self.cfg.det_limit_side_len, "det-limit-side-len"),
        ];
        for (value, flag) in uint_flags {
            if let Some(v) = value {
                flags.push(format!("--{flag}"));
                flags.push(v.to_string());
            }
        }

        let bool_flags: [(bool, &str); 6] = [
            (self.cfg.use_angle_cls, "angle"),
            (self.cfg.use_cuda, "cuda"),
            (self.cfg.use_tensorrt, "tensorrt"),
            (self.cfg.use_fp16, "fp16"),
            (self.cfg.use_clahe, "clahe"),
            (self.cfg.word_boxes, "word-boxes"),
        ];
        for (value, flag) in bool_flags {
            flags.push(format!("--{flag}={value}"));
        }

        // The odd one out: emitted only when true, unlike the six above.
        // `--no-download` arrived with model auto-download in v0.3.0, so any
        // binary older than the pin treats it as an unknown option — and
        // cxxopts answers an unknown option with a usage error and exit 1,
        // failing every recognize() call. Off-by-default therefore has to
        // mean "no token at all", not "--no-download=false", which keeps a
        // default Config runnable against a bin_path pointing at an older
        // release. Keeps the "=" form when it is emitted for the same
        // cxxopts reason as the block above.
        if self.cfg.no_download {
            flags.push("--no-download=true".to_string());
        }

        // Same opt-in-only shape as --no-download above, for the same reason:
        // both arrived in v0.4.0, and `false` is the binary's own default, so
        // restating it as `--flag=false` would add nothing but a usage error
        // against an older bin_path.
        if self.cfg.space_recovery {
            flags.push("--space-recovery=true".to_string());
        }
        if self.cfg.enable_cpu_mem_arena {
            flags.push("--enable-cpu-mem-arena=true".to_string());
        }

        flags
    }

    /// Prefetches the models for this config's `ocr_version`/`model_type`
    /// into arboOCR's own model cache by running `arboocr_demo
    /// --download-models`, which fetches and exits without doing any OCR.
    /// Returns the binary's per-file report (one `ok`/`skipped`/`absent`/
    /// `MISSING` line per model file) on success.
    ///
    /// Useful in a CI step or a Docker build layer so the first real
    /// [`Engine::recognize`] does not pay for the download mid-request: a
    /// missing model is fetched on demand anyway, so this is a
    /// warm-the-cache convenience rather than a prerequisite. It is the
    /// companion to [`installer::ensure_installed`] in a container build —
    /// that call bakes in the *binary*, this one bakes in the *weights*,
    /// which live in a separate cache.
    ///
    /// Live as of the pinned `v0.3.0`. A [`Config::bin_path`] pointing at an
    /// older binary has no `--download-models` flag and answers with a usage
    /// error and exit code 1.
    pub fn download_models(&self) -> Result<String, OcrError> {
        let mut args = vec!["--download-models".to_string()];
        args.extend(self.flags_from_config());

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
                    "arboocr_demo --download-models exited with code {}",
                    output.status.code().unwrap_or(-1)
                ),
                exit_code: output.status.code(),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            });
        }

        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
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
                word_boxes: true,
                ..Default::default()
            },
        };
        let flags = engine.flags_from_config();

        assert!(flags.contains(&"--angle=false".to_string()));
        assert!(flags.contains(&"--cuda=true".to_string()));
        assert!(flags.contains(&"--tensorrt=false".to_string()));
        assert!(flags.contains(&"--fp16=false".to_string()));
        assert!(flags.contains(&"--clahe=true".to_string()));
        assert!(flags.contains(&"--word-boxes=true".to_string()));
        for bare in [
            "--angle",
            "--cuda",
            "--tensorrt",
            "--fp16",
            "--clahe",
            "--word-boxes",
        ] {
            assert!(
                !flags.contains(&bare.to_string()),
                "{bare} must not appear as a bare token"
            );
        }
    }

    /// The value-taking options are the mirror image of the bool ones: they
    /// use the two-token form, and they must be absent entirely when unset
    /// so arboOCR applies its own defaults instead of ours.
    #[test]
    fn numeric_flags_are_two_token_and_omitted_when_unset() {
        let unset = Engine {
            bin_path: PathBuf::new(),
            cfg: Config::default(),
        };
        let flags = unset.flags_from_config();
        for absent in ["--min-confidence", "--rec-batch-num", "--det-limit-side-len"] {
            assert!(
                !flags.iter().any(|f| f.starts_with(absent)),
                "{absent} must not be emitted when unset"
            );
        }

        let set = Engine {
            bin_path: PathBuf::new(),
            cfg: Config {
                min_confidence: Some(0.25),
                rec_batch_num: Some(8),
                det_limit_side_len: Some(1280),
                log_level: Some("warn".to_string()),
                ..Default::default()
            },
        };
        let flags = set.flags_from_config();
        for (flag, value) in [
            ("--min-confidence", "0.25"),
            ("--rec-batch-num", "8"),
            ("--det-limit-side-len", "1280"),
            ("--log-level", "warn"),
        ] {
            let i = flags
                .iter()
                .position(|f| f == flag)
                .unwrap_or_else(|| panic!("{flag} missing"));
            assert_eq!(flags[i + 1], value, "{flag} value must be the next token");
        }
    }

    /// A default Config must produce a command line byte-identical to the
    /// pre-auto-download one — no `--no-download=false`, no empty
    /// `--models-url`. Two reasons, both still live now that the pin is
    /// v0.3.0: emitting nothing lets the binary apply its own defaults
    /// (download enabled, official models URL), and it keeps a default
    /// Config runnable against a `bin_path` aimed at a pre-v0.3.0 release,
    /// whose cxxopts parser answers an unknown option with a usage error and
    /// exit 1.
    #[test]
    fn download_flags_are_absent_from_a_default_config() {
        let unset = Engine {
            bin_path: PathBuf::new(),
            cfg: Config::default(),
        };
        let flags = unset.flags_from_config();

        for absent in ["--no-download", "--models-url", "--download-models"] {
            assert!(
                !flags.iter().any(|f| f.starts_with(absent)),
                "{absent} must not be emitted by a default Config"
            );
        }
    }

    /// `no_download` is a bool but not one of the always-emitted six, so it
    /// needs its own check that setting it produces the "=" form cxxopts
    /// requires; `models_url` is an ordinary two-token string option.
    #[test]
    fn download_flags_are_emitted_when_set() {
        let set = Engine {
            bin_path: PathBuf::new(),
            cfg: Config {
                no_download: true,
                models_url: Some("https://mirror.internal/arboocr/models/".to_string()),
                ..Default::default()
            },
        };
        let flags = set.flags_from_config();

        assert!(flags.contains(&"--no-download=true".to_string()));
        assert!(
            !flags.contains(&"--no-download".to_string()),
            "--no-download must not appear as a bare token"
        );

        let i = flags
            .iter()
            .position(|f| f == "--models-url")
            .expect("--models-url missing");
        assert_eq!(flags[i + 1], "https://mirror.internal/arboocr/models/");
    }

    /// The three v0.4.0 flags are all opt-in-only, for the same reason as
    /// `--no-download`: a `bin_path` aimed at an older release does not know
    /// them, and cxxopts answers an unknown option with a usage error and
    /// exit 1. A default Config must put nothing on the argv, and the two
    /// bools must stay silent even when explicitly set to `false` — that is
    /// the binary's own default, so restating it buys nothing and costs a
    /// broken run against a pre-v0.4.0 binary.
    #[test]
    fn v040_flags_are_opt_in_only() {
        let defaults = Engine {
            bin_path: PathBuf::new(),
            cfg: Config::default(),
        };
        let flags = defaults.flags_from_config();
        for absent in [
            "--min-det-box-area",
            "--space-recovery",
            "--enable-cpu-mem-arena",
        ] {
            assert!(
                !flags.iter().any(|f| f.starts_with(absent)),
                "{absent} must not be emitted by a default Config"
            );
        }

        let explicit_false = Engine {
            bin_path: PathBuf::new(),
            cfg: Config {
                space_recovery: false,
                enable_cpu_mem_arena: false,
                ..Default::default()
            },
        };
        let flags = explicit_false.flags_from_config();
        for absent in ["--space-recovery", "--enable-cpu-mem-arena"] {
            assert!(
                !flags.iter().any(|f| f.starts_with(absent)),
                "{absent} must not be emitted when false"
            );
        }
    }

    /// `0` is a real value for `min_det_box_area` — it disables the filter —
    /// which is why the field is an `Option<f32>` and not a bare `f32` whose
    /// zero would have to double as "unset" and make that inexpressible.
    #[test]
    fn min_det_box_area_zero_is_a_real_value() {
        let engine = Engine {
            bin_path: PathBuf::new(),
            cfg: Config {
                min_det_box_area: Some(0.0),
                space_recovery: true,
                enable_cpu_mem_arena: true,
                ..Default::default()
            },
        };
        let flags = engine.flags_from_config();

        let i = flags
            .iter()
            .position(|f| f == "--min-det-box-area")
            .expect("--min-det-box-area missing");
        assert_eq!(flags[i + 1], "0", "an explicit 0 must reach the binary");

        assert!(flags.contains(&"--space-recovery=true".to_string()));
        assert!(flags.contains(&"--enable-cpu-mem-arena=true".to_string()));
        for bare in ["--space-recovery", "--enable-cpu-mem-arena"] {
            assert!(
                !flags.contains(&bare.to_string()),
                "{bare} must not appear as a bare token"
            );
        }
    }
}
