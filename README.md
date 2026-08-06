# arbo-ocr-rust

Rust wrapper for [arboOCR](https://github.com/wafik/ArboOCR) — runs the
prebuilt `arboocr_demo` binary via `std::process::Command`, no C++ build
required.

## Install

```toml
[dependencies]
arbo-ocr = { git = "https://github.com/ARBO-TEAM/arbo-ocr-rust" }
```

`Engine::new` downloads the matching arboOCR release binary (Windows or
Linux, auto-detected) the first time it's used if `Config.bin_path` is
`None` — see "How it works" below. Verified working end to end against
[`v0.1.0-php1`](https://github.com/wafik/ArboOCR/releases/tag/v0.1.0-php1)
on both platforms. If it fails anyway (offline, unsupported OS), download a
release manually from the
[arboOCR releases page](https://github.com/wafik/ArboOCR/releases) and pass
`Config.bin_path` explicitly.

You also need the OCR models — arboOCR does not bundle them. See
[Models](#models) below for exactly which files each `model_type` needs and
where to get them.

## Models

arboOCR doesn't bundle OCR models — you point `Config.models_dir` at a
folder of PP-OCRv6 ONNX files. Only the recognizer has size variants; the
detector is always one file regardless of `model_type`:

| File | Needed for | Varies by `model_type`? |
|---|---|---|
| `PP-OCRv6_det.onnx` | detection | no — always this one file |
| `PP-OCRv6_rec_tiny.onnx` + `PP-OCRv6_rec_tiny_dict.txt` | `model_type: "tiny"` | yes |
| `PP-OCRv6_rec_small.onnx` + `PP-OCRv6_rec_small_dict.txt` | `model_type: "small"` (default) | yes |
| `PP-OCRv6_rec_medium.onnx` + `PP-OCRv6_rec_medium_dict.txt` | `model_type: "medium"` | yes |
| `PP-OCRv6_cls.onnx` | angle classification, only if `use_angle_cls` | no |

You only need the recognizer size(s) you'll actually use — e.g. for
`model_type: "small"` alone, `models_dir` just needs `PP-OCRv6_det.onnx` +
`PP-OCRv6_rec_small.onnx` + `PP-OCRv6_rec_small_dict.txt`. Switching sizes
later is just changing `model_type`; `models_dir` can hold all three sizes
side by side if you want to switch freely.

**Getting the files** — arboOCR doesn't host default download URLs (see its
own [Models section](https://github.com/wafik/ArboOCR#models)), so pick
whichever applies:
- Already have a Python `rapidocr` install? Copy its `models/` directory
  over, renaming files to match the layout above.
- Have your own PP-OCRv6 ONNX export? Place/rename the files as above.
- A local arboOCR checkout's `models/` directory already has the detector,
  classifier, and all three recognizer sizes — handy for local dev (see the
  tiny-model example below).

## Usage

```rust
use arbo_ocr::{Config, Engine};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = Engine::new(Config {
        models_dir: Some("/path/to/models".to_string()),
        // bin_path: Some("/custom/path/to/arboocr_demo".into()), // optional override
        // model_type: Some("small".to_string()), // tiny/small/medium — default small
        // use_angle_cls: true,
        // use_cuda: true,
        ..Default::default()
    })?;

    let result = engine.recognize("/path/to/image.jpg")?;

    println!("{}", result.backend); // cpu / cuda / tensorrt
    for line in &result.lines {
        println!("{} ({:.3})", line.text, line.score);
    }
    Ok(())
}
```

An empty `result.lines` vec means no text was found — not an error.
`Engine::recognize` returns `Err(OcrError)` only when the process itself
fails to start, exits non-zero, or produces unparseable output.

## Quick example (tiny model, fastest)

For a fast local smoke test, use `model_type: "tiny"` — the
smallest/fastest PP-OCRv6 recognizer. If you have an arboOCR checkout
handy, its `models/` folder already contains the tiny det/rec/cls ONNX
files (no extra download):

```rust
use arbo_ocr::{Config, Engine};

let engine = Engine::new(Config {
    models_dir: Some("/path/to/arboOCR/models".to_string()), // e.g. a local arboOCR checkout's models/ dir
    model_type: Some("tiny".to_string()),
    ..Default::default()
})?;

let result = engine.recognize("/path/to/receipt.jpg")?;

println!(
    "backend={} lines={} elapsedMs={:.1}",
    result.backend,
    result.lines.len(),
    result.elapsed_ms
);
for line in &result.lines {
    println!("  {:<40} score={:.3}", line.text, line.score);
}
```

The `tiny` model trades some accuracy for speed — good for quick local
testing; switch to `small` (the default) or `medium` for production-quality
recognition.

## How it works

This crate never builds or vendors arboOCR's C++ source. It downloads the
exact same prebuilt release asset the PHP and Go packages use
(`arboocr-windows-x64.zip` / `arboocr-linux-x64.tar.gz` from the
[wafik/ArboOCR releases](https://github.com/wafik/ArboOCR/releases)) — the
compiled binary and its DLLs are language-agnostic, this crate just runs
the same CLI tool via `std::process::Command`.

Cargo has no build-time install hook like Composer's `post-install-cmd`, so
`Engine::new` downloads lazily instead, the same way arbo-ocr-go does:
calling `installer::ensure_installed(None)` when `Config.bin_path` is
`None`, caching the result under the OS user cache directory
(`%LOCALAPPDATA%\arbo-ocr-rust\<platform>` on Windows,
`$XDG_CACHE_HOME/arbo-ocr-rust/<platform>` or
`~/.cache/arbo-ocr-rust/<platform>` on Linux). Call
`installer::ensure_installed(Some(dir))` yourself (e.g. in a container build
step) if you want to control exactly when the download happens, then pass
the returned path as `Config.bin_path`.

Like the PHP and Go packages, OCR models are never bundled or
auto-downloaded — see [Models](#models) above for exactly which files you
need.

`recognize` captures the subprocess's output with `Command::output()`,
which reads stdout and stderr concurrently on separate threads internally —
`arboocr_demo` can write ~200KB of ONNXRuntime warnings to stderr before
any stdout appears, enough to deadlock a naive sequential-pipe-read
implementation, a class of bug the PHP and Go wrappers had to explicitly
design around.

Bool options (`use_angle_cls`, `use_cuda`, `use_tensorrt`, `use_fp16`,
`use_clahe`) are always sent as a single `--flag=value` token, never a bare
`--flag` followed by a separate `true`/`false` token — `arboocr_demo`'s
argument parser (cxxopts) only binds a bool flag's value via `=`; the
two-token form leaves the flag implicitly `true` regardless of the intended
value. This exact bug was found and fixed in both arbo-ocr-php and
arbo-ocr-go after it crashed `arboocr_demo` in production; this crate
avoids it from the start.

## Benchmark

`arbo-ocr-rust` was compared against arbo-ocr-php and arbo-ocr-go on the
same 5-image SROIE smoke set — all three call the identical `arboocr_demo`
binary, so accuracy is the same across all three; this measures wrapper
overhead only (subprocess spawn − arboocr_demo's own reported time):

| Size | arbo-php | arbo-go | arbo-rust |
|--------|----------:|---------:|-----------:|
| tiny | 193 ms | 137 ms | 131 ms |
| small | 231 ms | 171 ms | 172 ms |
| medium | 303 ms | 248 ms | 249 ms |

Go and Rust overhead is essentially tied — both are compiled binaries
paying only process-spawn cost, no interpreter startup. PHP runs ~55–65ms
higher (`php.exe` interpreter startup on top of `proc_open`). Same accuracy
across all three; all three match or beat a PP-OCRv6-based Node/Bun
reference implementation on this sample at every size. Full methodology in
the "wrapper benchmark" section of the internal `compare/RESULTS.md`
companion doc (not published in this repo).

## License

Apache-2.0
