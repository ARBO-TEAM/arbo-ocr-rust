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
`None` — see "How it works" below. The pinned release is
[`v0.3.0`](https://github.com/wafik/ArboOCR/releases/tag/v0.4.0). If the
download fails anyway (offline, unsupported OS), grab a release manually
from the
[arboOCR releases page](https://github.com/wafik/ArboOCR/releases) and pass
`Config.bin_path` explicitly.

The OCR models are handled for you too. On first run the binary fetches any
model file it needs but can't find, SHA-256 verifies it, and caches it per
user, so the snippet above is genuinely all the setup there is. Pointing
`Config.models_dir` at a folder that already holds the files still wins and
still means zero network traffic — see [Models](#models) for the file
layout, the cache location, and how to turn downloading off.

## Models

arboOCR doesn't bundle OCR models in the release archive, but as of the
pinned `v0.4.0` it fetches them itself: any PP-OCRv6 file that isn't already
on disk is downloaded, SHA-256 verified, and cached per user before it's
used. A default `Config` therefore needs no model setup at all. Populating
`Config.models_dir` yourself is now an optimisation, not a prerequisite.

Which files a run actually needs, whether you're supplying them or letting
it fetch them — only the recognizer has size variants; the detector is
always one file regardless of `model_type`:

| File | Needed for | Varies by `model_type`? |
|---|---|---|
| `PP-OCRv6_det.onnx` | detection | no — always this one file |
| `PP-OCRv6_rec_tiny.onnx` + `PP-OCRv6_rec_tiny_dict.txt` | `model_type: "tiny"` | yes |
| `PP-OCRv6_rec_small.onnx` + `PP-OCRv6_rec_small_dict.txt` | `model_type: "small"` (default) | yes |
| `PP-OCRv6_rec_medium.onnx` + `PP-OCRv6_rec_medium_dict.txt` | `model_type: "medium"` | yes |
| `PP-OCRv6_cls.onnx` | angle classification, only if `use_angle_cls` | no |

Only the recognizer size(s) you actually use are involved — e.g. for
`model_type: "small"` alone, that's `PP-OCRv6_det.onnx` +
`PP-OCRv6_rec_small.onnx` + `PP-OCRv6_rec_small_dict.txt`. Switching sizes
later is just changing `model_type`, and the new size is fetched on demand;
`models_dir` can hold all three sizes side by side if you'd rather have them
all local.

**Supplying the files yourself** is the deliberate alternative to letting
them download — worth it for an air-gapped host, an image with the weights
baked in, or a first run that must not touch the network. Pick whichever
applies:
- Already have a Python `rapidocr` install? Copy its `models/` directory
  over, renaming files to match the layout above.
- Have your own PP-OCRv6 ONNX export? Place/rename the files as above.
- A local arboOCR checkout's `models/` directory already has the detector,
  classifier, and all three recognizer sizes — handy for local dev (see the
  tiny-model example below).

### Model auto-download

The pinned `v0.4.0` fetches missing models on first use: each file is
downloaded from
`https://github.com/ARBO-TEAM/arbo-ocr-models/releases/download/models-v1/`,
SHA-256 verified, and written to a per-user cache, so the second run is
local. You get this from a default `Config` with no fields set — the two
below exist to redirect it or switch it off.

| Field | Flag | Default | What it does |
|---|---|---|---|
| `no_download` | `--no-download` | `false` — flag omitted entirely | Never fetch missing models; fail instead |
| `models_url` | `--models-url` | `None` — flag omitted entirely | Directory URL to fetch missing models from, e.g. an internal mirror |

`Engine::download_models()` runs `arboocr_demo --download-models`, which
fetches the models for the configured `ocr_version`/`model_type` and exits
without doing any OCR — for a CI step or Docker build layer that wants the
cache warm before the first request pays for it. It returns the binary's
per-file report (one `ok` / `skipped` / `absent` / `MISSING` line per file)
as a `String`; it is a convenience, not a prerequisite, since the binary
fetches on demand anyway.

**In a container, it is the other half of the install.**
`installer::ensure_installed` in a build step bakes in the *binary* and
nothing else — models live in a separate cache, so an image built that way
still downloads weights on the first request at runtime, which is exactly
where you didn't want to pay for it. Run both in the same build step:

```rust
use std::path::Path;
use arbo_ocr::{installer, Config, Engine};

let bin = installer::ensure_installed(Some(Path::new("/opt/arboocr")))?; // the binary
let engine = Engine::new(Config {
    bin_path: Some(bin),
    // models_url: Some("https://mirror.internal/arboocr/models/".to_string()),
    ..Default::default()
})?;
print!("{}", engine.download_models()?); // ...and the weights
```

Set `ARBOOCR_CACHE_DIR` to a path inside the image if the build and runtime
users differ, otherwise the runtime user gets its own empty cache and the
prefetch is wasted.

`arboocr_demo` is a child process and `std::process::Command` inherits the
parent's environment, so arboOCR's own environment variables work with no
config field involved:

| Env var | Effect |
|---|---|
| `ARBOOCR_OFFLINE=1` | Same as `no_download: true`, process-wide |
| `ARBOOCR_MODELS_URL` | Default base URL override — same role as `models_url` |
| `ARBOOCR_CACHE_DIR` | Override the per-user model cache directory |

The model cache, tag-scoped so a future `models-v2` can never reuse a
`models-v1` file:

| Platform | Model cache directory |
|---|---|
| Windows | `%LOCALAPPDATA%\arboOCR\models\models-v1` |
| macOS | `~/Library/Caches/arboOCR/models/models-v1` |
| Linux | `$XDG_CACHE_HOME/arboOCR/models/models-v1`, else `~/.cache/arboOCR/models/models-v1` |

That is arboOCR's *model* cache, distinct from this crate's *binary* cache
under `arbo-ocr-rust/<version>/<platform>` described in
[How it works](#how-it-works). macOS is listed for completeness:
`installer::detect_platform` only auto-installs Windows and Linux x64
binaries, so a macOS user supplies `Config.bin_path` themselves.

Resolution order per file: an explicit path (`det_model_path`,
`cls_model_path`, `rec_model_path`, `dict_path`) is never substituted by a
download → an existing file in `models_dir` wins, with zero network traffic
→ only then is the file downloaded and verified. A populated `models_dir`
therefore behaves exactly as it did before auto-download existed: same
files, same absence of network traffic.

## Usage

```rust
use arbo_ocr::{Config, Engine};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = Engine::new(Config {
        // Every field is optional. Models are fetched and cached on first
        // run; point models_dir at your own copy to skip that.
        // models_dir: Some("/path/to/models".to_string()),
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

### GPU (CUDA / TensorRT)

`use_cuda` / `use_tensorrt` pick the ONNXRuntime execution provider, and
`result.backend` reports which one actually ran. Both need the pinned
`v0.3.0` or later: every earlier archive shipped without the
`onnxruntime_providers_shared` library, so the CUDA and TensorRT providers
could not be loaded from a release archive at all. If you pass
`Config.bin_path` yourself, check it came from a `v0.3.0`-or-later archive
before concluding your GPU isn't being picked up.

### Tuning options

Beyond the model/provider fields above, `Config` exposes arboOCR's accuracy
and throughput knobs. Each is `Option` and omitted from the command line
when `None`, so leaving it unset keeps arboOCR's own default:

| Field | Flag | Default when `None` | What it does |
|---|---|---|---|
| `min_confidence` | `--min-confidence` | `0.5` | Drops lines below this recognition confidence; `Some(0.0)` disables the filter |
| `rec_batch_num` | `--rec-batch-num` | `6` | Crops per recognition inference call |
| `det_limit_side_len` | `--det-limit-side-len` | `960` | Longest image side for the detection resize |
| `log_level` | `--log-level` | silent | `"debug"`/`"info"`/`"warn"`/`"error"` — engine logs on stderr |
| `word_boxes` | `--word-boxes` | `false` | Also fill `LineResult::words` (see below) |

`word_boxes: true` adds a `WordBox { text, score, polygon }` per word to
each line (per *character* for CJK, which has no spaces to split on):

```rust
let engine = Engine::new(Config {
    models_dir: Some("/path/to/models".to_string()),
    word_boxes: true,
    ..Default::default()
})?;

for line in &engine.recognize("/path/to/image.jpg")?.lines {
    for word in &line.words {
        println!("{} ({:.3})", word.text, word.score);
    }
}
```

`line.words` is empty when `word_boxes` is off — arboOCR omits the key
entirely in that case, which is the normal shape.

### Many images in one process — `recognize_batch`

`recognize` starts a fresh `arboocr_demo` for every image, and the process
start plus model load dominates a short page. `recognize_batch` runs **one**
process over a whole list instead:

```rust
let pages = engine.recognize_batch(&[
    "/scans/001.jpg".to_string(),
    "/scans/002.jpg".to_string(),
    "/scans/003.jpg".to_string(),
])?;

for (i, page) in pages.iter().enumerate() {
    println!("{}: {} lines", page.image, page.lines.len());
    let _ = i; // pages[i] belongs to the i-th path passed in
}
```

Over 5 SROIE receipts the saving measured 13.0% of wall time at `tiny`, 28.5%
at `small` and 13.6% at `medium`, with identical text on every image
(`bench_batch_go.py` in the internal `compare/` harness). That share is
`(process start + model load) / total`, so it moves with the model size and
the list length rather than being a fixed percentage.

Results are matched to inputs **by position**, and the count must agree —
`arboocr_demo` reports only an image's basename, so two same-named files in
different directories would be indistinguishable. A mismatch is returned as an
error rather than a shifted list. For the same reason a path that cannot
survive the newline-delimited list format (empty, containing a newline, or
starting with `#`, which the binary reads as a comment and would skip) is
rejected before anything runs. One temporary list file is created per call and
removed before returning.

A batch exits `1` when *any* image came back with no text. That is an ordinary
outcome, not a failure, and is tolerated as long as the JSON array is still on
stdout — a usage error (unknown flag) exits `1` too but leaves stdout empty,
and that one is returned as an `OcrError`.

### Errors and exit codes

`OcrError::exit_code` carries `arboocr_demo`'s exit status:

| Code | Meaning |
|---|---|
| `0` | Success (an empty `lines` vec is still success) |
| `1` | Usage error, or no text found |
| `2` | Nothing usable ran — model load failed or recognition threw |

Exit code `2` most often means a model couldn't be loaded. Now that missing
files are fetched automatically, that usually means the fetch was blocked
rather than that you forgot to install anything: `no_download: true`,
`ARBOOCR_OFFLINE=1`, an unreachable `models_url`, or no network. The other
common cause is an explicit `det_model_path`/`rec_model_path`/`dict_path`
pointing at a file that isn't there — explicit paths are never substituted
by a download. Note that as of arboOCR v0.2.0 the
binary is **silent on stderr unless `--log-level` is passed**, so
`OcrError::stderr` will be empty by default — set `log_level:
Some("error".to_string())` when you need the engine to explain itself.

## Quick example (tiny model, fastest)

For a fast local smoke test, use `model_type: "tiny"` — the
smallest/fastest PP-OCRv6 recognizer. Dropping `models_dir` is fine here;
the tiny files are fetched and cached on first run. If you have an arboOCR
checkout handy, its `models/` folder already contains the tiny det/rec/cls
ONNX files, so pointing at it skips even that:

```rust
use arbo_ocr::{Config, Engine};

let engine = Engine::new(Config {
    models_dir: Some("/path/to/arboOCR/models".to_string()), // optional — a local arboOCR checkout's models/ dir
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
(`%LOCALAPPDATA%\arbo-ocr-rust\<version>\<platform>` on Windows,
`$XDG_CACHE_HOME/arbo-ocr-rust/<version>/<platform>` or
`~/.cache/arbo-ocr-rust/<version>/<platform>` on Linux). The pinned arboOCR
release tag is part of that path on purpose: the installer short-circuits
when the binary already exists, so without it an upgrade of this crate would
keep reusing whatever release the machine downloaded first and never fetch
the new one. Cache directories for older versions are left in place rather
than deleted. Call
`installer::ensure_installed(Some(dir))` yourself (e.g. in a container build
step) if you want to control exactly when the download happens, then pass
the returned path as `Config.bin_path`. That installs the *binary* only —
the models are a separate cache, so pair it with `Engine::download_models()`
in the same step or the image will still fetch weights on its first request;
see [Model auto-download](#model-auto-download).

OCR models are never bundled in the crate, but they are no longer a manual
step either. Whether they get *downloaded* is a property of the
`arboocr_demo` build being run, not of this wrapper, and the pinned `v0.4.0`
fetches a file it can't find, SHA-256 verifies it, and caches it per user —
in its own cache directory, separate from this crate's binary cache above.
A populated `models_dir` short-circuits that entirely. See
[Models](#models) for the file layout and the flags controlling it.

`recognize` captures the subprocess's output with `Command::output()`,
which reads stdout and stderr concurrently on separate threads internally —
`arboocr_demo` can write ~200KB of ONNXRuntime warnings to stderr before
any stdout appears, enough to deadlock a naive sequential-pipe-read
implementation, a class of bug the PHP and Go wrappers had to explicitly
design around.

Bool options (`use_angle_cls`, `use_cuda`, `use_tensorrt`, `use_fp16`,
`use_clahe`, `word_boxes`) are always sent as a single `--flag=value` token, never a bare
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
