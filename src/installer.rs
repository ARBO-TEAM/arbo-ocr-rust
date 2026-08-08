//! Downloads and caches the prebuilt `arboocr_demo` binary from
//! wafik/ArboOCR's GitHub Releases. Kept separate from `engine` so "how to
//! get the binary" stays independently readable/testable from "how to run
//! it".

use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

/// The GitHub repository that publishes prebuilt `arboocr_demo` binaries as
/// release assets.
const REPO: &str = "wafik/ArboOCR";

/// The release tag this crate downloads. Same tag arbo-ocr-php and
/// arbo-ocr-go pin — the release binary itself is language-agnostic, so all
/// three packages track the same build.
///
/// TODO: bump this to the arboOCR release that adds model auto-download,
/// once it ships. Until then [`crate::Config::no_download`],
/// [`crate::Config::models_url`] and [`crate::Engine::download_models`] are
/// passthroughs to flags the installed binary does not have: v0.2.0 answers
/// an unknown option with a usage error and exit 1. The README's Models
/// section carries the same caveat and should lose it in the same commit.
const PINNED_VERSION: &str = "v0.2.0";

/// Returns `"windows-x64"` or `"linux-x64"` based on the compile-time
/// target, or `None` if unsupported.
pub fn detect_platform() -> Option<&'static str> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("windows", "x86_64") => Some("windows-x64"),
        ("linux", "x86_64") => Some("linux-x64"),
        _ => None,
    }
}

fn binary_name(platform: &str) -> &'static str {
    if platform == "windows-x64" {
        "arboocr_demo.exe"
    } else {
        "arboocr_demo"
    }
}

fn asset_name(platform: &str) -> &'static str {
    if platform == "windows-x64" {
        "arboocr-windows-x64.zip"
    } else {
        "arboocr-linux-x64.tar.gz"
    }
}

/// Best-effort user cache directory (no extra dependency): `%LOCALAPPDATA%`
/// on Windows, `$XDG_CACHE_HOME` or `~/.cache` on Linux.
fn user_cache_dir() -> Result<PathBuf, String> {
    if cfg!(windows) {
        std::env::var("LOCALAPPDATA")
            .map(PathBuf::from)
            .map_err(|_| "arbo_ocr: LOCALAPPDATA is not set".to_string())
    } else {
        if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
            return Ok(PathBuf::from(xdg));
        }
        std::env::var("HOME")
            .map(|home| Path::new(&home).join(".cache"))
            .map_err(|_| "arbo_ocr: HOME is not set".to_string())
    }
}

/// Makes sure the `arboocr_demo` binary exists locally, downloading it from
/// GitHub Releases if missing, and returns its absolute path. `bin_dir ==
/// None` means: use the default cache directory,
/// `<user_cache_dir>/arbo-ocr-rust/<PINNED_VERSION>/<platform>`. Returns an
/// error if the platform is unsupported or the download/extract fails.
pub fn ensure_installed(bin_dir: Option<&Path>) -> Result<PathBuf, String> {
    let platform = detect_platform().ok_or_else(|| {
        format!(
            "arbo_ocr: unsupported platform {}/{}; download a release manually from \
             https://github.com/{REPO}/releases and pass a bin_path explicitly",
            std::env::consts::OS,
            std::env::consts::ARCH
        )
    })?;

    let bin_dir = match bin_dir {
        Some(d) => d.to_path_buf(),
        // PINNED_VERSION is a path segment on purpose — do not "tidy" it out.
        // The is_file() check below short-circuits on "binary already exists",
        // so a version-less cache path makes a PINNED_VERSION bump a no-op for
        // everyone who ever ran an older release: the download URL changes, but
        // the stale binary still sits at the same path, so we return it and
        // never download the new one. Keying the directory by version means a
        // bump lands in a fresh empty directory and actually fetches.
        //
        // Old version directories are deliberately left in place rather than
        // cleaned up: a few MB of stale cache is a much smaller problem than
        // deletion logic quietly removing something a caller still points at.
        None => user_cache_dir()?
            .join("arbo-ocr-rust")
            .join(PINNED_VERSION)
            .join(platform),
    };

    let bin_path = bin_dir.join(binary_name(platform));
    if bin_path.is_file() {
        return Ok(bin_path); // already installed at this version
    }

    let asset = asset_name(platform);
    let url = format!("https://github.com/{REPO}/releases/download/{PINNED_VERSION}/{asset}");

    download_and_extract(&url, &bin_dir, asset)?;

    #[cfg(unix)]
    if platform == "linux-x64" {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&bin_path, fs::Permissions::from_mode(0o755))
            .map_err(|e| format!("arbo_ocr: could not make {} executable: {e}", bin_path.display()))?;
    }

    Ok(bin_path)
}

fn download_and_extract(url: &str, target_dir: &Path, asset: &str) -> Result<(), String> {
    fs::create_dir_all(target_dir)
        .map_err(|e| format!("arbo_ocr: could not create {}: {e}", target_dir.display()))?;

    let bytes = download(url)?;

    if asset.ends_with(".zip") {
        extract_zip(&bytes, target_dir)?;
    } else {
        extract_tar_gz(&bytes, target_dir)?;
    }

    flatten_single_subdir(target_dir)
}

fn download(url: &str) -> Result<Vec<u8>, String> {
    let resp = ureq::get(url)
        .timeout(std::time::Duration::from_secs(120))
        .call()
        .map_err(|e| format!("arbo_ocr: download failed: {url}: {e}"))?;

    let mut buf = Vec::new();
    resp.into_reader()
        .read_to_end(&mut buf)
        .map_err(|e| format!("arbo_ocr: could not save download from {url}: {e}"))?;
    Ok(buf)
}

fn extract_zip(bytes: &[u8], target_dir: &Path) -> Result<(), String> {
    let mut archive = zip::ZipArchive::new(io::Cursor::new(bytes))
        .map_err(|e| format!("arbo_ocr: could not open downloaded zip: {e}"))?;

    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .map_err(|e| format!("arbo_ocr: could not read zip entry: {e}"))?;
        let dest = safe_join(target_dir, entry.name())?;

        if entry.is_dir() {
            fs::create_dir_all(&dest).map_err(|e| format!("arbo_ocr: could not create {}: {e}", dest.display()))?;
            continue;
        }

        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("arbo_ocr: could not create {}: {e}", parent.display()))?;
        }
        let mut out = fs::File::create(&dest).map_err(|e| format!("arbo_ocr: could not create {}: {e}", dest.display()))?;
        io::copy(&mut entry, &mut out).map_err(|e| format!("arbo_ocr: could not extract {}: {e}", dest.display()))?;
    }
    Ok(())
}

fn extract_tar_gz(bytes: &[u8], target_dir: &Path) -> Result<(), String> {
    let gz = flate2::read::GzDecoder::new(io::Cursor::new(bytes));
    let mut archive = tar::Archive::new(gz);

    for entry in archive
        .entries()
        .map_err(|e| format!("arbo_ocr: could not read downloaded archive: {e}"))?
    {
        let mut entry = entry.map_err(|e| format!("arbo_ocr: could not read tar entry: {e}"))?;
        let name = entry
            .path()
            .map_err(|e| format!("arbo_ocr: could not read tar entry path: {e}"))?
            .to_string_lossy()
            .into_owned();
        let dest = safe_join(target_dir, &name)?;

        // tar crate's unpack_in already handles dirs/files; using entry
        // directly keeps this consistent with the safe_join guard above.
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("arbo_ocr: could not create {}: {e}", parent.display()))?;
        }
        entry
            .unpack(&dest)
            .map_err(|e| format!("arbo_ocr: could not extract {}: {e}", dest.display()))?;
    }
    Ok(())
}

/// Joins `name` onto `base_dir`, rejecting archive entries that would
/// escape it via ".." path segments ("zip slip").
fn safe_join(base_dir: &Path, name: &str) -> Result<PathBuf, String> {
    let dest = base_dir.join(name);
    let base = base_dir
        .canonicalize()
        .unwrap_or_else(|_| base_dir.to_path_buf());
    let dest_parent = dest.parent().unwrap_or(&dest).to_path_buf();
    let dest_check = dest_parent.canonicalize().unwrap_or(dest_parent);
    if dest_check != base && !dest_check.starts_with(&base) {
        return Err(format!("arbo_ocr: archive entry escapes target directory: {name}"));
    }
    Ok(dest)
}

/// Mirrors Installer.php's `flattenSingleSubdir` / arbo-ocr-go's
/// `flattenSingleSubdir`: release archives contain one top-level folder
/// (e.g. `arboocr-windows-x64/...`). Moves its contents up into
/// `target_dir` so callers get `target_dir/arboocr_demo` directly. Only
/// flattens if there's exactly one entry and it's a directory.
fn flatten_single_subdir(target_dir: &Path) -> Result<(), String> {
    let entries: Vec<_> = fs::read_dir(target_dir)
        .map_err(|e| format!("arbo_ocr: could not read {}: {e}", target_dir.display()))?
        .filter_map(|e| e.ok())
        .collect();

    if entries.len() != 1 || !entries[0].path().is_dir() {
        return Ok(());
    }

    let subdir = entries[0].path();
    let items: Vec<_> = fs::read_dir(&subdir)
        .map_err(|e| format!("arbo_ocr: could not read {}: {e}", subdir.display()))?
        .filter_map(|e| e.ok())
        .collect();

    for item in items {
        let old_path = item.path();
        let new_path = target_dir.join(item.file_name());
        fs::rename(&old_path, &new_path)
            .map_err(|e| format!("arbo_ocr: could not move {} to {}: {e}", old_path.display(), new_path.display()))?;
    }
    fs::remove_dir(&subdir).map_err(|e| format!("arbo_ocr: could not remove {}: {e}", subdir.display()))
}
