use std::path::PathBuf;
use std::time::Duration;

use arbo_ocr::{Config, Engine};

fn fake_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_fake_arboocr"))
}

fn engine_with_fake_bin() -> Engine {
    Engine::new(Config {
        bin_path: Some(fake_bin()),
        ..Default::default()
    })
    .expect("Engine::new with fake bin")
}

#[test]
fn recognize_parses_successful_json() {
    let engine = engine_with_fake_bin();
    let result = engine.recognize("/some/page.jpg").expect("recognize");

    assert_eq!(result.backend, "cpu");
    assert_eq!(result.image, "page.jpg");
    assert_eq!(result.elapsed_ms, 12.5);
    assert_eq!(result.lines.len(), 1);
    assert_eq!(result.lines[0].text, "hello");
    assert_eq!(result.lines[0].score, 0.9);
    assert_eq!(result.lines[0].det_score, 0.8);
    assert_eq!(result.lines[0].polygon.len(), 1);
    assert_eq!(result.lines[0].polygon[0].x, 1.0);
    assert_eq!(result.lines[0].polygon[0].y, 2.0);
    // arboOCR omits "words" entirely unless --word-boxes is on; this must
    // deserialize to an empty vec, not fail. Guards LineResult's
    // #[serde(default)] against being dropped.
    assert!(result.lines[0].words.is_empty());
}

#[test]
fn recognize_parses_word_boxes_when_present() {
    let engine = engine_with_fake_bin();
    let result = engine.recognize("WORDS").expect("recognize");

    let words = &result.lines[0].words;
    assert_eq!(words.len(), 2);
    assert_eq!(words[0].text, "hi");
    assert_eq!(words[0].score, 0.95);
    assert_eq!(words[0].polygon.len(), 1);
    assert_eq!(words[1].text, "there");
    assert_eq!(words[1].polygon[0].x, 3.0);
}

#[test]
fn recognize_returns_error_on_non_zero_exit() {
    let engine = engine_with_fake_bin();

    let result = engine.recognize("FAIL");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.exit_code, Some(2));
    assert!(err.to_string().contains("exited with code"));
}

#[test]
fn recognize_returns_error_on_unparseable_output() {
    let engine = engine_with_fake_bin();
    let result = engine.recognize("GARBAGE");
    assert!(result.is_err());
}

#[test]
fn download_models_returns_the_binarys_report() {
    let engine = engine_with_fake_bin();
    let report = engine.download_models().expect("download_models");

    // The real binary prints one ok/skipped/absent/MISSING line per model
    // file; the wrapper's job is to hand that back verbatim, not parse it.
    assert!(report.contains("det"));
    assert!(report.contains("rec"));
    assert!(!report.contains("MISSING"));
}

#[test]
fn new_engine_errors_when_bin_path_missing() {
    let missing = std::env::temp_dir().join("no-such-arboocr-binary-xyz");
    let result = Engine::new(Config {
        bin_path: Some(missing),
        ..Default::default()
    });
    assert!(result.is_err());
}

#[test]
fn recognize_does_not_deadlock_on_large_stderr() {
    let engine = engine_with_fake_bin();

    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let result = engine.recognize("NOISY");
        let _ = tx.send(result);
    });

    match rx.recv_timeout(Duration::from_secs(10)) {
        Ok(Ok(result)) => {
            assert_eq!(result.backend, "cpu");
            assert_eq!(result.lines.len(), 1);
            assert_eq!(result.lines[0].text, "hello");
        }
        Ok(Err(e)) => panic!("recognize: {e}"),
        Err(_) => panic!("recognize deadlocked"),
    }
}

/// The batch error paths are selected by sentinel *paths*, not by argv: the
/// engine writes the list to a file rather than onto the command line, so a
/// sentinel flag would never reach the fake. `recognize_batch` passes them
/// through verbatim (they are ordinary paths to it), which is what makes them
/// usable as test hooks.
fn paths(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

#[test]
fn recognize_batch_parses_array_in_input_order() {
    // The fake echoes each list path back as that page's line text, so input
    // order is observable rather than assumed.
    let engine = engine_with_fake_bin();

    let pages = engine
        .recognize_batch(&paths(&["/a/one.jpg", "/b/two.jpg", "/c/three.jpg"]))
        .expect("recognize_batch");

    assert_eq!(
        pages.iter().map(|p| p.lines[0].text.as_str()).collect::<Vec<_>>(),
        ["/a/one.jpg", "/b/two.jpg", "/c/three.jpg"]
    );
    assert_eq!(pages[0].image, "one.jpg");
}

#[test]
fn recognize_batch_empty_input_makes_no_process() {
    let engine = engine_with_fake_bin();

    assert!(engine.recognize_batch(&[]).expect("empty batch").is_empty());
}

#[test]
fn recognize_batch_rejects_unlistable_path() {
    let engine = engine_with_fake_bin();

    for bad in ["", "/b/two\n.jpg", "#commented.jpg"] {
        let err = engine
            .recognize_batch(&paths(&["/a/one.jpg", bad]))
            .expect_err("unlistable path must be rejected");
        assert!(
            err.message.contains("image_paths[1]"),
            "unexpected message: {}",
            err.message
        );
    }
}

#[test]
fn recognize_batch_tolerates_exit1_with_json() {
    // Exit 1 because a page came back empty is an ordinary batch outcome, not
    // a failure — the array is still on stdout. The fake keys this off a
    // sentinel *path* the same way recognize's error paths key off an image.
    let engine = engine_with_fake_bin();
    let pages = engine
        .recognize_batch(&paths(&["/a/one.jpg", "--batch-exit1"]))
        .expect("exit 1 with JSON must be tolerated");

    assert_eq!(pages.len(), 2);
}

#[test]
fn recognize_batch_count_mismatch_is_fatal() {
    // Every check after this one is positional, so a short array has to fail
    // here rather than shift text onto the wrong file.
    let engine = engine_with_fake_bin();
    let err = engine
        .recognize_batch(&paths(&["/a/one.jpg", "--batch-short"]))
        .expect_err("count mismatch must be fatal");

    assert!(
        err.message.contains("cannot match results to inputs by position"),
        "unexpected message: {}",
        err.message
    );
}
